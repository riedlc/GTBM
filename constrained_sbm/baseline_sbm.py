"""Free (unconstrained) K-block variational SBM as comparison baseline.

The E-step is shared with the constrained variant. The M-step has a
closed-form MLE: B[k,l] = m[k,l] / total[k,l].
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from constrained_sbm.utils import (
    directed_spectral_init,
    get_logger,
    kmeans_init,
    random_init,
    spectral_evec_inits,
    spectral_init,
)

PROB_FLOOR = 1e-10
PROB_CEIL = 1.0 - 1e-10
THETA_MIN = 0.01
THETA_MAX = 100.0


def _e_step(A, q, B, pi, directed=False):
    """Mean-field update of q given current B and prior pi.

    Computes the log unnormalized responsibilities per node, then softmaxes.
    Uses the sparse-efficient form with non-edge term derived from global
    soft counts N_l = sum_j q_j(l).

    For undirected graphs (default), only the out-edge contribution
    (`S = A @ q`, paired with logB.T) is computed — each undirected edge
    {i, j} appears once in i's neighbor list and once in j's, so the per-node
    update accumulates one contribution per incident undirected edge.

    For directed graphs, both out- and in-edge contributions are computed
    separately: an edge (i → j) contributes log B[z_i, z_j] to i's update and
    a separate edge (j → i) contributes log B[z_j, z_i] to i's update. The
    two are not interchangeable when A or B is asymmetric.
    """
    # Floor only: B may legitimately exceed 1 under the Poisson DC link, so it
    # is not clipped from above.
    logB = np.log(np.maximum(B, PROB_FLOOR))
    log1mB = np.log(np.maximum(1.0 - B, PROB_FLOOR))
    logpi = np.log(np.clip(pi, PROB_FLOOR, 1.0))

    N = q.sum(axis=0)  # (K,)
    S_out = A @ q      # (n, K)  — j with A_ij = 1

    edge_out = S_out @ logB.T                    # log B[k, z_j] for out-edges
    nonedge_global_out = (N[None, :] - q) @ log1mB.T
    nonedge_correct_out = S_out @ log1mB.T
    log_q = logpi[None, :] + edge_out + nonedge_global_out - nonedge_correct_out

    if directed:
        S_in = A.T @ q                           # j with A_ji = 1
        edge_in = S_in @ logB                    # log B[z_j, k] for in-edges
        nonedge_global_in = (N[None, :] - q) @ log1mB
        nonedge_correct_in = S_in @ log1mB
        log_q = log_q + edge_in + nonedge_global_in - nonedge_correct_in

    log_q -= log_q.max(axis=1, keepdims=True)
    q_new = np.exp(log_q)
    q_new /= q_new.sum(axis=1, keepdims=True)
    return q_new


def _suff_stats(A, q):
    """Expected edge / non-edge counts under q. Directed convention.

    Returns m and total such that:
        m[k,l]     = expected directed edges (k,l)
        total[k,l] = expected ordered pairs (k,l) excluding self
    """
    # A @ q gives sum over neighbors; q.T @ (A @ q) is the directed m matrix
    Aq = A @ q
    m = q.T @ Aq  # (K, K), symmetric for symmetric A

    N = q.sum(axis=0)  # (K,)
    # sum_i q_i(k) * q_i(l)
    self_term = q.T @ q  # (K, K)
    total = np.outer(N, N) - self_term

    non_edges = total - m
    # numerical safety
    m = np.maximum(m, 0.0)
    non_edges = np.maximum(non_edges, 0.0)
    return m, non_edges, total


def _elbo(m, non_edges, q, B, pi, directed=False):
    logB = np.log(np.maximum(B, PROB_FLOOR))
    log1mB = np.log(np.maximum(1.0 - B, PROB_FLOOR))
    logpi = np.log(np.clip(pi, PROB_FLOOR, 1.0))

    # m and non_edges are directed-convention sums. For an undirected graph
    # the underlying log-likelihood is over unordered pairs and the directed
    # sum double-counts, so we halve; for a directed graph each ordered pair
    # is a distinct Bernoulli and the full sum is the correct log-likelihood.
    scale = 1.0 if directed else 0.5
    ll = scale * (np.sum(m * logB) + np.sum(non_edges * log1mB))
    prior = float(np.sum(q * logpi[None, :]))
    entropy = -float(np.sum(q * np.log(np.clip(q, PROB_FLOOR, 1.0))))
    return ll + prior + entropy


# -----------------------------------------------------------------------------
# Degree-corrected (Poisson SBM) variants
# -----------------------------------------------------------------------------


def _e_step_dc(A, q, B, pi, theta, directed=False):
    """Poisson DC E-step.

    Undirected:
        log q(z_i = k) ∝ log π_k
            + Σ_{j ∈ N(i)} Σ_l q_j(l) · log B[k,l]   (edge term)
            - θ_i · Σ_l B[k,l] · κ_l                  (expected-degree term)

    Directed: out-edges and in-edges contribute separate edge / rate terms,
    using B and B.T respectively.

    Terms involving log θ_i and log θ_j are constant in k and drop after softmax.
    """
    logB = np.log(np.maximum(B, PROB_FLOOR))
    logpi = np.log(np.clip(pi, PROB_FLOOR, 1.0))

    kappa = (q * theta[:, None]).sum(axis=0)  # (K,)
    S_out = A @ q
    edge_out = S_out @ logB.T
    rate_out = theta[:, None] * (B @ kappa)[None, :]
    log_q = logpi[None, :] + edge_out - rate_out

    if directed:
        S_in = A.T @ q
        edge_in = S_in @ logB
        rate_in = theta[:, None] * (B.T @ kappa)[None, :]
        log_q = log_q + edge_in - rate_in

    log_q -= log_q.max(axis=1, keepdims=True)
    q_new = np.exp(log_q)
    q_new /= q_new.sum(axis=1, keepdims=True)
    return q_new


def _suff_stats_dc(A, q, theta):
    """Directed-convention edge / non-edge sufficient stats with degree correction.

    Same structure as `_suff_stats`, but the non-edge mass uses κ_k = Σ_i q_i(k) θ_i
    in place of N_k. For sparse graphs this approximates the Poisson DC-SBM.
    """
    Aq = A @ q
    m = q.T @ Aq  # observed edges are unchanged
    qt = q * theta[:, None]
    kappa = qt.sum(axis=0)  # (K,)
    self_term_dc = qt.T @ qt  # (K, K)
    total = np.outer(kappa, kappa) - self_term_dc
    non_edges = total - m
    m = np.maximum(m, 0.0)
    non_edges = np.maximum(non_edges, 0.0)
    return m, non_edges, total, kappa


def _elbo_dc(m, q, B, pi, theta, kappa, degree, directed=False):
    """Poisson DC ELBO.

    ELBO = scale · (Σ m · log B  -  Σ B · κ⊗κ)            # Poisson likelihood
         + Σ_i Σ_k q_i(k) · log π_k                        # prior
         - Σ_i Σ_k q_i(k) · log q_i(k)                     # entropy
         + Σ_i d_i · log θ_i                               # degree term

    `scale` is 0.5 for undirected graphs (m double-counts each undirected
    edge under the directed-sufficient-stat convention) and 1.0 for directed.
    """
    logB = np.log(np.maximum(B, PROB_FLOOR))
    logpi = np.log(np.clip(pi, PROB_FLOOR, 1.0))

    rate_sum = float(np.sum(B * np.outer(kappa, kappa)))
    scale = 1.0 if directed else 0.5
    ll = scale * (float(np.sum(m * logB)) - rate_sum)
    prior = float(np.sum(q * logpi[None, :]))
    entropy = -float(np.sum(q * np.log(np.clip(q, PROB_FLOOR, 1.0))))
    degree_term = float(np.sum(degree * np.log(np.clip(theta, PROB_FLOOR, None))))
    return ll + prior + entropy + degree_term


def _update_theta(q, B, degree, theta_prev, theta_max=THETA_MAX, theta_min=THETA_MIN,
                   directed=False):
    """Closed-form Poisson DC-SBM θ update, then normalize so κ_k = N_k.

    Undirected:
        θ_i = d_i / Σ_k q_i(k) · Σ_l B[k,l] · κ_l
    Directed (using total degree d_i = d_out_i + d_in_i):
        θ_i = d_i / Σ_k q_i(k) · (Σ_l (B[k,l] + B[l,k]) · κ_l)

    Renormalize: scale each θ_i so Σ_i q_i(k) θ_i = N_k per class.
    """
    kappa_prev = (q * theta_prev[:, None]).sum(axis=0)  # (K,)
    if directed:
        expected_rate = q @ (B @ kappa_prev) + q @ (B.T @ kappa_prev)
    else:
        expected_rate = q @ (B @ kappa_prev)
    theta = degree / np.maximum(expected_rate, 1e-12)
    theta = np.clip(theta, theta_min, theta_max)

    kappa_new = (q * theta[:, None]).sum(axis=0)
    N = q.sum(axis=0)
    scale_per_class = N / np.maximum(kappa_new, 1e-12)
    theta = theta * (q @ scale_per_class)
    theta = np.clip(theta, theta_min, theta_max)
    return theta


def _free_B_mle_dc(m, kappa, directed=False):
    """Closed-form MLE for free DC-SBM: B[k,l] = m[k,l] / (κ_k · κ_l).

    For undirected graphs B is symmetrized; for directed graphs the
    asymmetric MLE passes through.
    """
    denom = np.outer(kappa, kappa)
    B = m / np.maximum(denom, 1e-12)
    if not directed:
        B = 0.5 * (B + B.T)
    return np.maximum(B, PROB_FLOOR)


def _free_B_mle(m, total, directed=False):
    """Closed-form MLE B[k,l] = m[k,l] / total[k,l].

    For undirected graphs m and total are symmetric, so the symmetrization
    is a no-op up to floating-point round-off.
    For directed graphs, the asymmetric MLE is returned unchanged.
    """
    B = m / np.maximum(total, 1e-12)
    if not directed:
        B = 0.5 * (B + B.T)
    return np.maximum(np.minimum(B, 1.0), PROB_FLOOR)


def _initialize(A, init, seed, K=2):
    n = A.shape[0]
    if init == "spectral":
        return spectral_init(A, n_classes=K, seed=seed)
    if init == "kmeans":
        return kmeans_init(A, n_classes=K, seed=seed)
    if init == "directed":
        # Free SBM is label-permutation invariant, so no gauge ordering.
        return directed_spectral_init(A, n_classes=K, seed=seed,
                                      gauge_order=False)
    return random_init(n, n_classes=K, seed=seed)


def _normalize_pi(q):
    pi = q.mean(axis=0)
    pi = np.clip(pi, 1e-3, 1.0 - 1e-3)
    pi /= pi.sum()
    return pi


def _run_em(A, q, max_iter, tol, logger, label="", degree_corrected=False,
            theta_max=THETA_MAX, directed=False, init_theta=None):
    n = A.shape[0]
    pi = _normalize_pi(q)

    if degree_corrected:
        # Degree counts: out + in for directed (so θ scales the total expected
        # rate, in + out); the single sum is correct for undirected.
        if directed:
            degree = (np.asarray(A.sum(axis=1)).ravel()
                      + np.asarray(A.sum(axis=0)).ravel()).astype(float)
        else:
            degree = np.asarray(A.sum(axis=1)).ravel().astype(float)
        # init_theta lets a caller seed θ (e.g. a warm start from the
        # constrained fit) so the pre-loop ELBO is evaluated at that exact
        # (q, θ) — see the best-iterate tracking below.
        theta = np.ones(n) if init_theta is None else np.asarray(init_theta, float).copy()
        m, non_edges, total, kappa = _suff_stats_dc(A, q, theta)
        B = _free_B_mle_dc(m, kappa, directed=directed)
        init_elbo = _elbo_dc(m, q, B, pi, theta, kappa, degree, directed=directed)
    else:
        degree = None
        theta = None
        kappa = None
        m, non_edges, total = _suff_stats(A, q)
        B = _free_B_mle(m, total, directed=directed)
        init_elbo = _elbo(m, non_edges, q, B, pi, directed=directed)

    # The degree-corrected coordinate ascent is NOT monotone (the θ-renormalize
    # step can lower the ELBO and the iteration may settle into a small limit
    # cycle), so the final iterate is not necessarily the best one. Track the
    # best-ELBO iterate — seeded with the pre-loop (B-MLE) state — and return
    # that. Seeding the tracker with the pre-loop state also preserves a warm
    # start whose initial ELBO is higher than any later iterate.
    def _snapshot(elbo_val):
        snap = dict(q=q.copy(), pi=pi.copy(), B=B.copy(), m=m.copy(),
                    non_edges=non_edges.copy(), total=total.copy(),
                    elbo=float(elbo_val))
        if degree_corrected:
            snap["theta"] = theta.copy()
            snap["kappa"] = kappa.copy()
        return snap

    best_state = _snapshot(init_elbo)

    elbo_hist = []
    prev_elbo = -np.inf
    converged = False
    for it in range(max_iter):
        if degree_corrected:
            q = _e_step_dc(A, q, B, pi, theta, directed=directed)
            pi = _normalize_pi(q)
            theta = _update_theta(q, B, degree, theta, theta_max=theta_max,
                                  directed=directed)
            m, non_edges, total, kappa = _suff_stats_dc(A, q, theta)
            B = _free_B_mle_dc(m, kappa, directed=directed)
            elbo = _elbo_dc(m, q, B, pi, theta, kappa, degree, directed=directed)
        else:
            q = _e_step(A, q, B, pi, directed=directed)
            pi = _normalize_pi(q)
            m, non_edges, total = _suff_stats(A, q)
            B = _free_B_mle(m, total, directed=directed)
            elbo = _elbo(m, non_edges, q, B, pi, directed=directed)
        elbo_hist.append(elbo)
        # Only the degree-corrected ascent is non-monotone; track its best
        # iterate. The Bernoulli path is monotone and returns its final iterate.
        if degree_corrected and elbo > best_state["elbo"]:
            best_state = _snapshot(elbo)
        if logger is not None:
            freq_str = ", ".join(f"{p:.3f}" for p in pi)
            extra = ""
            if degree_corrected:
                extra = f"  theta=[{theta.min():.3f},{theta.max():.3f}]"
            logger.info(f"{label}iter {it:3d}  ELBO={elbo:.4f}  pi=({freq_str}){extra}")
        if it > 0 and abs(elbo - prev_elbo) / max(abs(prev_elbo), 1.0) < tol:
            converged = True
            break
        prev_elbo = elbo

    # Bernoulli path: return the final iterate (monotone EM).
    if not degree_corrected and elbo_hist:
        best_state = _snapshot(elbo_hist[-1])

    result = {
        "q": best_state["q"],
        "pi": best_state["pi"],
        "B": best_state["B"],
        "m": best_state["m"],
        "non_edges": best_state["non_edges"],
        "total": best_state["total"],
        "elbo": best_state["elbo"],
        "elbo_history": elbo_hist,
        "converged": converged,
        "iterations": len(elbo_hist),
        "degree_corrected": degree_corrected,
    }
    if degree_corrected:
        result["theta"] = best_state["theta"]
        result["kappa"] = best_state["kappa"]
        result["theta_max"] = float(theta_max)
    return result


def fit_free(
    A,
    K=2,
    max_iter=200,
    tol=1e-6,
    n_restarts=10,
    seed=0,
    verbose=False,
    degree_corrected=False,
    theta_max=THETA_MAX,
    directed=None,
    init_mode=None,
    n_spectral=1,
    warm_start_q=None,
    warm_start_theta=None,
):
    """Fit a free K-block SBM via mean-field variational EM.

    When degree_corrected=True, uses the Poisson DC-SBM (Karrer & Newman 2011)
    with per-node sociability parameters θ. Otherwise the standard Bernoulli
    free SBM.

    `directed=None` auto-detects from the symmetry of A. Pass True / False to
    override. In directed mode the recovered B is asymmetric, the E-step uses
    both in- and out-neighbours, and the likelihood/ELBO drops the 0.5 factor
    that compensates for double-counting in the undirected case.

    `init_mode="directed"` uses the directionality-preserving SVD co-clustering
    init for every restart (None = standard spectral→kmeans→random schedule).
    `warm_start_q` / `warm_start_theta`: optional (n, K) soft assignment and (n,)
    theta (e.g. a constrained fit's converged state) used as one extra restart.
    Identical to the constrained fitter so model comparisons share an init.

    `n_spectral>1` mirrors the constrained fitter: the first `n_spectral`
    restarts are seeded from distinct leading Laplacian eigenvectors (via
    `spectral_evec_inits`), then one kmeans, then random. Combined with the
    matched seed sequence this keeps the two models' restart schedules aligned
    so the comparison is fair. Ignored in directed mode.
    """
    from constrained_sbm.utils import is_symmetric

    A = sp.csr_matrix(A)
    if directed is None:
        directed = not is_symmetric(A)
    if not directed:
        # Symmetrize so the sufficient statistics are exact for a near-symmetric
        # input.
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    if A.nnz > 0:
        A.data = np.ones_like(A.data)

    logger = get_logger("free_sbm", verbose=verbose)
    rng = np.random.default_rng(seed)

    use_directed = init_mode == "directed"

    # Opt-in eigenvector-diverse spectral seeds, mirroring the constrained
    # fitter (n_spectral=1 uses the standard spectral->kmeans->random schedule).
    evec_inits = []
    if n_spectral and n_spectral > 1 and not use_directed:
        try:
            evec_inits = spectral_evec_inits(A, n_classes=K, n_evecs=n_spectral,
                                             seed=seed)
        except Exception:
            evec_inits = []
    n_spec = len(evec_inits)

    best = None
    for r in range(n_restarts):
        # Draw the per-restart seed first (matches the constrained fitter's
        # consume-once-per-restart order so both share the same seed sequence).
        sd = int(rng.integers(0, 1 << 31))
        if use_directed:
            init_kind = "directed"
            try:
                q0 = _initialize(A, init_kind, sd, K=K)
            except Exception:
                q0 = random_init(A.shape[0], K, seed=sd)
        elif n_spec > 0:
            # First n_spec restarts: eigenvector-diverse spectral seeds; then
            # one kmeans; then random.
            if r < n_spec:
                init_kind = f"spectral_evec{r + 1}"
                q0 = evec_inits[r]
            else:
                init_kind = "kmeans" if r == n_spec else "random"
                try:
                    q0 = _initialize(A, init_kind, sd, K=K)
                except Exception:
                    q0 = random_init(A.shape[0], K, seed=sd)
        else:
            if r == 0:
                init_kind = "spectral"
            elif r == 1 and n_restarts > 2:
                init_kind = "kmeans"
            else:
                init_kind = "random"
            try:
                q0 = _initialize(A, init_kind, sd, K=K)
            except Exception:
                q0 = random_init(A.shape[0], K, seed=sd)
        if verbose:
            logger.info(f"-- restart {r} ({init_kind}, seed={sd}) --")
        result = _run_em(
            A, q0, max_iter, tol, logger if verbose else None, label=f"r{r} ",
            degree_corrected=degree_corrected, theta_max=theta_max,
            directed=directed,
        )
        result["restart"] = r
        result["init"] = init_kind
        result["K"] = K
        result["directed"] = bool(directed)
        if best is None or result["elbo"] > best["elbo"]:
            best = result

    # Warm start from a supplied soft assignment (typically the constrained
    # fit's converged q, θ). Because the free B is the unconstrained MLE, the
    # free ELBO at the constrained model's (q, θ) is >= the constrained ELBO by
    # construction, so (with the best-iterate tracking in _run_em) the free
    # fit starts from a point whose ELBO is already >= the constrained ELBO.
    if warm_start_q is not None:
        q0 = np.asarray(warm_start_q, float).copy()
        if q0.ndim == 2 and q0.shape == (A.shape[0], K):
            q0 = np.clip(q0, PROB_FLOOR, None)
            q0 /= q0.sum(axis=1, keepdims=True)
            wtheta = None
            if degree_corrected and warm_start_theta is not None:
                wtheta = np.asarray(warm_start_theta, float)
            result = _run_em(
                A, q0, max_iter, tol, logger if verbose else None,
                label="warm ", degree_corrected=degree_corrected,
                theta_max=theta_max, directed=directed, init_theta=wtheta,
            )
            result["restart"] = "warm"
            result["init"] = "warm_constrained"
            result["K"] = K
            result["directed"] = bool(directed)
            if best is None or result["elbo"] > best["elbo"]:
                best = result

    return best
