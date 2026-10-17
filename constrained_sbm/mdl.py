"""MDL (Minimum Description Length) and principled model comparison.

Four measures, all computed post-hoc from converged fits:

1. MDL-BIC      — Σ = -log L + L(θ̂) + (d/2) log|E|
2. MDL-Laplace  — Σ = -log L + L(θ̂) + (d/2) log(|E|/2π) + (1/2) log det I(φ̂)
3. Vuong test   — V = √n · mean(δ) / std(δ),  δ_ij = log p_c(A_ij) − log p_f(A_ij)
4. ∆ELPD_kfold  — mean of per-fold held-out test log-likelihood differences

All values in nats. The free SBM has an analytic diagonal Fisher; the
constrained model's Fisher (w.r.t. payoff parameters) is computed by central
finite differences on the hard-assignment negative log-likelihood.

References
----------
Vuong (1989); Kass & Raftery (1995); Peixoto (2017); Vehtari, Gelman & Gabry
(2017); Vehtari, Sivula & Bürkner (2024).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.stats import norm

from constrained_sbm.model import M_to_B_K, compute_M_K, n_free_params

EPS = 1e-12


# -----------------------------------------------------------------------------
# Sufficient statistics
# -----------------------------------------------------------------------------


def compute_sufficient_stats(A, z, K, directed=False):
    """Edge counts and max-edge counts at hard assignments.

    Undirected (default):
        m[r, r]    = #undirected edges within block r
        m[r, s]    = m[s, r] = #undirected edges between blocks r and s (r ≠ s)
        M[r, r]    = n_r · (n_r − 1) / 2  (unordered within-block pairs)
        M[r, s]    = n_r · n_s            (unordered cross-block pairs, r ≠ s)

    Directed:
        m[r, s]    = #directed edges (i → j) with z_i = r, z_j = s.  Asymmetric.
        M[r, r]    = n_r · (n_r − 1)      (ordered i ≠ j within block)
        M[r, s]    = n_r · n_s            (ordered cross-block pairs)
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]
    z = np.asarray(z).astype(int)
    Z = sp.csr_matrix(
        (np.ones(n), (np.arange(n), z)), shape=(n, K)
    )
    # directed count: ZAZ[r, s] = #ordered (i, j) with z_i = r, z_j = s and A_ij = 1
    ZAZ = (Z.T @ A @ Z).toarray()
    n_blocks = np.array([int((z == r).sum()) for r in range(K)], dtype=float)

    if directed:
        m = ZAZ.astype(float).copy()
        M_max = np.outer(n_blocks, n_blocks).astype(float)
        # ordered within-block pairs: n_r * (n_r - 1)
        np.fill_diagonal(M_max, n_blocks * (n_blocks - 1.0))
        return m, M_max, n_blocks

    m = np.zeros((K, K), dtype=float)
    for r in range(K):
        m[r, r] = ZAZ[r, r] / 2.0
        for s in range(r + 1, K):
            m[r, s] = ZAZ[r, s]
            m[s, r] = m[r, s]
    M_max = np.zeros((K, K), dtype=float)
    for r in range(K):
        M_max[r, r] = n_blocks[r] * (n_blocks[r] - 1.0) / 2.0
        for s in range(r + 1, K):
            M_max[r, s] = n_blocks[r] * n_blocks[s]
            M_max[s, r] = M_max[r, s]
    return m, M_max, n_blocks


# -----------------------------------------------------------------------------
# Log-likelihoods at hard assignments
# -----------------------------------------------------------------------------


def log_likelihood_bernoulli(B, m, M_max, K, directed=False):
    """Bernoulli SBM log-likelihood at hard assignments.

    Undirected:
        log L = Σ_{r ≤ s} m_rs · log B_rs + (M_rs − m_rs) · log(1 − B_rs)
        (each unordered pair counted once; m and M_max symmetric.)

    Directed:
        log L = Σ_{r, s} m_rs · log B_rs + (M_rs − m_rs) · log(1 − B_rs)
        (sum over ALL ordered (r, s); m and M_max may be asymmetric.)

    A numerical floor is applied to each log argument. Under the Bernoulli OR
    link B ≤ 1 by construction; under the degree-corrected link B is a rate
    that may exceed 1, in which case log(1 − B) is floored and this value is
    an approximation used only for the Fisher curvature.
    """
    logB = np.log(np.maximum(B, EPS))
    log1mB = np.log(np.maximum(1.0 - B, EPS))
    if directed:
        return float(np.sum(m * logB) + np.sum((M_max - m) * log1mB))
    ll = 0.0
    for r in range(K):
        ll += m[r, r] * logB[r, r] + (M_max[r, r] - m[r, r]) * log1mB[r, r]
        for s in range(r + 1, K):
            ll += m[r, s] * logB[r, s] + (M_max[r, s] - m[r, s]) * log1mB[r, s]
    return float(ll)


def log_likelihood_poisson_dc(A, z, B, theta, K, directed=False):
    """Poisson DC-SBM log-likelihood at hard assignments.

    Undirected:
        log L = Σ_{i<j} [A_ij · log(θ_i θ_j B_{z_i,z_j}) − θ_i θ_j B_{z_i,z_j}]
              = Σ_i d_i log θ_i
                + Σ_{r ≤ s} m_rs · log B_rs
                − (1/2) (Σ_{r,s} B_rs κ_r κ_s − Σ_i θ_i² B_{z_i,z_i})

    Directed:
        log L = Σ_{i ≠ j} [A_ij · log(θ_i θ_j B_{z_i,z_j}) − θ_i θ_j B_{z_i,z_j}]
              = Σ_i (d_out_i + d_in_i) log θ_i
                + Σ_{r, s} m_rs · log B_rs                   (all r, s)
                − (Σ_{r,s} B_rs κ_r κ_s − Σ_i θ_i² B_{z_i,z_i})
        (no 1/2 — every ordered (i, j) is its own Poisson observation; the
         degree term uses the total degree d_out + d_in for the same reason.)

    κ_r = Σ_{i: z_i = r} θ_i.
    """
    A = sp.csr_matrix(A)
    z = np.asarray(z).astype(int)
    theta = np.asarray(theta, dtype=float)
    Bc = np.clip(B, EPS, None)
    if directed:
        degree = (np.asarray(A.sum(axis=1)).ravel()
                  + np.asarray(A.sum(axis=0)).ravel()).astype(float)
    else:
        degree = np.asarray(A.sum(axis=1)).ravel().astype(float)
    log_theta = np.log(np.clip(theta, EPS, None))
    deg_term = float((degree * log_theta).sum())
    m, _, _ = compute_sufficient_stats(A, z, K, directed=directed)
    if directed:
        edge_term = float(np.sum(m * np.log(Bc)))
    else:
        edge_term = 0.0
        for r in range(K):
            edge_term += m[r, r] * np.log(Bc[r, r])
            for s in range(r + 1, K):
                edge_term += m[r, s] * np.log(Bc[r, s])
    kappa = np.array([theta[z == r].sum() for r in range(K)], dtype=float)
    rate_sum = float(np.sum(Bc * np.outer(kappa, kappa)))
    self_correction = float(np.sum((theta ** 2) * Bc[z, z]))
    scale = 1.0 if directed else 0.5
    rate_term = scale * (rate_sum - self_correction)
    return float(deg_term + edge_term - rate_term)


# -----------------------------------------------------------------------------
# Partition encoding cost (Peixoto 2017)
# -----------------------------------------------------------------------------


def partition_encoding_cost(n_blocks, N):
    """Cost of encoding the partition: log(N-1 choose K-1) + log(N!/Π n_r!)."""
    K = len(n_blocks)
    if K < 1 or N < 1:
        return 0.0
    # log C(N-1, K-1)
    log_choose = gammaln(N) - gammaln(K) - gammaln(N - K + 1)
    # log multinomial: log(N!) - Σ log(n_r!)
    log_multinom = gammaln(N + 1) - sum(gammaln(nr + 1) for nr in n_blocks)
    return float(log_choose + log_multinom)


# -----------------------------------------------------------------------------
# MDL: BIC and Laplace
# -----------------------------------------------------------------------------


def mdl_bic(neg_log_lik, partition_cost, d, n_edges):
    """Σ_BIC = -log L + L(θ̂) + (d/2) log|E|."""
    return float(neg_log_lik + partition_cost + 0.5 * d * np.log(max(n_edges, 1)))


def mdl_laplace(neg_log_lik, partition_cost, d, n_edges, log_det_fisher):
    """Σ_Laplace = -log L + L(θ̂) + (d/2) log(|E|/2π) + (1/2) log det I.

    Returns None if log_det_fisher is None (Fisher not PD — fall back to BIC).
    """
    if log_det_fisher is None:
        return None
    return float(
        neg_log_lik + partition_cost
        + 0.5 * d * np.log(max(n_edges, 1) / (2.0 * np.pi))
        + 0.5 * log_det_fisher
    )


# -----------------------------------------------------------------------------
# Fisher information
# -----------------------------------------------------------------------------


def fisher_free_sbm(B, M_max, K, directed=False):
    """Analytic log det Fisher information for the free Bernoulli SBM.

    Undirected: I is diagonal over the K(K+1)/2 upper-triangle B entries,
        I[rs, rs] = M_rs / (B_rs · (1 − B_rs)).
    Directed:   I is diagonal over all K² entries (B is asymmetric, every cell
        is its own free parameter).

    Returns log det I (a scalar). Returns None if any entry is degenerate.
    """
    Bc = np.clip(B, EPS, 1.0 - EPS)
    log_det = 0.0
    if directed:
        for r in range(K):
            for s in range(K):
                denom = Bc[r, s] * (1.0 - Bc[r, s])
                if M_max[r, s] <= 0 or denom <= 0:
                    return None
                log_det += np.log(M_max[r, s] / denom)
        return float(log_det)
    for r in range(K):
        for s in range(r, K):
            denom = Bc[r, s] * (1.0 - Bc[r, s])
            if M_max[r, s] <= 0 or denom <= 0:
                return None
            log_det += np.log(M_max[r, s] / denom)
    return float(log_det)


def fisher_constrained_numerical(neg_log_lik_fn, phi_hat, eps=1e-5):
    """Central finite-difference Hessian of `neg_log_lik_fn` at `phi_hat`.

    Returns (log_det_fisher, hessian, eigenvalues).  log_det_fisher is None
    if not PD (smallest eigenvalue ≤ 0 or any non-finite value). Callers can
    inspect `eigenvalues` to diagnose the failure mode.
    """
    phi = np.asarray(phi_hat, dtype=float)
    d = len(phi)
    H = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ei[i] = eps
            ej = np.zeros(d); ej[j] = eps
            fpp = neg_log_lik_fn(phi + ei + ej)
            fpm = neg_log_lik_fn(phi + ei - ej)
            fmp = neg_log_lik_fn(phi - ei + ej)
            fmm = neg_log_lik_fn(phi - ei - ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[j, i] = H[i, j]
    try:
        eigvals = np.linalg.eigvalsh(H)
    except np.linalg.LinAlgError:
        return None, H, np.array([])
    if np.any(eigvals <= 0) or not np.all(np.isfinite(eigvals)):
        return None, H, eigvals
    return float(np.sum(np.log(eigvals))), H, eigvals


def _make_constrained_neg_ll(m_hard, M_hard, freqs, n, num_edges,
                              pinned_diag, fixed_beta, infer_beta, K,
                              gauge="pin_diag", degree_corrected=False,
                              directed=False):
    """Closure: phi -> -log L(B(phi); hard-assignment stats) for the constrained model.

    If infer_beta, phi has length K(K-1)+1 with the last entry being β.
    Else phi has length K(K-1) and β is taken from fixed_beta.

    `degree_corrected` and `directed` are forwarded to `M_to_B_K` so the
    curvature is computed around the same link function the fit used.
    """
    nfp = n_free_params(K)

    def neg_ll(phi):
        phi = np.asarray(phi, dtype=float)
        if infer_beta:
            free_params = phi[:nfp]
            beta = float(phi[nfp])
        else:
            free_params = phi[:nfp]
            beta = float(fixed_beta)
        free_params = np.clip(free_params, 1e-3, 1.0 - 1e-3)
        beta = float(np.clip(beta, 0.05, 30.0))
        M_jt = compute_M_K(free_params, freqs, beta, pinned_diag, gauge=gauge)
        Bp = M_to_B_K(M_jt, freqs, n, num_edges,
                       directed=directed, degree_corrected=degree_corrected)
        return -log_likelihood_bernoulli(Bp, m_hard, M_hard, K, directed=directed)

    return neg_ll


# -----------------------------------------------------------------------------
# Vuong test
# -----------------------------------------------------------------------------


def vuong_test(A, z_constr, B_constr, z_free, B_free, K,
               d_constr=None, d_free=None, directed=False):
    """Vuong (1989) test of equivalence between two SBM fits.

    Block-grouped via combined labels (K² combined classes). Both models'
    pointwise log-likelihoods are written in terms of (m_rs, M_rs − m_rs)
    sufficient statistics over the joint label, then compared.

    Undirected mode (default): each unordered pair {i, j} is one observation;
    iterate over c1 ≤ c2 in the combined-label space.
    Directed mode: each ordered pair (i, j), i ≠ j, is one observation; B may
    be asymmetric, and we iterate over all (c1, c2).
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]
    z_c = np.asarray(z_constr).astype(int)
    z_f = np.asarray(z_free).astype(int)
    combined = z_c * K + z_f  # in 0..K²-1
    K2 = K * K
    C = sp.csr_matrix(
        (np.ones(n), (np.arange(n), combined)), shape=(n, K2)
    )
    # m[c1, c2] = #ordered (i,j) with combined-label (c1, c2) and A_ij=1
    CAC = (C.T @ A @ C).toarray()
    counts = np.array([int((combined == c).sum()) for c in range(K2)], dtype=float)

    Bc = np.clip(B_constr, EPS, 1.0 - EPS)
    Bf = np.clip(B_free, EPS, 1.0 - EPS)
    logBc = np.log(Bc)
    log1mBc = np.log(1.0 - Bc)
    logBf = np.log(Bf)
    log1mBf = np.log(1.0 - Bf)

    sum_delta = 0.0
    sum_delta_sq = 0.0
    n_pairs_total = 0.0

    if directed:
        # Every ordered (i, j), i ≠ j, is one Bernoulli observation. B may be
        # asymmetric, so B[rc, sc] differs from B[sc, rc]; no halving anywhere.
        for c1 in range(K2):
            rc, rf = divmod(c1, K)
            for c2 in range(K2):
                sc, sf = divmod(c2, K)
                edges = CAC[c1, c2]
                if c1 == c2:
                    max_edges = counts[c1] * (counts[c1] - 1.0)
                else:
                    max_edges = counts[c1] * counts[c2]
                non_edges = max_edges - edges
                if max_edges <= 0:
                    continue
                n_pairs_total += max_edges
                d_edge = (logBc[rc, sc] - logBf[rf, sf])
                d_noedge = (log1mBc[rc, sc] - log1mBf[rf, sf])
                sum_delta += edges * d_edge + non_edges * d_noedge
                sum_delta_sq += edges * d_edge ** 2 + non_edges * d_noedge ** 2
    else:
        for c1 in range(K2):
            rc, rf = divmod(c1, K)
            for c2 in range(K2):
                sc, sf = divmod(c2, K)
                # consider unordered pair (c1, c2): use c1 <= c2 only, with appropriate counting
                if c1 > c2:
                    continue
                if c1 == c2:
                    # diagonal: undirected pairs within this combined class
                    edges = CAC[c1, c2] / 2.0
                    max_edges = counts[c1] * (counts[c1] - 1.0) / 2.0
                else:
                    # off-diagonal: undirected pairs across (c1, c2)
                    edges = CAC[c1, c2]  # symmetric: equals CAC[c2, c1]
                    max_edges = counts[c1] * counts[c2]
                non_edges = max_edges - edges
                if max_edges <= 0:
                    continue
                n_pairs_total += max_edges
                d_edge = (logBc[rc, sc] - logBf[rf, sf])
                d_noedge = (log1mBc[rc, sc] - log1mBf[rf, sf])
                sum_delta += edges * d_edge + non_edges * d_noedge
                sum_delta_sq += edges * d_edge**2 + non_edges * d_noedge**2

    if n_pairs_total <= 0:
        return {
            "V": float("nan"), "V_corrected": None, "p_value": float("nan"),
            "mean_delta": float("nan"), "std_delta": float("nan"),
            "n_pairs": 0,
        }
    mean_d = sum_delta / n_pairs_total
    var_d = max(sum_delta_sq / n_pairs_total - mean_d ** 2, 0.0)
    std_d = float(np.sqrt(var_d))
    if std_d <= 0.0:
        return {
            "V": float("inf") if mean_d > 0 else (float("-inf") if mean_d < 0 else 0.0),
            "V_corrected": None,
            "p_value": 0.0 if abs(mean_d) > 0 else 1.0,
            "mean_delta": float(mean_d), "std_delta": 0.0,
            "n_pairs": int(n_pairs_total),
        }
    # Degeneracy guard: if both models predict essentially the same probabilities,
    # the pointwise differences are pure numerical noise and the Vuong statistic
    # (especially V_corrected with a nonzero parameter-count adjustment) is
    # undefined. Floor at 1e-5: well below any meaningful δ_ij (which scales with
    # log B differences, typically ≥ 1e-3) but well above floating-point noise.
    NOISE_FLOOR = 1e-5
    if std_d < NOISE_FLOOR:
        return {
            "V": 0.0, "V_corrected": 0.0, "p_value": 1.0,
            "mean_delta": float(mean_d), "std_delta": float(std_d),
            "n_pairs": int(n_pairs_total),
            "degenerate": True,
        }
    V = float(np.sqrt(n_pairs_total) * mean_d / std_d)
    V_corrected = None
    if d_constr is not None and d_free is not None:
        aic_term = abs(d_constr - d_free) / 2.0
        # The corrected statistic is only informative when the signal (sum_delta)
        # is comparable to or larger than the parameter-count adjustment. Otherwise
        # V_corrected is dominated by AIC bias and is misleading.
        if abs(sum_delta) > aic_term:
            adj = (sum_delta - (d_constr - d_free) / 2.0) / (np.sqrt(n_pairs_total) * std_d)
            V_corrected = float(adj)
    p_value = float(2.0 * norm.sf(abs(V)))
    return {
        "V": V, "V_corrected": V_corrected, "p_value": p_value,
        "mean_delta": float(mean_d), "std_delta": std_d,
        "n_pairs": int(n_pairs_total),
    }


# -----------------------------------------------------------------------------
# ∆ELPD_kfold
# -----------------------------------------------------------------------------


def compute_delta_elpd(per_fold_test_ll_constrained, per_fold_test_ll_free):
    """∆ELPD_kfold from per-fold held-out log-likelihoods."""
    cll = np.asarray(per_fold_test_ll_constrained, dtype=float)
    fll = np.asarray(per_fold_test_ll_free, dtype=float)
    F = len(cll)
    assert len(fll) == F, "fold count mismatch"
    deltas = cll - fll
    mean_d = float(deltas.mean())
    if F > 1:
        se = float(deltas.std(ddof=1)) / np.sqrt(F)
    else:
        se = 0.0
    if se > 0:
        z = mean_d / se
        p = float(2.0 * norm.sf(abs(z)))
        reliable = bool(abs(mean_d) > 4.0 * se)
    else:
        z = 0.0 if mean_d == 0.0 else float("inf") * np.sign(mean_d)
        p = 1.0 if mean_d == 0.0 else 0.0
        reliable = abs(mean_d) > 0.0
    return {
        "delta_elpd": mean_d,
        "se": se,
        "z": float(z),
        "p_value": p,
        "reliable": reliable,
        "per_fold_deltas": deltas.tolist(),
        "n_folds": int(F),
    }


# -----------------------------------------------------------------------------
# Parameter counts
# -----------------------------------------------------------------------------


def parameter_count_constrained(K, infer_beta=False, n_nodes=0,
                                  degree_corrected=False, directed=False):
    """The constrained payoff matrix has K(K−1) off-diagonal free entries in
    both undirected and directed mode — the partner-choice equations don't
    introduce extra parameters when the graph is directed. `directed` is
    accepted for symmetry with `parameter_count_free` and ignored here."""
    del directed  # unused; same payoff count in both modes
    d = K * (K - 1)
    if infer_beta:
        d += 1
    if degree_corrected:
        d += n_nodes
    return d


def parameter_count_free(K, n_nodes=0, degree_corrected=False, directed=False):
    """Number of free B entries in the unconstrained SBM.

    Undirected: K(K+1)/2 (symmetric — upper triangle incl. diagonal).
    Directed:   K²        (asymmetric — every cell is its own free parameter).
    DC adds n_nodes per-node θ entries on top.
    """
    if directed:
        d = K * K
    else:
        d = K * (K + 1) // 2
    if degree_corrected:
        d += n_nodes
    return d


# -----------------------------------------------------------------------------
# Top-level: assemble all measures
# -----------------------------------------------------------------------------


def compute_model_comparison(
    A, result_constrained, result_free, K,
    infer_beta=False, degree_corrected=False,
    directed=None,
):
    """Compute MDL (BIC + Laplace) and Vuong for a constrained vs free fit pair.

    `directed=None` (default) infers the mode from the fit result dicts —
    `result_constrained["directed"]` if present, else from A's symmetry. Pass
    True/False to override. In directed mode the free-SBM parameter count is
    K² (vs K(K+1)/2 undirected), the sufficient statistics keep their full
    asymmetry, and the likelihood / Fisher sums run over every ordered pair.

    Returns a flat dict suitable for both pretty-printing and JSON dump.
    """
    A = sp.csr_matrix(A)
    n_nodes = A.shape[0]

    if directed is None:
        directed = bool(result_constrained.get("directed",
                                                result_free.get("directed",
                                                                 (A != A.T).nnz != 0)))
    n_edges = int(A.nnz) if directed else int(A.nnz // 2)

    # Hard assignments
    z_constr = np.asarray(result_constrained["q"]).argmax(axis=1)
    z_free = np.asarray(result_free["q"]).argmax(axis=1)

    # Sufficient statistics (directed → asymmetric m, M)
    m_c, M_c, n_blocks_c = compute_sufficient_stats(A, z_constr, K, directed=directed)
    m_f, M_f, n_blocks_f = compute_sufficient_stats(A, z_free, K, directed=directed)

    # Log-likelihoods at hard assignments
    B_constr = np.asarray(result_constrained["B"])
    B_free = np.asarray(result_free["B"])
    if degree_corrected:
        theta_c = np.asarray(result_constrained["theta"])
        theta_f = np.asarray(result_free["theta"])
        ll_c = log_likelihood_poisson_dc(A, z_constr, B_constr, theta_c, K,
                                           directed=directed)
        ll_f = log_likelihood_poisson_dc(A, z_free, B_free, theta_f, K,
                                           directed=directed)
    else:
        ll_c = log_likelihood_bernoulli(B_constr, m_c, M_c, K, directed=directed)
        ll_f = log_likelihood_bernoulli(B_free, m_f, M_f, K, directed=directed)

    # Partition costs (different hard partitions for each model)
    L_part_c = partition_encoding_cost(n_blocks_c, n_nodes)
    L_part_f = partition_encoding_cost(n_blocks_f, n_nodes)

    # Parameter counts
    d_c = parameter_count_constrained(K, infer_beta=infer_beta, n_nodes=n_nodes,
                                       degree_corrected=degree_corrected,
                                       directed=directed)
    d_f = parameter_count_free(K, n_nodes=n_nodes,
                                degree_corrected=degree_corrected,
                                directed=directed)

    # MDL-BIC
    sigma_bic_c = mdl_bic(-ll_c, L_part_c, d_c, n_edges)
    sigma_bic_f = mdl_bic(-ll_f, L_part_f, d_f, n_edges)

    # Fisher: free is analytic, constrained is numerical
    log_det_fisher_f = fisher_free_sbm(B_free, M_f, K, directed=directed)
    # Build the constrained neg-log-lik closure for the Fisher
    freqs_c = np.asarray(result_constrained["pi"])
    pd_raw = result_constrained.get("pinned_diag")
    pinned_diag = np.asarray(pd_raw) if pd_raw is not None else None
    gauge_c = result_constrained.get("gauge", "pin_diag")
    fp_hat = np.asarray(result_constrained["free_params"])
    if infer_beta:
        phi_hat = np.concatenate([fp_hat, [float(result_constrained["beta"])]])
    else:
        phi_hat = fp_hat
    neg_ll_fn = _make_constrained_neg_ll(
        m_c, M_c, freqs_c, n_nodes, n_edges, pinned_diag,
        fixed_beta=float(result_constrained["beta"]),
        infer_beta=infer_beta, K=K, gauge=gauge_c,
        degree_corrected=degree_corrected,
        directed=directed,
    )
    log_det_fisher_c, _, fisher_eigvals_c = fisher_constrained_numerical(neg_ll_fn, phi_hat)
    empty_blocks_c = [int(k) for k in range(K) if n_blocks_c[k] == 0]

    # MDL-Laplace
    sigma_lap_c = mdl_laplace(-ll_c, L_part_c, d_c, n_edges, log_det_fisher_c)
    sigma_lap_f = mdl_laplace(-ll_f, L_part_f, d_f, n_edges, log_det_fisher_f)
    laplace_valid = sigma_lap_c is not None and sigma_lap_f is not None

    # Vuong test — Bernoulli pointwise log-likelihoods (both models)
    vuong = vuong_test(A, z_constr, B_constr, z_free, B_free, K,
                       d_constr=d_c, d_free=d_f, directed=directed)

    out = {
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "directed": bool(directed),
        "log_lik_constrained": ll_c,
        "log_lik_free": ll_f,
        "partition_cost_constrained": L_part_c,
        "partition_cost_free": L_part_f,
        "d_constrained": d_c,
        "d_free": d_f,
        "sigma_bic_constrained": sigma_bic_c,
        "sigma_bic_free": sigma_bic_f,
        "delta_bic": sigma_bic_c - sigma_bic_f,
        "log_det_fisher_constrained": log_det_fisher_c,
        "log_det_fisher_free": log_det_fisher_f,
        "fisher_eigvals_constrained": (
            fisher_eigvals_c.tolist() if hasattr(fisher_eigvals_c, "tolist") else fisher_eigvals_c
        ),
        "empty_blocks_constrained": empty_blocks_c,
        "sigma_laplace_constrained": sigma_lap_c,
        "sigma_laplace_free": sigma_lap_f,
        "delta_laplace": (sigma_lap_c - sigma_lap_f) if laplace_valid else None,
        "laplace_valid": laplace_valid,
        "vuong": vuong,
    }
    return out


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def kass_raftery_category(abs_delta):
    """Map |ΔΣ| (nats) to a Kass & Raftery 1995 verbal scale."""
    if abs_delta < 1.0:
        return "negligible"
    if abs_delta < 3.0:
        return "weak"
    if abs_delta < 5.0:
        return "moderate"
    if abs_delta < 10.0:
        return "strong"
    return "decisive"


def verdict(delta_value, abs_value):
    """Return a human-readable verdict from a ΔΣ value.

    Negative ΔΣ → constrained preferred; positive → free preferred.
    """
    if abs_value < 1.0:
        return "indistinguishable"
    category = kass_raftery_category(abs_value)
    side = "game-theoretic" if delta_value < 0 else "free"
    return f"{side} ({category})"
