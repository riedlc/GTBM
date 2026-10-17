"""Variational EM for the constrained (game-theoretic) K-block SBM."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from constrained_sbm.baseline_sbm import (
    PROB_FLOOR,
    _e_step,
    _e_step_dc,
    _elbo,
    _elbo_dc,
    _normalize_pi,
    _suff_stats,
    _suff_stats_dc,
    _update_theta,
)
from constrained_sbm.model import (
    DEFAULT_PINNED_DIAG,
    M_to_B_K,
    _payoff_matrix,
    compute_M_K,
    n_free_params,
)
from constrained_sbm.utils import (
    directed_spectral_init,
    edge_count,
    get_logger,
    is_symmetric,
    kmeans_init,
    random_init,
    spectral_evec_inits,
    spectral_init,
)

PARAM_LO = 0.01
PARAM_HI = 0.99
BETA_LO = 0.1
BETA_HI = 20.0


def _neg_expected_ll(params, m, non_edges, freqs, n, num_edges,
                     beta_fixed, infer_beta, pinned_diag, gauge,
                     degree_corrected=False, directed=False):
    K = len(freqs)
    n_free = n_free_params(K)
    if infer_beta:
        free_params = params[:n_free]
        beta = float(params[n_free])
    else:
        free_params = params
        beta = beta_fixed
    free_params = np.clip(np.asarray(free_params, dtype=float), PARAM_LO, PARAM_HI)
    beta = float(np.clip(beta, BETA_LO, BETA_HI))

    M = compute_M_K(free_params, freqs, beta, pinned_diag, gauge=gauge)
    B = M_to_B_K(M, freqs, n, num_edges,
                  directed=directed, degree_corrected=degree_corrected)
    logB = np.log(np.maximum(B, PROB_FLOOR))
    log1mB = np.log(np.maximum(1.0 - B, PROB_FLOOR))
    # m and non_edges are directed-convention sufficient stats. For an
    # undirected graph the ordered sum double-counts and we halve; for a
    # directed graph each ordered pair is a distinct Bernoulli.
    scale = 1.0 if directed else 0.5
    ll = scale * (np.sum(m * logB) + np.sum(non_edges * log1mB))
    return -ll


def _optimize_payoffs(m, non_edges, freqs, n, num_edges,
                      beta_fixed, infer_beta, free_params0, beta0,
                      pinned_diag, gauge, n_starts, rng,
                      degree_corrected=False, directed=False):
    K = len(freqs)
    n_free = n_free_params(K)
    bounds_free = [(PARAM_LO, PARAM_HI)] * n_free
    if infer_beta:
        bounds = bounds_free + [(BETA_LO, BETA_HI)]
    else:
        bounds = bounds_free

    def pack(fp, b):
        return np.concatenate([fp, [b]]) if infer_beta else np.asarray(fp, dtype=float)

    x0_list = [pack(free_params0, beta0)]
    init_lo, init_hi = (0.05, 0.95) if gauge == "row_sum" else (0.1, 0.9)
    for _ in range(max(n_starts - 1, 0)):
        rand_free = rng.uniform(init_lo, init_hi, size=n_free)
        rand_beta = rng.uniform(0.5, 8.0) if infer_beta else beta0
        x0_list.append(pack(rand_free, rand_beta))

    best_fun = np.inf
    best_x = x0_list[0]
    for x0 in x0_list:
        try:
            res = minimize(
                _neg_expected_ll, x0,
                args=(m, non_edges, freqs, n, num_edges,
                      beta_fixed, infer_beta, pinned_diag, gauge,
                      degree_corrected, directed),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best_fun and np.isfinite(res.fun):
                best_fun = res.fun
                best_x = res.x
        except Exception:
            continue

    if infer_beta:
        free_params = np.array(best_x[:n_free], dtype=float)
        beta = float(best_x[n_free])
    else:
        free_params = np.array(best_x, dtype=float)
        beta = float(beta_fixed)
    return free_params, beta, -best_fun


def _initialize(A, init, seed, K=2, gauge_order=False):
    if init == "spectral":
        return spectral_init(A, n_classes=K, seed=seed)
    if init == "kmeans":
        return kmeans_init(A, n_classes=K, seed=seed)
    if init == "directed":
        return directed_spectral_init(A, n_classes=K, seed=seed,
                                      gauge_order=gauge_order)
    return random_init(A.shape[0], n_classes=K, seed=seed)


def _run_em(A, q, beta_init, infer_beta, pinned_diag, n_mstep_starts,
            max_iter, tol, logger, label, rng, degree_corrected=False,
            theta_max=100.0, gauge="pin_diag", directed=False):
    n = A.shape[0]
    num_edges = edge_count(A, directed=directed)
    K = q.shape[1]
    n_free = n_free_params(K)

    pi = _normalize_pi(q)
    init_lo, init_hi = (0.05, 0.95) if gauge == "row_sum" else (0.1, 0.9)
    free_params = rng.uniform(init_lo, init_hi, size=n_free)
    beta = float(beta_init)

    if degree_corrected:
        if directed:
            degree = (np.asarray(A.sum(axis=1)).ravel()
                      + np.asarray(A.sum(axis=0)).ravel()).astype(float)
        else:
            degree = np.asarray(A.sum(axis=1)).ravel().astype(float)
        theta = np.ones(n)
        kappa = (q * theta[:, None]).sum(axis=0)
    else:
        degree = None
        theta = None
        kappa = None

    # ---- Initial M-step from the structured q (M-step FIRST). ----
    # This translates the spectral / k-means structure directly into (b, c)
    # before any E-step has a chance to blur it.
    if degree_corrected:
        m, non_edges, total, kappa = _suff_stats_dc(A, q, theta)
    else:
        m, non_edges, total = _suff_stats(A, q)
    free_params, beta, _ = _optimize_payoffs(
        m, non_edges, pi, n, num_edges,
        beta_fixed=beta if not infer_beta else None,
        infer_beta=infer_beta,
        free_params0=free_params, beta0=beta,
        pinned_diag=pinned_diag,
        gauge=gauge,
        n_starts=n_mstep_starts,
        rng=rng,
        degree_corrected=degree_corrected,
        directed=directed,
    )
    M = compute_M_K(free_params, pi, beta, pinned_diag, gauge=gauge)
    B = M_to_B_K(M, pi, n, num_edges,
                  directed=directed, degree_corrected=degree_corrected)

    # The degree-corrected coordinate ascent is not strictly monotone (the
    # θ-renormalize step can lower the ELBO and the iteration may settle into a
    # small limit cycle), so track the best-ELBO iterate and return that rather
    # than the final one. Mirrors the free fitter so the comparison is fair.
    def _snapshot(elbo_val):
        snap = dict(q=q.copy(), pi=pi.copy(), m=m.copy(),
                    non_edges=non_edges.copy(), total=total.copy(),
                    free_params=np.asarray(free_params, float).copy(),
                    beta=float(beta), M=M.copy(), B=B.copy(),
                    elbo=float(elbo_val))
        if degree_corrected:
            snap["theta"] = theta.copy()
            snap["kappa"] = kappa.copy()
        return snap

    best_state = None

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
        else:
            q = _e_step(A, q, B, pi, directed=directed)
            pi = _normalize_pi(q)
            m, non_edges, total = _suff_stats(A, q)

        free_params, beta, _ = _optimize_payoffs(
            m, non_edges, pi, n, num_edges,
            beta_fixed=beta if not infer_beta else None,
            infer_beta=infer_beta,
            free_params0=free_params, beta0=beta,
            pinned_diag=pinned_diag,
            gauge=gauge,
            n_starts=n_mstep_starts,
            rng=rng,
            degree_corrected=degree_corrected,
            directed=directed,
        )
        M = compute_M_K(free_params, pi, beta, pinned_diag, gauge=gauge)
        B = M_to_B_K(M, pi, n, num_edges,
                      directed=directed, degree_corrected=degree_corrected)

        if degree_corrected:
            elbo = _elbo_dc(m, q, B, pi, theta, kappa, degree, directed=directed)
        else:
            elbo = _elbo(m, non_edges, q, B, pi, directed=directed)
        elbo_hist.append(elbo)
        # Only the degree-corrected ascent is non-monotone; track its best
        # iterate. The Bernoulli path is monotone and returns its final
        # iterate (handled after the loop).
        if degree_corrected and (best_state is None or elbo > best_state["elbo"]):
            best_state = _snapshot(elbo)
        if logger is not None:
            fp_str = ", ".join(f"{v:.3f}" for v in free_params)
            pi_str = ", ".join(f"{v:.3f}" for v in pi)
            extra = ""
            if degree_corrected:
                extra = f"  theta=[{theta.min():.3f},{theta.max():.3f}]"
            logger.info(
                f"{label}iter {it:3d}  ELBO={elbo:.4f}  "
                f"pi=({pi_str})  free=({fp_str})  beta={beta:.4f}{extra}"
            )
        if it > 0 and abs(elbo - prev_elbo) / max(abs(prev_elbo), 1.0) < tol:
            converged = True
            break
        prev_elbo = elbo

    if (not degree_corrected or best_state is None) and elbo_hist:
        # Bernoulli path (monotone): return the final iterate.
        best_state = _snapshot(elbo_hist[-1])
    elif best_state is None:
        # No iterations ran at all; fall back to the pre-loop state.
        best_state = _snapshot(-np.inf)
    q = best_state["q"]
    pi = best_state["pi"]
    m = best_state["m"]
    non_edges = best_state["non_edges"]
    total = best_state["total"]
    free_params = best_state["free_params"]
    beta = best_state["beta"]
    M = best_state["M"]
    B = best_state["B"]
    if degree_corrected:
        theta = best_state["theta"]
        kappa = best_state["kappa"]

    P = _payoff_matrix(free_params, K=K, pinned_diag=pinned_diag, gauge=gauge)
    result = {
        "K": K,
        "q": q,
        "pi": pi,
        "freqs": pi,
        "free_params": free_params,
        "pinned_diag": (np.asarray(pinned_diag) if pinned_diag is not None else None),
        "payoff_matrix": P,
        "gauge": gauge,
        "beta": beta,
        "M": M,
        "B": B,
        "elbo": best_state["elbo"],
        "elbo_history": elbo_hist,
        "converged": converged,
        "iterations": len(elbo_hist),
        "degree_corrected": degree_corrected,
        "directed": bool(directed),
    }
    if degree_corrected:
        result["theta"] = theta
        result["kappa"] = kappa
        result["theta_max"] = float(theta_max)
    # K=2 convenience keys (a, d included so callers don't have to branch on gauge)
    if K == 2:
        result["x"] = float(pi[0])
        result["b"] = float(free_params[1])  # P[1, 0]
        result["c"] = float(free_params[0])  # P[0, 1]
        result["a"] = float(P[0, 0])
        result["d"] = float(P[1, 1])
    return result


def fit(
    A,
    K=2,
    beta=3.0,
    max_iter=200,
    tol=1e-6,
    n_restarts=10,
    pinned_diag=None,
    n_mstep_starts=None,
    seed=0,
    verbose=False,
    degree_corrected=False,
    theta_max=100.0,
    gauge=None,
    directed=None,
    init_mode=None,
    n_spectral=1,
):
    """Fit the constrained game-theoretic SBM via variational EM.

    Args:
        A: scipy.sparse adjacency (n x n), binary, no self-loops. In
            undirected mode the matrix is symmetrized via `A.maximum(A.T)`;
            in directed mode it is used as-is.
        K: number of blocks (>=2). Default 2.
        beta: inverse temperature. Set to None to infer.
        max_iter, tol: per-restart EM controls.
        n_restarts: number of random initializations (best ELBO wins).
        pinned_diag: tuple of pinned diagonal payoffs, length K. Used only
            with gauge='pin_diag'. Defaults to (2/3, 1/3) for K=2 and
            (2/3, 1/2, 1/3) for K=3.
        n_mstep_starts: inner L-BFGS-B starts per M-step. Default 1 for K=2
            (warm-start only — the outer EM restarts already supply exploration)
            and 2 for K>=3 (one warm + one random, for the harder 6+D landscape).
        gauge: 'row_sum' or 'pin_diag'. Default: 'row_sum' for K=2,
            'pin_diag' for K>=3. Under 'row_sum' each
            row of the payoff matrix sums to 1 and diagonals are derived;
            under 'pin_diag' diagonals are pinned to `pinned_diag`.
        directed: None (default) auto-detects from symmetry of A. Pass True /
            False to override. In directed mode the recovered B is asymmetric,
            the E-step uses both in- and out-neighbours, and M→B drops the
            symmetrization step.
        init_mode: None (default) uses the standard spectral→kmeans→random
            restart schedule. "directed" uses a directionality-preserving SVD
            co-clustering init (`directed_spectral_init`) for every restart,
            with gauge ordering when gauge='pin_diag'. Use for non-assortative
            structure (cyclic / bipartite / core-periphery) where the
            symmetric-Laplacian spectral init carries no signal.
        seed: RNG seed.
        n_spectral: number of leading non-trivial Laplacian eigenvectors to seed
            restarts from (default 1 = the standard spectral->kmeans->random
            schedule). When >1, the first `n_spectral` restarts are seeded from
            distinct low eigenvectors (via `spectral_evec_inits`), then one
            kmeans, then random for the rest. Useful when the community axis is
            not the first non-trivial eigenvector (e.g. when the Fiedler vector
            tracks node degree). Random restarts still cover non-assortative
            basins; the best-ELBO restart is returned.
        degree_corrected: fit the Poisson DC-SBM with per-node theta.
        theta_max: upper clip on theta in DC mode.
        verbose: print per-iteration ELBO.

    Returns:
        dict with K-generic keys (q, freqs, free_params, M, B, elbo, ...)
        plus K=2 convenience keys (b, c, x, a, d) when K=2.
    """
    A = sp.csr_matrix(A)
    if directed is None:
        directed = not is_symmetric(A)
    if not directed:
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    if A.nnz > 0:
        A.data = np.ones_like(A.data)

    # Resolve gauge default per K
    if gauge is None:
        gauge = "row_sum" if K == 2 else "pin_diag"
    if gauge not in ("row_sum", "pin_diag"):
        raise ValueError(f"Unknown gauge: {gauge!r}")

    if gauge == "pin_diag":
        if pinned_diag is None:
            if K in DEFAULT_PINNED_DIAG:
                pinned_diag = DEFAULT_PINNED_DIAG[K]
            else:
                pinned_diag = tuple(np.linspace(2.0 / 3.0, 1.0 / 3.0, K))
        pinned_diag = tuple(float(p) for p in pinned_diag)
        assert len(pinned_diag) == K, "pinned_diag length must equal K"
    else:
        if pinned_diag is not None:
            # row_sum ignores pinned_diag; warn the user once
            import warnings
            warnings.warn(
                "pinned_diag is ignored when gauge='row_sum' (diagonals derived).",
                stacklevel=2,
            )
        pinned_diag = None

    if n_mstep_starts is None:
        # Given q the M-step objective is nearly unimodal in the payoffs. With
        # M-step-first initialization and (b, c) randomized per outer restart,
        # the outer loop provides the exploration; the inner step needs one
        # warm start. K=3's 6-D landscape can occasionally benefit from one
        # extra random start.
        n_mstep_starts = 1 if K == 2 else 2

    infer_beta = beta is None
    beta_init = 3.0 if beta is None else float(beta)

    logger = get_logger("constrained_sbm", verbose=verbose)
    rng = np.random.default_rng(seed)

    use_directed = init_mode == "directed"
    gauge_order = use_directed and gauge == "pin_diag"

    # Opt-in eigenvector-diverse spectral seeds (default n_spectral=1 -> none).
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
        sd = int(rng.integers(0, 1 << 31))
        if use_directed:
            init_kind = "directed"
            try:
                q0 = _initialize(A, init_kind, sd, K=K, gauge_order=gauge_order)
            except Exception:
                q0 = random_init(A.shape[0], K, seed=sd)
        elif n_spec > 0:
            # First n_spec restarts: eigenvector-diverse spectral seeds; then one
            # kmeans; then random. (Random restarts preserve non-assortative
            # exploration; best-ELBO selects across all.)
            if r < n_spec:
                init_kind = f"spectral_evec{r + 1}"
                q0 = evec_inits[r]
            else:
                init_kind = "kmeans" if r == n_spec else "random"
                try:
                    q0 = _initialize(A, init_kind, sd, K=K, gauge_order=gauge_order)
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
                q0 = _initialize(A, init_kind, sd, K=K, gauge_order=gauge_order)
            except Exception:
                q0 = random_init(A.shape[0], K, seed=sd)
        if verbose:
            logger.info(f"-- restart {r} ({init_kind}, seed={sd}) --")
        result = _run_em(
            A, q0,
            beta_init=beta_init,
            infer_beta=infer_beta,
            pinned_diag=pinned_diag,
            n_mstep_starts=n_mstep_starts,
            max_iter=max_iter,
            tol=tol,
            logger=logger if verbose else None,
            label=f"r{r} ",
            rng=np.random.default_rng(sd),
            degree_corrected=degree_corrected,
            theta_max=theta_max,
            gauge=gauge,
            directed=directed,
        )
        result["restart"] = r
        result["init"] = init_kind
        if best is None or result["elbo"] > best["elbo"]:
            best = result

    return best
