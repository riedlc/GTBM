"""Gibbs sampling for the FREE (unconstrained) Stochastic Block Model.

Parallel to `constrained_sbm.posterior` but the K×K block-to-block edge matrix
B is sampled directly from its conjugate posterior instead of being derived
from payoff parameters via M_to_B. No MH step is needed — every full-conditional
is conjugate:

    z       | rest   :  categorical (Poisson-DC form, same as constrained)
    π       | z      :  Dirichlet
    θ       | rest   :  Gamma (DC only; same as constrained)
    B[k,l]  | rest   :  Beta(α + edges, β + non_edges)              (non-DC)
                        Gamma(α + edges, β + Σ θ_iθ_j)               (DC)

Init from `baseline_sbm.fit_free` to warm-start the chain. Returns a result
dict shaped like `fit_gibbs` (same keys), so downstream code (the CLI, WAIC,
etc.) can consume it interchangeably.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from constrained_sbm.baseline_sbm import fit_free as fit_free_em
from constrained_sbm.posterior import (
    _kappa,
    effective_sample_size,
    r_hat,
    sample_pi,
    sample_theta,
    sample_z,
)

EPS = 1e-12
PROB_FLOOR = 1e-10
PROB_CEIL = 1.0 - 1e-10


# -----------------------------------------------------------------------------
# Sufficient statistics in undirected convention
# -----------------------------------------------------------------------------


def _undirected_suff_stats(A_csr, z, K, degree_corrected, theta=None):
    """Compute symmetric K×K matrices of *undirected* counts.

    Returns:
        m_und[k, l]     : number of undirected edges with endpoints in (k, l)
        non_edges_und   : non-edges in the same convention (for non-DC)
        exposure_und    : Σ θ_i θ_j over undirected (i, j) pairs (for DC)

    For k == l: within-class pairs, counted once each (i < j).
    For k != l: cross-class pairs, counted once each.

    Notes:
        Z.T @ A @ Z gives directed counts. For symmetric A, off-diagonal entries
        of that product equal the undirected cross-class count exactly (one
        unordered cross-pair {i, j} with z_i=k, z_j=l contributes exactly +1 to
        the (k, l) cell — the reverse tuple goes to (l, k)). Diagonal entries
        equal 2× the undirected within-class count (each within-edge contributes
        to both (i, j) and (j, i)). So we halve the diagonal only.
    """
    n = A_csr.shape[0]
    Z = sp.csr_matrix(
        (np.ones(n), (np.arange(n), z)), shape=(n, K)
    )
    m_dir = (Z.T @ A_csr @ Z).toarray()
    m_und = m_dir.copy()
    np.fill_diagonal(m_und, np.diag(m_dir) / 2.0)
    m_und = np.maximum(m_und, 0.0)

    N = np.bincount(z, minlength=K).astype(float)
    total_und = np.outer(N, N).astype(float)
    np.fill_diagonal(total_und, N * (N - 1) / 2.0)
    # off-diagonal currently N_k N_l which already equals the undirected
    # cross-class pair count (each unordered pair counted once).

    non_edges_und = np.maximum(total_und - m_und, 0.0)

    exposure_und = None
    if degree_corrected and theta is not None:
        # Σ_{i<j, z_i=k, z_j=l (or l, k)} θ_i θ_j   (symmetric)
                # For k != l: Σ_{i in k, j in l} θ_i θ_j (each unordered pair once) = κ_k κ_l
        # For k == l: ½ (κ_k² − Σ_{z_i=k} θ_i²)
        kappa = _kappa(z, theta, K)
        theta_sq_by_class = np.array(
            [np.sum(theta[z == k] ** 2) for k in range(K)],
            dtype=float,
        )
        exposure_und = np.outer(kappa, kappa)
        np.fill_diagonal(
            exposure_und, 0.5 * (kappa ** 2 - theta_sq_by_class)
        )
        exposure_und = np.maximum(exposure_und, 0.0)

    return m_und, non_edges_und, exposure_und


def _directed_suff_stats(A_csr, z, K, degree_corrected, theta=None):
    """Asymmetric K×K matrices of *directed* counts.

    Returns:
        m_dir[k, l]     : #ordered (i, j) with A_ij=1, z_i=k, z_j=l
        non_edges_dir   : #ordered (i, j), i≠j, A_ij=0, z_i=k, z_j=l (for non-DC)
        exposure_dir    : Σ_{ordered i≠j} θ_i θ_j over (z_i=k, z_j=l) (for DC)
    """
    n = A_csr.shape[0]
    Z = sp.csr_matrix(
        (np.ones(n), (np.arange(n), z)), shape=(n, K)
    )
    m_dir = (Z.T @ A_csr @ Z).toarray()
    m_dir = np.maximum(m_dir, 0.0)

    N = np.bincount(z, minlength=K).astype(float)
    total_dir = np.outer(N, N).astype(float)
    # Within-class ordered pairs i ≠ j: N_k * (N_k - 1)
    np.fill_diagonal(total_dir, N * (N - 1))
    non_edges_dir = np.maximum(total_dir - m_dir, 0.0)

    exposure_dir = None
    if degree_corrected and theta is not None:
        kappa = _kappa(z, theta, K)
        theta_sq_by_class = np.array(
            [np.sum(theta[z == k] ** 2) for k in range(K)],
            dtype=float,
        )
        exposure_dir = np.outer(kappa, kappa)
        # Within-class ordered: Σ_{i≠j, both in k} θ_i θ_j = κ_k² − Σ_{i in k} θ_i²
        np.fill_diagonal(
            exposure_dir, kappa ** 2 - theta_sq_by_class
        )
        exposure_dir = np.maximum(exposure_dir, 0.0)

    return m_dir, non_edges_dir, exposure_dir


# -----------------------------------------------------------------------------
# Conjugate B sampler (free SBM)
# -----------------------------------------------------------------------------


def sample_B_free(m_und, non_edges_und, exposure_und, K, alpha_B, beta_B,
                  rng, degree_corrected, directed=False):
    """Sample B (K, K) from independent per-cell conjugate posteriors.

    Undirected (default): B is symmetric — sample upper-triangle then mirror.
    Directed: every (k, l) cell is independent; B is in general asymmetric.

    - Non-DC (Bernoulli SBM): B[k,l] ~ Beta(α_B + edges, β_B + non_edges).
    - DC (Poisson SBM): B[k,l] ~ Gamma(α_B + edges, β_B + Σ θ_iθ_j).
      Here B is the block-rate factor; per-pair rate = θ_i θ_j B[z_i, z_j].
    """
    B = np.empty((K, K), dtype=float)
    if directed:
        if degree_corrected:
            for k in range(K):
                for l in range(K):
                    shape = alpha_B + m_und[k, l]
                    rate = beta_B + exposure_und[k, l]
                    samp = rng.gamma(shape, 1.0 / max(rate, EPS))
                    B[k, l] = float(max(samp, PROB_FLOOR))
        else:
            for k in range(K):
                for l in range(K):
                    a = alpha_B + m_und[k, l]
                    b = beta_B + non_edges_und[k, l]
                    samp = float(np.clip(rng.beta(a, b), PROB_FLOOR, PROB_CEIL))
                    B[k, l] = samp
        return B
    if degree_corrected:
        for k in range(K):
            for l in range(k, K):
                shape = alpha_B + m_und[k, l]
                rate = beta_B + exposure_und[k, l]
                samp = rng.gamma(shape, 1.0 / max(rate, EPS))
                samp = float(max(samp, PROB_FLOOR))
                B[k, l] = samp
                B[l, k] = samp
    else:
        for k in range(K):
            for l in range(k, K):
                a = alpha_B + m_und[k, l]
                b = beta_B + non_edges_und[k, l]
                samp = float(np.clip(rng.beta(a, b), PROB_FLOOR, PROB_CEIL))
                B[k, l] = samp
                B[l, k] = samp
    return B


# -----------------------------------------------------------------------------
# Single-chain sampler
# -----------------------------------------------------------------------------


def posterior_sample_free(
    A, K=2, degree_corrected=False, theta_max=100.0,
    alpha_pi=1.0, alpha_theta=1.0,
    alpha_B=1.0, beta_B=1.0,
    n_samples=1500, burn_in=750, thin=2,
    em_restarts=3, em_max_iter=200, em_tol=1e-6,
    random_seed=0, verbose=False,
    init_state=None, directed=False, init_mode=None,
):
    """Single Gibbs chain for the free SBM. Conjugate updates throughout.

    If `init_state` is provided (keys: `z`, `B`, `pi`, optional `theta`), the
    chain starts from that exact state and the per-chain EM warm-start is
    skipped. Lets a caller start all chains from one shared EM state for
    conditional posterior inference.
    """
    rng = np.random.default_rng(random_seed)
    A_csr = sp.csr_matrix(A)
    n = A_csr.shape[0]
    if directed:
        A_csc = A_csr.tocsc()
        degree = (np.asarray(A_csr.sum(axis=1)).ravel()
                  + np.asarray(A_csr.sum(axis=0)).ravel()).astype(float)
    else:
        A_csc = None
        degree = np.asarray(A_csr.sum(axis=1)).ravel().astype(float)

    if init_state is not None:
        z = np.asarray(init_state["z"], dtype=int).copy()
        pi = np.asarray(init_state.get("pi", np.full(K, 1.0 / K)),
                        dtype=float).copy()
        if degree_corrected:
            theta_arr = init_state.get("theta")
            theta = (np.asarray(theta_arr, dtype=float).copy()
                     if theta_arr is not None else np.ones(n, dtype=float))
        else:
            theta = np.ones(n, dtype=float)
        B = np.asarray(init_state["B"], dtype=float).copy()
        if not degree_corrected:
            B = np.clip(B, PROB_FLOOR, PROB_CEIL)
    else:
        em = fit_free_em(
            A_csr, K=K, n_restarts=em_restarts, max_iter=em_max_iter, tol=em_tol,
            degree_corrected=degree_corrected, theta_max=theta_max,
            seed=random_seed, verbose=False, directed=directed,
            init_mode=init_mode,
        )
        z = em["q"].argmax(axis=1).astype(int)
        pi = np.asarray(em["pi"], dtype=float).copy()
        theta = (np.asarray(em["theta"], dtype=float).copy()
                 if degree_corrected else np.ones(n, dtype=float))
        B = np.asarray(em["B"], dtype=float).copy()
        if not degree_corrected:
            B = np.clip(B, PROB_FLOOR, PROB_CEIL)

    samples = {
        "z": np.zeros((n_samples, n), dtype=np.int32),
        "B": np.zeros((n_samples, K, K), dtype=float),
        "pi": np.zeros((n_samples, K), dtype=float),
    }
    if degree_corrected:
        samples["theta"] = np.zeros((n_samples, n), dtype=float)

    total_iter = burn_in + n_samples * thin
    sample_idx = 0

    for t in range(total_iter):
        is_burn_in = t < burn_in

        z = sample_z(A_csr, z, B, theta, pi, K, rng,
                      directed=directed, A_csc=A_csc)
        pi = sample_pi(z, K, alpha_pi, rng)
        if degree_corrected:
            theta = sample_theta(degree, z, B, theta, K, alpha_theta, rng,
                                  directed=directed)

        if directed:
            m_stat, ne_stat, exp_stat = _directed_suff_stats(
                A_csr, z, K, degree_corrected, theta
            )
        else:
            m_stat, ne_stat, exp_stat = _undirected_suff_stats(
                A_csr, z, K, degree_corrected, theta
            )
        B = sample_B_free(
            m_stat, ne_stat, exp_stat, K, alpha_B, beta_B, rng,
            degree_corrected, directed=directed,
        )

        if not is_burn_in and (t - burn_in) % thin == 0 and sample_idx < n_samples:
            samples["z"][sample_idx] = z
            samples["B"][sample_idx] = B
            samples["pi"][sample_idx] = pi
            if degree_corrected:
                samples["theta"][sample_idx] = theta
            sample_idx += 1

        if verbose and (t + 1) % 200 == 0:
            phase = "burn-in" if is_burn_in else "sampling"
            print(f"  iter {t+1}/{total_iter} ({phase})")

    samples["n_samples"] = int(sample_idx)
    return samples


# -----------------------------------------------------------------------------
# Label alignment across samples and chains (Stephens 2000)
# -----------------------------------------------------------------------------
#
# The free SBM posterior has a K! exact symmetry: permuting (z, B, π) by the
# same σ leaves the joint density unchanged. Across chains (and even within a
# long chain), the sampler can settle into different elements of this
# permutation orbit, which inflates R-hat / ESS / WAIC nonsensically.
#
# Stephens' algorithm finds a canonical labeling by iteratively maximizing
# agreement with the empirical posterior assignment probabilities
#     q[i, k] = (1/S) Σ_s 1{z_s[i] = k}.
# For each sample, the best σ maximizes Σ_i q[i, σ(z_s[i])] = Σ_k C[k, σ(k)]
# with cost matrix C[k, l] = Σ_{i: z_s[i] = k} q[i, l]. Reapplying σ to (z, B, π)
# updates q; iterate until no sample's σ changes.
#
# Consensus-based alignment is more robust than aligning every sample to a
# single reference sample (brittle when that sample is a fluke).


def _build_consensus_q(chains, K, n):
    """Empirical posterior assignment matrix q (n, K) from current labelings."""
    q = np.zeros((n, K), dtype=float)
    S_total = 0
    for c in chains:
        S = c["z"].shape[0]
        S_total += S
        for k in range(K):
            q[:, k] += (c["z"] == k).sum(axis=0)
    if S_total > 0:
        q /= S_total
    return q


def _best_perm_to_consensus(z_s, q, K, perm_iter):
    """Find σ that maximizes Σ_k C[k, σ(k)] for sample's z under consensus q."""
    C = np.zeros((K, K))
    for k in range(K):
        mask = z_s == k
        if mask.any():
            C[k] = q[mask].sum(axis=0)
    identity = tuple(range(K))
    best_perm = identity
    best_score = float(np.trace(C))
    for sigma in perm_iter:
        score = float(sum(C[k, sigma[k]] for k in range(K)))
        if score > best_score + 1e-12:
            best_score = score
            best_perm = sigma
    return best_perm


def align_samples_stephens(chains, K, max_iters=20):
    """Stephens (2000) iterative relabeling over all chains and samples.

    Modifies each chain dict in place (z, B, π). θ is per-node and is not
    permuted. Returns (n_iters, last_changes) for diagnostics.
    """
    if K == 1:
        return 0, 0
    n = chains[0]["z"].shape[1]
    identity = tuple(range(K))
    perm_iter = [p for p in itertools.permutations(range(K)) if p != identity]

    last_changes = 0
    for it in range(max_iters):
        q = _build_consensus_q(chains, K, n)
        changes = 0
        for chain in chains:
            S = chain["z"].shape[0]
            for s in range(S):
                best_perm = _best_perm_to_consensus(chain["z"][s], q, K, perm_iter)
                if best_perm == identity:
                    continue
                sigma_arr = np.array(best_perm, dtype=int)
                chain["z"][s] = sigma_arr[chain["z"][s]]
                chain["B"][s] = chain["B"][s][np.ix_(best_perm, best_perm)]
                chain["pi"][s] = chain["pi"][s][list(best_perm)]
                changes += 1
        last_changes = changes
        if changes == 0:
            return it + 1, 0
    return max_iters, last_changes


# -----------------------------------------------------------------------------
# Multi-chain top-level
# -----------------------------------------------------------------------------


def _run_free_chain(kwargs):
    """Module-level worker for ProcessPoolExecutor — must be picklable."""
    return posterior_sample_free(**kwargs)


def fit_free_gibbs(
    A, K=2,
    degree_corrected=False, theta_max=100.0,
    n_samples=1500, burn_in=750, thin=2,
    n_chains=2, em_restarts=3, em_max_iter=200, em_tol=1e-6,
    alpha_pi=1.0, alpha_theta=1.0, alpha_B=1.0, beta_B=1.0,
    seed=0, verbose=False,
    init_state=None, directed=None, n_jobs=1, init_mode=None,
):
    """Multi-chain Gibbs estimator for the free SBM.

    Returns a result dict with keys parallel to `fit_gibbs` (constrained):
    `q`, `pi`, `B`, `theta` (DC), plus `posterior_chains` (list of per-chain
    samples dicts with `z`, `B`, `pi`, optional `theta`) for downstream WAIC.

    `directed=None` auto-detects from A's symmetry. When directed, the B
    samples are asymmetric (every K² cell is conditionally independent).
    """
    from constrained_sbm.utils import is_symmetric

    A_csr = sp.csr_matrix(A)
    if directed is None:
        directed = not is_symmetric(A_csr)
    if not directed:
        A_csr = A_csr.maximum(A_csr.T)
    A_csr.setdiag(0)
    A_csr.eliminate_zeros()
    if A_csr.nnz > 0:
        A_csr.data = np.ones_like(A_csr.data)
    n = A_csr.shape[0]

    from constrained_sbm.utils import run_chains_parallel

    parallel = n_jobs is not None and n_jobs > 1 and n_chains > 1
    # Suppress per-chain verbose under parallel execution to avoid interleaved
    # output; the launch banner below summarizes instead.
    chain_verbose = verbose and not parallel
    chain_args_list = [
        dict(
            A=A_csr, K=K,
            degree_corrected=degree_corrected, theta_max=theta_max,
            alpha_pi=alpha_pi, alpha_theta=alpha_theta,
            alpha_B=alpha_B, beta_B=beta_B,
            n_samples=n_samples, burn_in=burn_in, thin=thin,
            em_restarts=em_restarts, em_max_iter=em_max_iter, em_tol=em_tol,
            random_seed=seed + c, verbose=chain_verbose,
            init_state=init_state, directed=directed, init_mode=init_mode,
        )
        for c in range(n_chains)
    ]
    if parallel:
        if verbose:
            print(f"--- launching {n_chains} free chains across "
                  f"{min(n_jobs, n_chains)} worker process(es) "
                  f"(seeds {seed}..{seed + n_chains - 1}) ---")
        chains = run_chains_parallel(_run_free_chain, chain_args_list, n_jobs)
    else:
        chains = []
        for c, args in enumerate(chain_args_list):
            if verbose:
                print(f"--- free chain {c+1}/{n_chains} (seed={seed + c}) ---")
            chains.append(_run_free_chain(args))

    # The free SBM has a K! relabeling symmetry that no part of the sampler
    # breaks (unlike the constrained sampler where pinned diagonals / row-sum
    # gauge act as a tie-breaker). Run Stephens (2000) iterative relabeling
    # over ALL chains and samples *before* computing R-hat / ESS / WAIC.
    stephens_iters, stephens_remaining = align_samples_stephens(
        chains, K, max_iters=30
    )
    if verbose:
        print(f"Stephens alignment: {stephens_iters} iters "
              f"(remaining changes: {stephens_remaining})")

    stacked = {
        "z": np.concatenate([c["z"] for c in chains], axis=0),
        "B": np.concatenate([c["B"] for c in chains], axis=0),
        "pi": np.concatenate([c["pi"] for c in chains], axis=0),
    }
    if degree_corrected:
        stacked["theta"] = np.concatenate([c["theta"] for c in chains], axis=0)

    # Posterior summaries
    B_mean = stacked["B"].mean(axis=0)
    pi_mean = stacked["pi"].mean(axis=0)
    assign = np.zeros((n, K), dtype=float)
    for k in range(K):
        assign[:, k] = (stacked["z"] == k).mean(axis=0)
    theta_mean = (stacked["theta"].mean(axis=0)
                  if degree_corrected else None)

    # Per-parameter R-hat / ESS over the unique B entries.
    # Undirected: upper-triangle incl. diagonal (K(K+1)/2 cells).
    # Directed:   all K² cells, since B is asymmetric.
    if directed:
        iu_k, iv_k = np.indices((K, K))
        iu_k = iu_k.ravel()
        iv_k = iv_k.ravel()
    else:
        iu_k, iv_k = np.triu_indices(K, k=0)
    rhat_B, ess_B = [], []
    for k, l in zip(iu_k, iv_k):
        per_chain = [c["B"][:, k, l] for c in chains]
        rhat_B.append(r_hat(per_chain))
        ess_B.append(effective_sample_size(np.concatenate(per_chain)))

    result = {
        "K": K,
        "q": assign,
        "pi": pi_mean,
        "B": B_mean,
        "degree_corrected": degree_corrected,
        "directed": bool(directed),
        "estimator": "free_gibbs",
        "posterior_chains": chains,
        "posterior_stacked": stacked,
        "n_chains": int(n_chains),
        "n_samples_per_chain": int(chains[0]["n_samples"]),
        "rhat_B": rhat_B,
        "ess_B": ess_B,
        "B_param_labels": [f"B[{k},{l}]" for k, l in zip(iu_k, iv_k)],
        "stephens_iters": int(stephens_iters),
        "stephens_remaining_changes": int(stephens_remaining),
    }
    if degree_corrected and theta_mean is not None:
        result["theta"] = theta_mean
    return result
