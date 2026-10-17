"""Full Bayesian inference via Gibbs sampling for the constrained SBM.

The sampler cycles through the conditionals
    1. z       | rest      (categorical, direct sampling)
    2. π       | z          (Dirichlet, conjugate)
    3. θ       | z, B, A    (Gamma, conjugate under Poisson DC likelihood)
    4. payoffs | z, θ, A    (random-walk Metropolis-Hastings)
    5. β       | rest       (random-walk MH in log β; only when infer_beta=True)

EM (`fit()`) is used only to initialize the chain for faster burn-in. The
returned `fit_gibbs()` dict has the same keys as the EM result dict so the
downstream diagnostics (MDL, Vuong, ∆ELPD, evaluation, printing) accept either.

References
----------
Karrer & Newman (2011); Peixoto (2017); Gelman & Rubin (1992).
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from constrained_sbm.baseline_sbm import (
    _elbo as _baseline_elbo,
    _elbo_dc as _baseline_elbo_dc,
    _suff_stats as _baseline_suff_stats,
    _suff_stats_dc as _baseline_suff_stats_dc,
)
from constrained_sbm.model import (
    DEFAULT_PINNED_DIAG,
    M_to_B_K,
    _payoff_matrix,
    compute_M_K,
    n_free_params,
)
from constrained_sbm.variational_em import fit as fit_em

PROB_FLOOR = 1e-10
PROB_CEIL = 1.0 - 1e-10
PARAM_LO = 0.01
PARAM_HI = 0.99


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _kappa(z, theta, K):
    return np.array([theta[z == l].sum() for l in range(K)], dtype=float)


# -----------------------------------------------------------------------------
# Sufficient statistics under hard assignments
# -----------------------------------------------------------------------------


def _suff_stats_hard(A_csr, z, theta, K, degree_corrected, directed=False):
    """Edge counts m[k,l] and the κ⊗κ (or N⊗N) total, with the same accounting
    as the EM M-step objective in `variational_em`.

    Returns m, non_edges. Both (K, K). Asymmetric in directed mode.

    Convention:
      m[r, s] = #directed pairs (i, j) with A_ij = 1, z_i = r, z_j = s.
    For an undirected graph (with symmetric A), every within-block edge
    contributes twice to the diagonal of m_dir; we halve that diagonal to
    keep the same accounting as `baseline_sbm._suff_stats`, where the
    likelihood evaluator's 0.5 factor will recover the correct undirected
    log-likelihood. For directed graphs A is asymmetric and no halving is
    applied.
    """
    n = A_csr.shape[0]
    Z = sp.csr_matrix(
        (np.ones(n), (np.arange(n), z)), shape=(n, K)
    )
    m_dir = (Z.T @ A_csr @ Z).toarray()
    m = m_dir.copy()
    if not directed:
        np.fill_diagonal(m, np.diag(m) / 2.0)

    if degree_corrected:
        kappa = _kappa(z, theta, K)
        qt = (Z.toarray() * theta[:, None])
        self_term = qt.T @ qt
        total = np.outer(kappa, kappa) - self_term
    else:
        N = np.bincount(z, minlength=K).astype(float)
        total = np.outer(N, N) - np.diag(N)
    non_edges = total - m
    m = np.maximum(m, 0.0)
    non_edges = np.maximum(non_edges, 0.0)
    return m, non_edges


def _log_lik_from_stats(free_params, m, non_edges, freqs, n, num_edges,
                         beta, pinned_diag, gauge="pin_diag",
                         alpha_pay=1.0, beta_pay=1.0,
                         degree_corrected=False, directed=False):
    """Bernoulli-style log-likelihood at hard assignments — same form as the
    M-step objective. For DC mode the totals are κ⊗κ (passed in via non_edges)
    so this approximates the Poisson DC log-likelihood for sparse graphs.

    If `alpha_pay != 1` or `beta_pay != 1`, a Beta(alpha_pay, beta_pay) log-prior
    on each free payoff is added (normalizer dropped). The MH acceptance ratio
    in `sample_payoffs` then becomes the correct log-posterior ratio. With the
    default Uniform(0,1) (α=β=1), the prior term vanishes.

    `degree_corrected` selects the link function used by `M_to_B_K`
    (Bernoulli OR vs. Poisson rate addition).
    """
    free_params = np.clip(np.asarray(free_params, dtype=float), PARAM_LO, PARAM_HI)
    M = compute_M_K(free_params, freqs, beta, pinned_diag, gauge=gauge)
    B = M_to_B_K(M, freqs, n, num_edges,
                  directed=directed, degree_corrected=degree_corrected)
    logB = np.log(np.maximum(B, PROB_FLOOR))
    log1mB = np.log(np.maximum(1.0 - B, PROB_FLOOR))
    # In undirected mode m / non_edges double-count each pair (the diagonal is
    # already halved in `_suff_stats_hard`; off-diagonals retain the directed
    # sum which equals 2x the undirected cross count, so the 0.5 factor below
    # gives the correct undirected log-likelihood). In directed mode each
    # ordered pair is a distinct Bernoulli and the full sum is correct.
    scale = 1.0 if directed else 0.5
    ll = scale * float(np.sum(m * logB) + np.sum(non_edges * log1mB))
    if alpha_pay != 1.0 or beta_pay != 1.0:
        ll += float(
            np.sum((alpha_pay - 1.0) * np.log(free_params)
                   + (beta_pay - 1.0) * np.log(1.0 - free_params))
        )
    return ll


# -----------------------------------------------------------------------------
# Conditional 1: sample z (community assignments)
# -----------------------------------------------------------------------------


def sample_z(A_csr, z, B, theta, pi, K, rng, directed=False, A_csc=None):
    """One full sweep of z updates.

    Undirected (default): log P(z_i = k | rest) =
        log π_k
        + Σ_{j ∈ N(i)} log B[k, z_j]               (edge term)
        − θ_i · (B @ κ_{-i})[k]                     (expected-degree term)

    Directed: separate out- and in-edge contributions are accumulated. For
    a directed edge (i → j) the contribution to z_i is log B[k, z_j] and
    the expected-rate penalty is θ_i · (B @ κ)[k]. For an in-edge (j → i)
    the contribution to z_i is log B[z_j, k] and the rate penalty is
    θ_i · (Bᵀ @ κ)[k]. `A_csc` (the CSC representation of A) is required
    in directed mode to read in-edges efficiently; if omitted it is built
    from `A_csr` on the fly.
    """
    n = A_csr.shape[0]
    log_B = np.log(np.maximum(B, PROB_FLOOR))
    log_pi = np.log(np.clip(pi, PROB_FLOOR, 1.0))

    kappa = _kappa(z, theta, K)

    order = rng.permutation(n)
    indptr = A_csr.indptr
    indices = A_csr.indices

    if directed:
        if A_csc is None:
            A_csc = A_csr.tocsc()
        indptr_in = A_csc.indptr
        indices_in = A_csc.indices

    for i in order:
        old_k = int(z[i])
        kappa[old_k] -= theta[i]

        a, b = indptr[i], indptr[i + 1]
        out_nbrs = indices[a:b]
        z_out = z[out_nbrs]
        edge_term = np.zeros(K)
        if len(out_nbrs) > 0:
            for k in range(K):
                edge_term[k] = log_B[k, z_out].sum()
        rate_term = theta[i] * (B @ kappa)

        if directed:
            a2, b2 = indptr_in[i], indptr_in[i + 1]
            in_nbrs = indices_in[a2:b2]
            z_in = z[in_nbrs]
            if len(in_nbrs) > 0:
                for k in range(K):
                    edge_term[k] += log_B[z_in, k].sum()
            rate_term = rate_term + theta[i] * (B.T @ kappa)

        log_q = log_pi + edge_term - rate_term
        probs = _softmax(log_q)
        new_k = int(rng.choice(K, p=probs))
        z[i] = new_k
        kappa[new_k] += theta[i]

    return z


# -----------------------------------------------------------------------------
# Conditional 2: sample π
# -----------------------------------------------------------------------------


def sample_pi(z, K, alpha_pi, rng):
    counts = np.bincount(z, minlength=K).astype(float)
    return rng.dirichlet(alpha_pi + counts)


# -----------------------------------------------------------------------------
# Conditional 3: sample θ (degree correction)
# -----------------------------------------------------------------------------


def sample_theta(degree, z, B, theta, K, alpha_theta, rng, directed=False):
    """Vectorized Gamma posterior update + per-community renormalization.

    Undirected:
        θ_i | rest ~ Gamma(α_θ + d_i, α_θ + Σ_l B[z_i, l] · κ_l)
    Directed (degree = d_out + d_in):
        θ_i | rest ~ Gamma(α_θ + d_i,
                            α_θ + Σ_l (B[z_i, l] + B[l, z_i]) · κ_l)
    """
    kappa = _kappa(z, theta, K)
    shapes = alpha_theta + degree
    if directed:
        rates = alpha_theta + (B[z] @ kappa) + (B.T[z] @ kappa)
    else:
        rates = alpha_theta + (B[z] @ kappa)
    rates = np.maximum(rates, 1e-12)
    theta_new = rng.gamma(shapes, 1.0 / rates)
    # Fixed clip; the `theta_max` argument governs the EM initialisation only.
    theta_new = np.clip(theta_new, 1e-4, 1e3)
    # Renormalize so κ_k = N_k within each community (identifiability w/ B)
    for k in range(K):
        mask = z == k
        m_sum = theta_new[mask].sum()
        if mask.any() and m_sum > 0:
            theta_new[mask] *= mask.sum() / m_sum
    return theta_new


# -----------------------------------------------------------------------------
# Conditional 4: sample payoffs via Metropolis-Hastings
# -----------------------------------------------------------------------------


class AdaptiveProposal:
    """Robbins-Monro-style adaptation of the MH proposal std during burn-in."""

    def __init__(self, initial_std=0.05, target_rate=0.30):
        self.std = float(initial_std)
        self.target_rate = target_rate
        self.n_accepted = 0
        self.n_proposed = 0

    def adapt(self, accepted):
        self.n_proposed += 1
        self.n_accepted += int(accepted)
        if self.n_proposed % 50 == 0:
            rate = self.n_accepted / self.n_proposed
            if rate > self.target_rate + 0.05:
                self.std *= 1.1
            elif rate < self.target_rate - 0.05:
                self.std *= 0.9
            self.n_accepted = 0
            self.n_proposed = 0


# -----------------------------------------------------------------------------
# Conditional 5: sample β via Metropolis-Hastings (when infer_beta is on)
# -----------------------------------------------------------------------------


def sample_beta_mh(beta_current, log_lik_current, free_params, m, non_edges,
                   freqs, n, num_edges, pinned_diag, gauge,
                   prior_shape, prior_rate, proposal_log_std, rng,
                   alpha_pay=1.0, beta_pay=1.0,
                   degree_corrected=False, directed=False):
    """One MH proposal on the inverse temperature β.

    Prior: β ~ Gamma(shape=prior_shape, rate=prior_rate), so
        log p(β) = (prior_shape − 1) log β − prior_rate · β  (+ const).

    Proposal: log(β') = log(β) + N(0, σ²) — a random walk in log-space, which
    keeps β strictly positive and naturally adapts to its scale. The proposal
    density in β-space is asymmetric, so the MH ratio includes a Hastings
    adjustment:
        log q(β | β') − log q(β' | β) = log(β') − log(β)

    Likelihood is the same Bernoulli-on-sufficient-stats form used by the
    payoffs MH step (`_log_lik_from_stats`), evaluated at the current accepted
    payoffs and frequencies with the proposed β.

    Returns (beta, log_lik_or_post, accepted). The returned log_lik mirrors
    whatever `_log_lik_from_stats` returns under the active payoff prior — pure
    log-likelihood under Uniform(0,1) payoff prior, log-posterior up to a
    constant otherwise.
    """
    log_beta_proposed = float(np.log(beta_current) + rng.normal(0.0, proposal_log_std))
    beta_proposed = float(np.exp(log_beta_proposed))
    if not np.isfinite(beta_proposed) or beta_proposed <= 0.0:
        return beta_current, log_lik_current, False

    log_lik_proposed = _log_lik_from_stats(
        free_params, m, non_edges, freqs, n, num_edges,
        beta_proposed, pinned_diag, gauge=gauge,
        alpha_pay=alpha_pay, beta_pay=beta_pay,
        degree_corrected=degree_corrected, directed=directed,
    )

    log_prior_current = (prior_shape - 1.0) * np.log(beta_current) - prior_rate * beta_current
    log_prior_proposed = (prior_shape - 1.0) * np.log(beta_proposed) - prior_rate * beta_proposed
    log_hastings = np.log(beta_proposed) - np.log(beta_current)

    log_alpha = (log_lik_proposed + log_prior_proposed
                 - log_lik_current - log_prior_current
                 + log_hastings)

    if np.log(rng.uniform()) < log_alpha:
        return beta_proposed, log_lik_proposed, True
    return beta_current, log_lik_current, False


def sample_payoffs(current_payoffs, log_lik_current, m, non_edges, freqs, n,
                   num_edges, beta, pinned_diag, proposal_std, rng,
                   gauge="pin_diag", alpha_pay=1.0, beta_pay=1.0,
                   degree_corrected=False, directed=False):
    """One MH proposal on payoffs. Returns (payoffs, log_lik, accepted).

    The `log_lik` quantity passed in/out is whatever `_log_lik_from_stats`
    returns — log-likelihood under the default uniform prior, or log-posterior
    (up to a constant) when a non-trivial Beta(α, β) prior is in effect.
    """
    n_params = len(current_payoffs)
    proposed = current_payoffs + rng.normal(0, proposal_std, size=n_params)
    proposed = np.clip(proposed, PARAM_LO, PARAM_HI)

    log_lik_proposed = _log_lik_from_stats(
        proposed, m, non_edges, freqs, n, num_edges, beta, pinned_diag,
        gauge=gauge, alpha_pay=alpha_pay, beta_pay=beta_pay,
        degree_corrected=degree_corrected, directed=directed,
    )
    log_alpha = log_lik_proposed - log_lik_current
    if np.log(rng.uniform()) < log_alpha:
        return proposed, log_lik_proposed, True
    return current_payoffs, log_lik_current, False


# -----------------------------------------------------------------------------
# Single-chain Gibbs sampler
# -----------------------------------------------------------------------------


def posterior_sample(
    A, K=2, beta=3.0, pinned_diag=None,
    degree_corrected=False, theta_max=100.0,
    alpha_pi=1.0, alpha_theta=1.0,
    alpha_pay=1.0, beta_pay=1.0,
    infer_beta=False, prior_beta_shape=2.0, prior_beta_rate=2.0/3.0,
    beta_proposal_log_std=0.15,
    n_samples=1500, burn_in=750, thin=2, n_mh_per_iter=5,
    em_restarts=3, em_max_iter=200, em_tol=1e-6,
    random_seed=0, verbose=False, gauge=None,
    init_state=None, directed=False, init_mode=None,
):
    """Run a single Gibbs chain. Returns a samples dict (see below).

    The chain is initialized from a quick EM fit at the same K, β, DC mode,
    and gauge.

    `alpha_pay`, `beta_pay` parametrize a Beta(α, β) prior on each free payoff
    used by the MH step. Defaults to Uniform(0,1) — pass e.g. (2, 2) to gently
    regularize payoffs away from 0/1 on small / weak-signal graphs.

    When `infer_beta=True`, the inverse temperature is sampled by an additional
    MH step inserted after the payoff updates. Prior:
    β ~ Gamma(`prior_beta_shape`, `prior_beta_rate`) — default Gamma(2, 2/3)
    is weakly informative around β=3 (mean 3, std ≈ 2.12). The proposal is a
    log-Normal random walk with std `beta_proposal_log_std` (adaptive during
    burn-in). The input `beta` then acts as the initialization rather than a
    fixed value, and `samples["beta"]` carries the full trace.
    """
    rng = np.random.default_rng(random_seed)
    A_csr = sp.csr_matrix(A)
    n = A_csr.shape[0]
    if directed:
        num_edges = int(A_csr.nnz)
        A_csc = A_csr.tocsc()
        degree = (np.asarray(A_csr.sum(axis=1)).ravel()
                  + np.asarray(A_csr.sum(axis=0)).ravel()).astype(float)
    else:
        num_edges = int(A_csr.nnz // 2)
        A_csc = None
        degree = np.asarray(A_csr.sum(axis=1)).ravel().astype(float)

    if gauge is None:
        gauge = "row_sum" if K == 2 else "pin_diag"
    if gauge == "pin_diag":
        if pinned_diag is None:
            pinned_diag = DEFAULT_PINNED_DIAG.get(K, tuple(np.linspace(2/3, 1/3, K)))
        pinned_diag = tuple(float(p) for p in pinned_diag)
    else:
        pinned_diag = None

    # --- Initialization: either from a pre-computed EM MAP state (shared
    # across all chains for conditional posterior inference), or from a
    # per-chain quick EM warm-start (default).
    if init_state is not None:
        z = np.asarray(init_state["z"], dtype=int).copy()
        payoffs = np.asarray(init_state["payoffs"], dtype=float).copy()
        pi = np.asarray(init_state.get("pi", np.full(K, 1.0 / K)),
                        dtype=float).copy()
        if degree_corrected:
            theta_arr = init_state.get("theta")
            theta = (np.asarray(theta_arr, dtype=float).copy()
                     if theta_arr is not None else np.ones(n, dtype=float))
        else:
            theta = np.ones(n, dtype=float)
        if "beta" in init_state and init_state["beta"] is not None:
            beta = float(init_state["beta"])
    else:
        em = fit_em(
            A, K=K, beta=beta, pinned_diag=pinned_diag,
            n_restarts=em_restarts, max_iter=em_max_iter, tol=em_tol,
            degree_corrected=degree_corrected, theta_max=theta_max,
            seed=random_seed, verbose=False, gauge=gauge,
            directed=directed, init_mode=init_mode,
        )
        z = em["q"].argmax(axis=1).astype(int)
        payoffs = np.asarray(em["free_params"], dtype=float).copy()
        pi = np.asarray(em["pi"], dtype=float).copy()
        theta = (np.asarray(em["theta"], dtype=float).copy()
                 if degree_corrected else np.ones(n, dtype=float))

    n_free = n_free_params(K)
    adapter = AdaptiveProposal(initial_std=0.05)
    beta_adapter = (AdaptiveProposal(initial_std=beta_proposal_log_std)
                    if infer_beta else None)

    samples = {
        "payoffs": np.zeros((n_samples, n_free), dtype=float),
        "freqs": np.zeros((n_samples, K), dtype=float),
        "pi": np.zeros((n_samples, K), dtype=float),
        "z": np.zeros((n_samples, n), dtype=np.int32),
        "log_lik": np.zeros(n_samples, dtype=float),
    }
    if degree_corrected:
        samples["theta"] = np.zeros((n_samples, n), dtype=float)
    if infer_beta:
        samples["beta"] = np.zeros(n_samples, dtype=float)

    mh_total = 0
    mh_accepted = 0
    beta_mh_total = 0
    beta_mh_accepted = 0
    total_iter = burn_in + n_samples * thin
    sample_idx = 0

    for t in range(total_iter):
        is_burn_in = t < burn_in

        # Recompute B from current payoffs/freqs
        freqs = np.bincount(z, minlength=K).astype(float) / n
        # Guard against degenerate freqs (an empty class breaks compute_M_K)
        freqs = np.clip(freqs, 1e-4, 1.0)
        freqs /= freqs.sum()
        M = compute_M_K(payoffs, freqs, beta, pinned_diag, gauge=gauge)
        B = M_to_B_K(M, freqs, n, num_edges,
                      directed=directed, degree_corrected=degree_corrected)

        # Step 1: z
        z = sample_z(A_csr, z, B, theta, pi, K, rng,
                      directed=directed, A_csc=A_csc)

        # Step 2: π
        pi = sample_pi(z, K, alpha_pi, rng)

        # Step 3: θ (only if DC)
        if degree_corrected:
            theta = sample_theta(degree, z, B, theta, K, alpha_theta, rng,
                                  directed=directed)

        # Step 4: payoffs (MH, multiple proposals)
        freqs = np.bincount(z, minlength=K).astype(float) / n
        freqs = np.clip(freqs, 1e-4, 1.0)
        freqs /= freqs.sum()
        m_stat, non_edges = _suff_stats_hard(A_csr, z, theta, K, degree_corrected,
                                              directed=directed)
        log_lik = _log_lik_from_stats(
            payoffs, m_stat, non_edges, freqs, n, num_edges, beta, pinned_diag,
            gauge=gauge, alpha_pay=alpha_pay, beta_pay=beta_pay,
            degree_corrected=degree_corrected, directed=directed,
        )
        for _ in range(n_mh_per_iter):
            payoffs, log_lik, accepted = sample_payoffs(
                payoffs, log_lik, m_stat, non_edges, freqs, n, num_edges,
                beta, pinned_diag, adapter.std, rng, gauge=gauge,
                alpha_pay=alpha_pay, beta_pay=beta_pay,
                degree_corrected=degree_corrected, directed=directed,
            )
            mh_total += 1
            mh_accepted += int(accepted)
            if is_burn_in:
                adapter.adapt(accepted)

        # Step 5: β (MH) — only when infer_beta=True. β depends on the current
        # payoffs and freqs (via M and B), so it must be sampled AFTER the
        # payoff updates. The Hastings-corrected log-Normal random walk keeps
        # β > 0.
        if infer_beta:
            beta, log_lik, accepted_b = sample_beta_mh(
                beta, log_lik, payoffs, m_stat, non_edges, freqs, n, num_edges,
                pinned_diag, gauge, prior_beta_shape, prior_beta_rate,
                beta_adapter.std, rng,
                alpha_pay=alpha_pay, beta_pay=beta_pay,
                degree_corrected=degree_corrected, directed=directed,
            )
            beta_mh_total += 1
            beta_mh_accepted += int(accepted_b)
            if is_burn_in:
                beta_adapter.adapt(accepted_b)

        # Store
        if not is_burn_in and (t - burn_in) % thin == 0 and sample_idx < n_samples:
            samples["payoffs"][sample_idx] = payoffs
            samples["freqs"][sample_idx] = freqs
            samples["pi"][sample_idx] = pi
            samples["z"][sample_idx] = z
            samples["log_lik"][sample_idx] = log_lik
            if infer_beta:
                samples["beta"][sample_idx] = beta
            if degree_corrected:
                samples["theta"][sample_idx] = theta
            sample_idx += 1

        if verbose and (t + 1) % 200 == 0:
            phase = "burn-in" if is_burn_in else "sampling"
            rate = mh_accepted / max(mh_total, 1)
            print(f"  iter {t+1}/{total_iter} ({phase})  "
                  f"MH accept={rate:.1%}  log_lik={log_lik:.2f}")

    samples["acceptance_rate"] = float(mh_accepted / max(mh_total, 1))
    if infer_beta:
        samples["beta_acceptance_rate"] = float(beta_mh_accepted / max(beta_mh_total, 1))
    samples["n_samples"] = int(sample_idx)
    return samples


# -----------------------------------------------------------------------------
# Label alignment (post-hoc relabeling)
# -----------------------------------------------------------------------------


def _permute_payoffs(payoffs_flat, perm, K):
    """Apply community permutation σ to the flattened off-diagonal payoff vector.

    `payoffs_flat` is row-major over off-diagonal entries (i ≠ j). Element k
    corresponds to (i, j) with j > i mapped via row-major skip-diagonal scan;
    rebuilding the KxK matrix, permuting, and flattening is robust.
    """
    P_full = np.zeros((K, K), dtype=float)
    idx = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                P_full[i, j] = payoffs_flat[idx]
                idx += 1
    perm = np.asarray(perm, dtype=int)
    P_perm = P_full[np.ix_(perm, perm)]
    out = np.zeros_like(payoffs_flat)
    idx = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                out[idx] = P_perm[i, j]
                idx += 1
    return out


def align_samples(samples, K):
    """Align all samples to the first sample's labeling via brute-force permutation."""
    n_samples = samples["z"].shape[0]
    z_ref = samples["z"][0]
    perm_iter = list(itertools.permutations(range(K)))
    for t in range(1, n_samples):
        z_t = samples["z"][t]
        best_perm, best_agree = perm_iter[0], -1
        for sigma in perm_iter:
            sigma_arr = np.array(sigma, dtype=int)
            agree = int((sigma_arr[z_t] == z_ref).sum())
            if agree > best_agree:
                best_agree = agree
                best_perm = sigma
        sigma_arr = np.array(best_perm, dtype=int)
        samples["z"][t] = sigma_arr[z_t]
        samples["pi"][t] = samples["pi"][t][list(best_perm)]
        samples["freqs"][t] = samples["freqs"][t][list(best_perm)]
        samples["payoffs"][t] = _permute_payoffs(samples["payoffs"][t], best_perm, K)
        # θ is per-node so it stays with the node (no permutation needed)
    return samples


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def r_hat(chains):
    """Gelman-Rubin diagnostic. `chains` is a list of 1D arrays, one per chain."""
    chains = [np.asarray(c, dtype=float) for c in chains]
    m = len(chains)
    n = len(chains[0])
    if m < 2 or n < 2:
        return float("nan")
    means = np.array([c.mean() for c in chains])
    vars_ = np.array([c.var(ddof=1) for c in chains])
    W = vars_.mean()
    B = n * means.var(ddof=1)
    var_hat = (1.0 - 1.0 / n) * W + (1.0 / n) * B
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_hat / W))


def effective_sample_size(x):
    """ESS via autocorrelation sum until the first negative pair."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(n)
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return float(n)
    acf_full = np.correlate(x, x, mode="full") / (n * var)
    acf = acf_full[n - 1:]  # k = 0, 1, ...
    s = 0.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        s += acf[k]
    return float(n / (1.0 + 2.0 * s))


# -----------------------------------------------------------------------------
# Posterior summary
# -----------------------------------------------------------------------------


def compute_elbo_from_posterior(A_csr, q, B, theta, pi, degree_corrected,
                                  directed=False):
    """Variational ELBO evaluated at the Gibbs-derived soft assignments q.

    The Gibbs sampler stores per-sample log-likelihoods at *hard* z; the
    posterior summary then averages indicators across samples to give soft
    assignment probabilities `q = posterior_summary["assignment_probs"]`.
    Evaluating the same ELBO formula EM uses (likelihood + Σ q log π − Σ q log q,
    plus the Poisson DC degree term in DC mode) at this q gives a quantity
    that is directly comparable to the EM ELBO and to the free-SBM ELBO.

    The entropy term −Σ q log q is what makes this *lower* than the
    hard-assignment log-likelihood whenever any nodes have ambiguous
    posterior assignments: hard z carries no entropy penalty, soft q does.

    Args:
        A_csr: sparse adjacency
        q: (n, K) posterior assignment probabilities (rows sum to 1)
        B: (K, K) edge-probability matrix from posterior-mean payoffs/freqs
        theta: (n,) degree parameters (ones if not DC; only read when DC)
        pi: (K,) community proportions
        degree_corrected: bool
        directed: bool, evaluate the directed likelihood

    Returns:
        float ELBO, using EM's exact formula via `_elbo` / `_elbo_dc`.
    """
    if degree_corrected:
        m, _, _, kappa = _baseline_suff_stats_dc(A_csr, q, theta)
        if directed:
            degree = (np.asarray(A_csr.sum(axis=1)).ravel()
                      + np.asarray(A_csr.sum(axis=0)).ravel()).astype(float)
        else:
            degree = np.asarray(A_csr.sum(axis=1)).ravel().astype(float)
        return float(_baseline_elbo_dc(m, q, B, pi, theta, kappa, degree,
                                         directed=directed))
    m, non_edges, _ = _baseline_suff_stats(A_csr, q)
    return float(_baseline_elbo(m, non_edges, q, B, pi, directed=directed))


def posterior_summary(samples, K):
    """Compute posterior means, 95% CIs, and per-node assignment probabilities."""
    payoffs = samples["payoffs"]
    freqs = samples["freqs"]
    n_samples = payoffs.shape[0]

    summary = {
        "payoff_mean": payoffs.mean(axis=0),
        "payoff_median": np.median(payoffs, axis=0),
        "payoff_ci_95": np.percentile(payoffs, [2.5, 97.5], axis=0),
        "payoff_std": payoffs.std(axis=0, ddof=1) if n_samples > 1 else np.zeros_like(payoffs[0]),
        "freq_mean": freqs.mean(axis=0),
        "freq_ci_95": np.percentile(freqs, [2.5, 97.5], axis=0),
    }
    if "theta" in samples:
        th = samples["theta"]
        summary["theta_mean"] = th.mean(axis=0)
        summary["theta_ci_95"] = np.percentile(th, [2.5, 97.5], axis=0)
    if "beta" in samples:
        b_ = samples["beta"]
        summary["beta_mean"] = float(b_.mean())
        summary["beta_median"] = float(np.median(b_))
        summary["beta_ci_95"] = np.percentile(b_, [2.5, 97.5])
        summary["beta_std"] = float(b_.std(ddof=1)) if n_samples > 1 else 0.0

    z = samples["z"]
    n = z.shape[1]
    assign = np.zeros((n, K), dtype=float)
    for k in range(K):
        assign[:, k] = (z == k).mean(axis=0)
    summary["assignment_probs"] = assign
    return summary


def diagnostics_report(samples_per_chain):
    """Per-parameter ESS and R-hat (if multi-chain)."""
    payoff_arrays = [c["payoffs"] for c in samples_per_chain]
    n_free = payoff_arrays[0].shape[1]
    rhat_per_param = []
    ess_per_param = []
    for j in range(n_free):
        rhat_per_param.append(r_hat([p[:, j] for p in payoff_arrays]))
        ess_per_param.append(effective_sample_size(np.concatenate([p[:, j] for p in payoff_arrays])))
    report = {
        "rhat_payoffs": rhat_per_param,
        "ess_payoffs": ess_per_param,
        "acceptance_rate": float(np.mean([c["acceptance_rate"] for c in samples_per_chain])),
        "n_chains": len(samples_per_chain),
        "n_samples_per_chain": int(samples_per_chain[0]["n_samples"]),
    }
    # β diagnostics, if β was inferred.
    if "beta" in samples_per_chain[0]:
        beta_arrays = [c["beta"] for c in samples_per_chain]
        report["rhat_beta"] = float(r_hat(beta_arrays))
        report["ess_beta"] = float(effective_sample_size(np.concatenate(beta_arrays)))
        report["beta_acceptance_rate"] = float(np.mean(
            [c.get("beta_acceptance_rate", 0.0) for c in samples_per_chain]
        ))
    return report


# -----------------------------------------------------------------------------
# Multi-chain top-level: fit_gibbs (returns EM-compatible result dict)
# -----------------------------------------------------------------------------


def _run_constrained_chain(kwargs):
    """Module-level worker for ProcessPoolExecutor — must be picklable.

    Runs one constrained Gibbs chain and aligns its samples for label
    consistency within the chain.
    """
    K = kwargs["K"]
    samples = posterior_sample(**kwargs)
    align_samples(samples, K)
    return samples


def fit_gibbs(
    A, K=2, beta=3.0, pinned_diag=None,
    degree_corrected=False, theta_max=100.0,
    n_samples=1500, burn_in=750, thin=2, n_mh_per_iter=5,
    n_chains=2, em_restarts=3, em_max_iter=200, em_tol=1e-6,
    alpha_pi=1.0, alpha_theta=1.0,
    alpha_pay=2.0, beta_pay=2.0,
    infer_beta=False, prior_beta_shape=2.0, prior_beta_rate=2.0/3.0,
    beta_proposal_log_std=0.15,
    seed=0, verbose=False, gauge=None,
    init_state=None, directed=None, n_jobs=1, init_mode=None,
):
    """Multi-chain Gibbs estimator. Returns a result dict with the same keys
    as `variational_em.fit()` plus extra Gibbs keys ("posterior_summary",
    "posterior_chains", "rhat_payoffs", "ess_payoffs", ...).

    A Beta(`alpha_pay`, `beta_pay`) prior on each free payoff regularizes the
    payoff MH step away from the 0/1 boundary — important on small graphs where
    the likelihood alone is too weak to penalize extreme payoffs. Defaults to
    Beta(2, 2). Pass (1, 1) to recover the uniform prior.

    When `infer_beta=True`, the inverse temperature β is also sampled by an MH
    step inserted after the payoff updates. Prior:
    β ~ Gamma(`prior_beta_shape`, `prior_beta_rate`); default Gamma(2, 2/3) is
    weakly informative around β=3 (mean 3, std ≈ 2.12). The proposal is a
    log-Normal random walk in β with std `beta_proposal_log_std` (adaptive
    during burn-in, targeting ~30% acceptance). The `beta` argument is then the
    chain initialization rather than a fixed value, and the result dict gains
    `beta_samples`, `rhat_beta`, `ess_beta`, and `beta_acceptance_rate` for
    convergence diagnostics; `result["beta"]` becomes the posterior mean.
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

    if gauge is None:
        gauge = "row_sum" if K == 2 else "pin_diag"
    if gauge == "pin_diag":
        if pinned_diag is None:
            pinned_diag = DEFAULT_PINNED_DIAG.get(K, tuple(np.linspace(2/3, 1/3, K)))
        pinned_diag = tuple(float(p) for p in pinned_diag)
    else:
        pinned_diag = None

    from constrained_sbm.utils import run_chains_parallel

    parallel = n_jobs is not None and n_jobs > 1 and n_chains > 1
    # Suppress per-chain verbose under parallel execution to avoid interleaved
    # output; the launch banner below summarizes instead.
    chain_verbose = verbose and not parallel
    chain_args_list = [
        dict(
            A=A_csr, K=K, beta=beta, pinned_diag=pinned_diag,
            degree_corrected=degree_corrected, theta_max=theta_max,
            alpha_pi=alpha_pi, alpha_theta=alpha_theta,
            alpha_pay=alpha_pay, beta_pay=beta_pay,
            infer_beta=infer_beta,
            prior_beta_shape=prior_beta_shape,
            prior_beta_rate=prior_beta_rate,
            beta_proposal_log_std=beta_proposal_log_std,
            n_samples=n_samples, burn_in=burn_in, thin=thin,
            n_mh_per_iter=n_mh_per_iter,
            em_restarts=em_restarts, em_max_iter=em_max_iter, em_tol=em_tol,
            random_seed=seed + c,
            verbose=chain_verbose,
            gauge=gauge,
            init_state=init_state, directed=directed, init_mode=init_mode,
        )
        for c in range(n_chains)
    ]
    if parallel:
        if verbose:
            print(f"--- launching {n_chains} chains across "
                  f"{min(n_jobs, n_chains)} worker process(es) "
                  f"(seeds {seed}..{seed + n_chains - 1}) ---")
        chains = run_chains_parallel(_run_constrained_chain, chain_args_list, n_jobs)
    else:
        chains = []
        for c, args in enumerate(chain_args_list):
            if verbose:
                print(f"--- chain {c+1}/{n_chains} (seed={seed + c}) ---")
            chains.append(_run_constrained_chain(args))

    # Align chain 1..k to chain 0
    ref_z = chains[0]["z"][0]
    for c in range(1, n_chains):
        chain = chains[c]
        z_t = chain["z"][0]
        best_perm = (0,)
        best_agree = -1
        for sigma in itertools.permutations(range(K)):
            sigma_arr = np.array(sigma, dtype=int)
            agree = int((sigma_arr[z_t] == ref_z).sum())
            if agree > best_agree:
                best_agree = agree
                best_perm = sigma
        # Re-apply the per-sample alignment with this fixed permutation
        sigma_arr = np.array(best_perm, dtype=int)
        for t in range(chain["z"].shape[0]):
            chain["z"][t] = sigma_arr[chain["z"][t]]
            chain["pi"][t] = chain["pi"][t][list(best_perm)]
            chain["freqs"][t] = chain["freqs"][t][list(best_perm)]
            chain["payoffs"][t] = _permute_payoffs(chain["payoffs"][t], best_perm, K)

    # Stack chains
    stacked = {
        "payoffs": np.concatenate([c["payoffs"] for c in chains], axis=0),
        "freqs": np.concatenate([c["freqs"] for c in chains], axis=0),
        "pi": np.concatenate([c["pi"] for c in chains], axis=0),
        "z": np.concatenate([c["z"] for c in chains], axis=0),
        "log_lik": np.concatenate([c["log_lik"] for c in chains], axis=0),
        "acceptance_rate": float(np.mean([c["acceptance_rate"] for c in chains])),
        "n_samples": int(sum(c["n_samples"] for c in chains)),
    }
    if degree_corrected:
        stacked["theta"] = np.concatenate([c["theta"] for c in chains], axis=0)
    if infer_beta:
        stacked["beta"] = np.concatenate([c["beta"] for c in chains], axis=0)

    summary = posterior_summary(stacked, K)
    diag = diagnostics_report(chains)

    # Build EM-compatible result dict from posterior means.
    # When β is inferred, the posterior-mean β is what we plug into M_to_B_K
    # for the posterior point summary; otherwise use the fixed input.
    payoffs_mean = summary["payoff_mean"]
    freqs_mean = summary["freq_mean"]
    beta_point = summary["beta_mean"] if infer_beta else float(beta)
    n = A_csr.shape[0]
    num_edges = int(A_csr.nnz) if directed else int(A_csr.nnz // 2)
    M = compute_M_K(payoffs_mean, freqs_mean, beta_point, pinned_diag, gauge=gauge)
    B = M_to_B_K(M, freqs_mean, n, num_edges,
                  directed=directed, degree_corrected=degree_corrected)
    P = _payoff_matrix(payoffs_mean, K=K, pinned_diag=pinned_diag, gauge=gauge)
    q_soft = summary["assignment_probs"]

    # Variational ELBO at the Gibbs-derived soft q, using EM's formula, so it is
    # comparable with the EM and free-SBM ELBOs: the hard-assignment log-lik
    # omits the entropy of q and overstates fit when assignments are uncertain
    # (it is reported separately as `log_lik_hard_mean`).
    theta_for_elbo = (summary.get("theta_mean", np.ones(n, dtype=float))
                      if degree_corrected else np.ones(n, dtype=float))
    elbo_soft = compute_elbo_from_posterior(
        A_csr, q_soft, B, theta_for_elbo, freqs_mean, degree_corrected,
        directed=directed,
    )

    result = {
        "K": K,
        "q": q_soft,
        "pi": freqs_mean,
        "freqs": freqs_mean,
        "free_params": payoffs_mean,
        "pinned_diag": (np.asarray(pinned_diag) if pinned_diag is not None else None),
        "payoff_matrix": P,
        "gauge": gauge,
        "beta": float(beta_point),
        "M": M,
        "B": B,
        "elbo": float(elbo_soft),
        "log_lik_hard_mean": float(stacked["log_lik"].mean()),
        # Keys shared with the EM result dict so downstream code accepts either:
        # the history is the hard-assignment log-likelihood trace (not an ELBO)
        # and "iterations" is the number of posterior draws.
        "elbo_history": stacked["log_lik"].tolist(),
        "converged": True,
        "iterations": stacked["n_samples"],
        "degree_corrected": degree_corrected,
        "directed": bool(directed),
        "estimator": "gibbs",
        # Gibbs-specific
        "posterior_summary": summary,
        "posterior_chains": chains,
        "rhat_payoffs": diag["rhat_payoffs"],
        "ess_payoffs": diag["ess_payoffs"],
        "acceptance_rate": stacked["acceptance_rate"],
        "n_chains": diag["n_chains"],
        "n_samples_per_chain": diag["n_samples_per_chain"],
    }
    if K == 2:
        result["x"] = float(freqs_mean[0])
        # K=2 ordering: free_params = [P[0,1], P[1,0]] = [c, b]
        result["c"] = float(payoffs_mean[0])
        result["b"] = float(payoffs_mean[1])
        result["a"] = float(P[0, 0])
        result["d"] = float(P[1, 1])
    if degree_corrected:
        result["theta"] = summary["theta_mean"]
        result["kappa"] = _kappa(stacked["z"][-1], summary["theta_mean"], K)
        result["theta_max"] = float(theta_max)
    if infer_beta:
        result["infer_beta"] = True
        result["beta_samples"] = stacked["beta"]
        result["beta_ci_95"] = summary["beta_ci_95"]
        result["rhat_beta"] = diag["rhat_beta"]
        result["ess_beta"] = diag["ess_beta"]
        result["beta_acceptance_rate"] = diag["beta_acceptance_rate"]
        result["prior_beta_shape"] = float(prior_beta_shape)
        result["prior_beta_rate"] = float(prior_beta_rate)

    return result
