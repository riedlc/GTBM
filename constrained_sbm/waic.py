"""Pointwise log-likelihoods and WAIC for SBM Gibbs posteriors.

Provides a single evaluator that consumes the per-sample (z, B, θ) tuples
saved by either the constrained Gibbs sampler (`fit_gibbs`) or the free
Gibbs sampler (`fit_free_gibbs`) and produces an S×N pointwise log-likelihood
matrix over the appropriate dyad enumeration.

The same likelihood family is used regardless of which sampler produced the
posterior, so WAIC comparisons across models are on the same scale:

    non-DC : log p_Bernoulli(A_ij; B[z_i, z_j])
    DC     : log p_Poisson (A_ij; θ_i θ_j B[z_i, z_j])

Dyad enumeration follows the fit's `directed` flag:
    undirected : N = n·(n−1)/2  unordered {i, j} pairs (upper triangle)
    directed   : N = n·(n−1)    ordered  (i, j), i ≠ j

In directed mode the recovered B is asymmetric and the two ordered pairs
(i, j) / (j, i) are *separate* Bernoulli/Poisson observations.

Standard WAIC (Watanabe 2010 / Gelman-Hwang-Vehtari 2014):

    lppd     = Σ_n log( (1/S) Σ_s exp(log_lik[s, n]) )
    p_waic   = Σ_n Var_s( log_lik[s, n] )
    WAIC     = -2 (lppd - p_waic)

Akaike weights converted from WAIC differences give a posterior probability
over the candidate models under a uniform prior.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from constrained_sbm.model import M_to_B_K, compute_M_K

EPS = 1e-12
RATE_FLOOR = 1e-12


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _dyad_indices(n, directed=False):
    """Return (row, col) index arrays for the dyad enumeration.

    Undirected (default): upper triangle i < j  →  n(n−1)/2 pairs.
    Directed: all ordered (i, j) with i ≠ j     →  n(n−1) pairs.
    """
    if directed:
        ii, jj = np.indices((n, n))
        mask = ii != jj
        return ii[mask], jj[mask]
    return np.triu_indices(n, k=1)


def _dense_A_at(A_csr, iu, iv):
    """Read A[iu, iv] as binary int8. Works for upper-tri or all-ordered indices."""
    # Binarize: a weighted or multigraph adjacency may hold counts > 1; the SBM
    # assumes A_ij in {0, 1}.
    # The SBM models all assume A_ij ∈ {0, 1}, so anything > 0 is an edge.
    A_dense = (np.asarray(A_csr.todense()) > 0).astype(np.int8)
    return A_dense[iu, iv]


def _pointwise_loglik_one(A_vec, z_s, B_s, theta_s, iu, iv, degree_corrected):
    """Log-lik vector over N dyads for a single posterior sample."""
    B_ij = B_s[z_s[iu], z_s[iv]]  # (N,)
    if degree_corrected:
        rate = theta_s[iu] * theta_s[iv] * B_ij
        rate = np.maximum(rate, RATE_FLOOR)
        # Poisson log-pmf at A_ij ∈ {0, 1} (gammaln(A_ij+1) = 0 either way)
        return A_vec * np.log(rate) - rate
    p = np.clip(B_ij, EPS, 1.0 - EPS)
    return A_vec * np.log(p) + (1 - A_vec) * np.log1p(-p)


# -----------------------------------------------------------------------------
# Pointwise log-likelihood matrices
# -----------------------------------------------------------------------------


def pointwise_loglik_constrained(A, result):
    """S×N log-lik matrix from a constrained `fit_gibbs` result dict.

    Reconstructs B per sample by mapping the saved payoff/freqs samples through
    the same M_to_B_K used during sampling. θ is read from saved samples in DC
    mode and treated as ones otherwise.

    Dyad enumeration matches the fit's `directed` flag — N = n(n−1)/2 for
    undirected, N = n(n−1) for directed. The per-sample B is then constructed
    in the same mode so its values line up with the dyad indices.
    """
    A_csr = sp.csr_matrix(A)
    n = A_csr.shape[0]
    K = int(result["K"])
    beta = float(result["beta"])
    pinned_diag = result.get("pinned_diag")
    if pinned_diag is not None:
        pinned_diag = tuple(float(p) for p in np.asarray(pinned_diag).ravel())
    gauge = result.get("gauge", "row_sum" if K == 2 else "pin_diag")
    degree_corrected = bool(result.get("degree_corrected", False))
    directed = bool(result.get("directed", False))
    num_edges = int(A_csr.nnz) if directed else int(A_csr.nnz // 2)

    chains = result["posterior_chains"]
    payoffs_all = np.concatenate([c["payoffs"] for c in chains], axis=0)
    freqs_all = np.concatenate([c["freqs"] for c in chains], axis=0)
    z_all = np.concatenate([c["z"] for c in chains], axis=0)
    if degree_corrected:
        theta_all = np.concatenate([c["theta"] for c in chains], axis=0)
    else:
        theta_all = None

    S = z_all.shape[0]
    iu, iv = _dyad_indices(n, directed=directed)
    A_vec = _dense_A_at(A_csr, iu, iv).astype(np.int8)
    N_dyads = A_vec.size

    out = np.empty((S, N_dyads), dtype=np.float32)
    for s in range(S):
        M_s = compute_M_K(payoffs_all[s], freqs_all[s], beta, pinned_diag, gauge=gauge)
        B_s = M_to_B_K(M_s, freqs_all[s], n, num_edges,
                        directed=directed, degree_corrected=degree_corrected)
        theta_s = (theta_all[s] if degree_corrected else None)
        ll = _pointwise_loglik_one(A_vec, z_all[s], B_s, theta_s, iu, iv,
                                    degree_corrected)
        out[s] = ll.astype(np.float32)
    return out


def pointwise_loglik_free(A, result):
    """S×N log-lik matrix from a free-Gibbs `fit_free_gibbs` result dict.

    Reads B directly from the saved samples (no payoff conversion). Dyad
    enumeration matches the fit's `directed` flag; in directed mode the
    saved B samples are already asymmetric per the directed free sampler,
    so `B[z_i, z_j]` and `B[z_j, z_i]` enter as the two distinct observations.
    """
    A_csr = sp.csr_matrix(A)
    n = A_csr.shape[0]
    degree_corrected = bool(result.get("degree_corrected", False))
    directed = bool(result.get("directed", False))

    stacked = result["posterior_stacked"]
    z_all = stacked["z"]
    B_all = stacked["B"]
    theta_all = stacked["theta"] if degree_corrected else None

    S = z_all.shape[0]
    iu, iv = _dyad_indices(n, directed=directed)
    A_vec = _dense_A_at(A_csr, iu, iv).astype(np.int8)
    N_dyads = A_vec.size

    out = np.empty((S, N_dyads), dtype=np.float32)
    for s in range(S):
        theta_s = (theta_all[s] if degree_corrected else None)
        ll = _pointwise_loglik_one(A_vec, z_all[s], B_all[s], theta_s, iu, iv,
                                    degree_corrected)
        out[s] = ll.astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# WAIC
# -----------------------------------------------------------------------------


def waic_from_loglik(loglik):
    """Compute WAIC from an S×N pointwise log-likelihood matrix.

    Returns dict with keys: lppd, p_waic, waic, se_waic (approx SE over dyads).
    """
    loglik = np.asarray(loglik, dtype=np.float64)
    S, N = loglik.shape
    # lppd: sum over dyads of log( mean_s exp(ll[s, n]) ) = logsumexp - log S
    lppd_pointwise = logsumexp(loglik, axis=0) - np.log(S)
    lppd = float(lppd_pointwise.sum())

    var_pointwise = loglik.var(axis=0, ddof=1) if S > 1 else np.zeros(N)
    p_waic = float(var_pointwise.sum())

    waic = -2.0 * (lppd - p_waic)

    # Per-dyad WAIC contribution: -2 (lppd_n - var_n). SE = sqrt(N · Var of contributions).
    contributions = -2.0 * (lppd_pointwise - var_pointwise)
    se_waic = float(np.sqrt(N * contributions.var(ddof=1))) if N > 1 else 0.0

    return {
        "lppd": lppd,
        "p_waic": p_waic,
        "waic": float(waic),
        "se_waic": se_waic,
        "n_dyads": int(N),
        "n_samples": int(S),
    }


def compare_waic(waic_a, waic_b, label_a="model_a", label_b="model_b"):
    """Return Δ, SE(Δ), and Akaike weights for a two-model WAIC comparison."""
    delta = waic_a["waic"] - waic_b["waic"]
    # Lower WAIC = better. Akaike weights:
    waics = np.array([waic_a["waic"], waic_b["waic"]])
    w = np.exp(-0.5 * (waics - waics.min()))
    w /= w.sum()
    preferred = label_a if waic_a["waic"] < waic_b["waic"] else label_b
    return {
        "delta_waic": float(delta),  # waic_a - waic_b
        "weight_" + label_a: float(w[0]),
        "weight_" + label_b: float(w[1]),
        "preferred": preferred,
    }


# -----------------------------------------------------------------------------
# PSIS-LOO-CV via arviz
# -----------------------------------------------------------------------------


def loo_from_loglik(loglik, n_chains):
    """PSIS-LOO-CV from an S×N pointwise log-likelihood matrix.

    Uses `arviz.loo` (Pareto-smoothed importance sampling). The S rows must be
    layered chain-major: rows [0, S/n_chains) belong to chain 0, the next block
    to chain 1, etc. This matches the layout produced by stacking the Gibbs
    samplers' per-chain trace dicts.

    Returns a dict with the same `waic_from_loglik` shape plus LOO-specific
    fields:
        elpd_loo  : expected log pointwise predictive density (higher is better)
        se_elpd_loo
        p_loo     : effective number of parameters
        looic     : -2·elpd_loo  (lower is better; comparable to WAIC)
        se_looic  : 2·SE(elpd_loo)
        n_obs     : N (dyads)
        n_samples : S
        max_pareto_k, n_pareto_k_high : diagnostics on PSIS reliability
        warning   : True iff arviz flagged any observation as unreliable
    """
    import warnings

    import arviz as az

    loglik = np.asarray(loglik, dtype=np.float64)
    S, N = loglik.shape
    if S % n_chains != 0:
        raise ValueError(
            f"S={S} not divisible by n_chains={n_chains}; refusing to reshape."
        )
    n_per = S // n_chains
    ll_cdo = loglik.reshape(n_chains, n_per, N)

    # arviz needs a 'posterior' group present even if we only care about
    # log_likelihood — supply a dummy 1-D parameter.
    idata = az.from_dict(
        posterior={"_dummy": np.zeros((n_chains, n_per))},
        log_likelihood={"obs": ll_cdo},
    )
    # Suppress arviz's "Pareto shape > 0.7" UserWarning; the diagnostic is
    # extracted from `res.pareto_k` below and returned as `max_pareto_k` /
    # `n_pareto_k_high` instead.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Pareto distribution.*",
            category=UserWarning,
        )
        res = az.loo(idata, pointwise=True)

    pareto_k = np.asarray(res.pareto_k) if hasattr(res, "pareto_k") else None
    good_k = float(getattr(res, "good_k", 0.7))
    n_high = int((pareto_k > good_k).sum()) if pareto_k is not None else 0
    max_k = float(pareto_k.max()) if pareto_k is not None else float("nan")

    elpd = float(res.elpd_loo)
    se = float(res.se)
    return {
        "elpd_loo": elpd,
        "se_elpd_loo": se,
        "p_loo": float(res.p_loo),
        "looic": -2.0 * elpd,
        "se_looic": 2.0 * se,
        "n_obs": int(N),
        "n_samples": int(S),
        "max_pareto_k": max_k,
        "n_pareto_k_high": n_high,
        "good_k_threshold": good_k,
        "warning": bool(res.warning),
    }


def compare_loo(loo_a, loo_b, label_a="model_a", label_b="model_b"):
    """ΔLOOIC and Akaike weights (lower LOOIC preferred), parallel to compare_waic."""
    delta = loo_a["looic"] - loo_b["looic"]
    looics = np.array([loo_a["looic"], loo_b["looic"]])
    w = np.exp(-0.5 * (looics - looics.min()))
    w /= w.sum()
    preferred = label_a if loo_a["looic"] < loo_b["looic"] else label_b
    return {
        "delta_looic": float(delta),  # looic_a - looic_b
        "weight_" + label_a: float(w[0]),
        "weight_" + label_b: float(w[1]),
        "preferred": preferred,
    }
