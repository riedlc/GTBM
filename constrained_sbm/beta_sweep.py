"""Brent's-method sweep over the inverse temperature β.

ELBO(β) is expected to be unimodal (β→0 gives random mixing; β→∞ gives degenerate
concentration; the peak is somewhere in between), so a bounded scalar optimizer
finds the maximum in ~10–15 EM fits — much cheaper than a grid search.
"""

from __future__ import annotations

from scipy.optimize import minimize_scalar

from constrained_sbm.variational_em import fit


def sweep_beta(
    A,
    K,
    beta_low,
    beta_high,
    n_restarts,
    seed,
    degree_corrected=False,
    theta_max=100.0,
    pinned_diag=None,
    max_iter=200,
    tol=1e-6,
    verbose=False,
    xatol=0.01,
    maxiter=30,
    gauge=None,
    directed=None,
    n_spectral=1,
):
    """Find the β maximizing the constrained-SBM ELBO via Brent's method.

    Parameters
    ----------
    A : scipy.sparse adjacency
    K, n_restarts, seed, degree_corrected, theta_max, pinned_diag, max_iter, tol,
    gauge, directed, n_spectral
        Passed through unchanged to `fit()`.
    beta_low, beta_high : float
        Bounds for the Brent search interval.
    verbose : bool
        Print one line per Brent evaluation.
    xatol : float
        Brent convergence tolerance on β. Default 0.01 — plenty since ELBO is
        flat near the peak; each iteration is a full EM run.
    maxiter : int
        Hard safety cap. Brent's should converge in ~15 iterations on a typical
        interval.

    Returns
    -------
    best_result : dict
        The full `fit()` result dict at the optimal β (no redundant re-fit; the
        best EM run is captured on the fly during the search).
    trace : list of (beta, elbo) tuples
        Every β evaluated during the search, in evaluation order.
    """
    trace = []
    best = {"elbo": float("-inf"), "result": None}

    def neg_elbo(beta):
        result = fit(
            A, K=K, beta=beta,
            n_restarts=n_restarts,
            seed=seed,
            degree_corrected=degree_corrected,
            theta_max=theta_max,
            pinned_diag=pinned_diag,
            max_iter=max_iter,
            tol=tol,
            gauge=gauge,
            directed=directed,
            n_spectral=n_spectral,
        )
        elbo = result["elbo"]
        trace.append((float(beta), float(elbo)))
        if elbo > best["elbo"]:
            best["elbo"] = elbo
            best["result"] = result
        if verbose:
            print(f"  β={float(beta):.4f}  ELBO={elbo:.4f}")
        return -elbo

    minimize_scalar(
        neg_elbo,
        bounds=(beta_low, beta_high),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    return best["result"], trace
