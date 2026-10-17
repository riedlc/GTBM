"""Orientation convention for K=2 pin_diag fits (the exact X<->Y gauge tie).

Relabeling the two communities maps the fitted parameters by
(b, c, x) -> (c*d/a, b*a/d, 1-x) and reproduces the identical likelihood, so
the orientation must be fixed by convention. See Riedl & Smead (2026).
"""
from __future__ import annotations

import numpy as np


def canonicalize_orientation(b, c, x, a=2.0 / 3.0, d=1.0 / 3.0):
    """Payoff-only tie-break for the residual X<->Y label-swap degeneracy.

    Pinning the diagonals a,d removes the continuous gauge freedom but leaves an
    EXACT discrete label swap: relabeling the two communities maps the inferred
    parameters by (b, c, x) -> (c*d/a, b*a/d, 1-x) and reproduces the identical
    fit (same M, B, ELBO). See Riedl & Smead (2026) for the derivation.

    We fix the orientation by the convention **c/b <= a/d** (equivalently, and
    division-free, c*d <= a*b). If a fit violates it, apply the swap so the
    reported orientation is deterministic across seeds. This
    bundles the two "in-group-favoring" signals onto the SAME label: X is both
    the higher within-group-payoff type (pinned a>d) AND the more homophilous
    type (a/c >= d/b, i.e. the stronger relative in-group preference). It is a
    *declared convention* using only the payoffs; the
    mirror c/b >= a/d is equally valid but would staple the high within-group
    payoff onto the more outward-linking community. It degrades near the pivot
    c/b = a/d (the self-symmetric case where the two orientations coincide).

    Returns:
        (b, c, x, swapped). When swapped is True the caller must ALSO relabel any
        block-indexed arrays it keeps: q columns (q[:, ::-1]), M and B rows+cols
        (X[::-1, ::-1]), and freqs (freqs[::-1]). Within/between density readouts
        are permutation-invariant and need no change.
    """
    if c * d > a * b:                      # c/b > a/d  ->  swap to canonical
        return c * d / a, b * a / d, 1.0 - x, True
    return b, c, x, False


def canonicalize_result(con):
    """Apply the c/b <= a/d orientation convention to a fitted K=2 GTBM result,
    relabeling EVERY block-indexed field in place so the whole result stays
    internally consistent (payoffs, q, pi/freqs, M, B, payoff_matrix,
    free_params, and the DC kappa). No-op unless K==2 and the fit used
    gauge='pin_diag' -- the swap map is only exact when a,d are pinned.

    Because the X<->Y swap is an EXACT likelihood symmetry, every
    gauge-invariant downstream quantity (ELBO, model comparison, held-out
    likelihood, diagnostics) is unchanged; only the orientation of the reported
    payoffs / communities is fixed. Safe to call before or after model comparison.

    NOTE: intended for EM (point-estimate) results. Do NOT use on a Gibbs result:
    a posterior needs per-DRAW alignment before summarizing, not a single swap of
    the posterior-mean payoffs.

    Returns (con, swapped).
    """
    if con.get("K") != 2 or con.get("gauge") != "pin_diag":
        return con, False
    a = float(con.get("a", np.asarray(con["payoff_matrix"])[0, 0]))
    d = float(con.get("d", np.asarray(con["payoff_matrix"])[1, 1]))
    b = float(con["b"])
    c = float(con["c"])
    x = float(con.get("x", np.asarray(con["pi"])[0]))
    nb, nc, nx, swapped = canonicalize_orientation(b, c, x, a=a, d=d)
    if not swapped:
        return con, False
    con["b"], con["c"], con["x"] = nb, nc, nx
    con["free_params"] = np.array([nc, nb], dtype=float)   # stored order [c, b]
    P = np.asarray(con["payoff_matrix"], dtype=float).copy()
    P[0, 1], P[1, 0] = nc, nb                              # diagonal (a,d) unchanged
    con["payoff_matrix"] = P
    con["q"] = np.asarray(con["q"])[:, ::-1]
    pi_rev = np.asarray(con["pi"])[::-1]
    con["pi"] = pi_rev
    con["freqs"] = pi_rev
    for key in ("M", "B"):
        if con.get(key) is not None:
            con[key] = np.asarray(con[key])[::-1, ::-1]
    if con.get("kappa") is not None:                       # per-block DC normalizer
        con["kappa"] = np.asarray(con["kappa"])[::-1]
    return con, swapped
