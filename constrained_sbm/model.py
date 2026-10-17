"""Game-theoretic partner-choice equations and SBM mapping (K-generic).

The joint edge-type distribution M is derived from a KxK payoff matrix whose
diagonal entries are either pinned (gauge='pin_diag') or derived from row sums
(gauge='row_sum'), with the K(K-1) off-diagonal entries inferred. For K=2 a 2x2 convenience wrapper is provided.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
PROB_FLOOR = 1e-10
PROB_CEIL = 1.0 - 1e-10

# Default pinned diagonals by K. K=3 uses (2/3, 1/2, 1/3).
DEFAULT_PINNED_DIAG = {
     2: (2.0 / 3.0, 1.0 / 3.0),
     3: (2.0 / 3.0, 0.5, 1.0 / 3.0),
}


def n_free_params(K):
    """Number of off-diagonal payoff entries for a KxK game."""
    return K * K - K


def _payoff_matrix(free_params, K=None, pinned_diag=None, gauge="pin_diag"):
    """Reconstruct the full KxK payoff matrix.

    free_params is in row-major off-diagonal order:
        P[0,1], P[0,2], ..., P[0,K-1], P[1,0], P[1,2], ..., P[K-1,K-2].

    Args:
        free_params: array of length K^2 - K.
        K: number of blocks. If None, inferred from pinned_diag.
        pinned_diag: pinned diagonal values, required when gauge == "pin_diag".
        gauge: "pin_diag" (use pinned_diag for diagonals) or "row_sum"
            (diagonals derived as 1 − sum of off-diagonals in that row).

    Returns:
        P: (K, K) payoff matrix.
    """
    if K is None:
        if pinned_diag is None:
            raise ValueError("Need either K or pinned_diag to determine K.")
        K = len(pinned_diag)
    P = np.zeros((K, K), dtype=float)
    idx = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                P[i, j] = free_params[idx]
                idx += 1
    if gauge == "pin_diag":
        if pinned_diag is None:
            raise ValueError("pinned_diag is required when gauge='pin_diag'.")
        for i in range(K):
            P[i, i] = pinned_diag[i]
    elif gauge == "row_sum":
        for i in range(K):
            off_sum = P[i, :].sum()  # diagonal still 0, so this is the off-diag sum
            P[i, i] = max(1.0 - off_sum, EPS)
    else:
        raise ValueError(f"Unknown gauge: {gauge!r}")
    return P


def compute_M_K(free_params, freqs, beta, pinned_diag=None, gauge="pin_diag"):
    """K-generic joint edge-type distribution.

    Args:
        free_params: array-like of length K^2 - K (off-diagonal payoffs).
        freqs: array-like of length K (type frequencies, summing to 1).
        beta: inverse temperature.
        pinned_diag: array-like of length K. Required when gauge='pin_diag';
            ignored (and may be None) when gauge='row_sum'.
        gauge: 'pin_diag' (default) or 'row_sum'. Under 'row_sum' each row of
            the payoff matrix sums to 1 and the diagonals are derived from the
            off-diagonals; under 'pin_diag' the diagonals are pinned.

    Returns:
        M: (K, K) array summing to 1.
    """
    freqs = np.asarray(freqs, dtype=float)
    free_params = np.asarray(free_params, dtype=float).ravel()
    K = len(freqs)
    if gauge == "pin_diag":
        if pinned_diag is None:
            raise ValueError("pinned_diag is required when gauge='pin_diag'.")
        pinned_diag = np.asarray(pinned_diag, dtype=float)
        assert len(pinned_diag) == K, "pinned_diag length must equal K"
    elif gauge != "row_sum":
        raise ValueError(f"Unknown gauge: {gauge!r}")
    assert len(free_params) == K * K - K, f"free_params must have length {K*K-K}"

    P = _payoff_matrix(free_params, K=K, pinned_diag=pinned_diag, gauge=gauge)
    Pb = np.power(P, beta)
    Z = Pb @ freqs + EPS
    M = (freqs[:, None] * freqs[None, :]) * Pb / Z[:, None]
    total = M.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full((K, K), 1.0 / (K * K))
    return M / total


def M_to_B_K(M, freqs, n, num_edges, directed=False, degree_corrected=False):
    """K-generic mapping from joint edge-type distribution M to SBM matrix B.

    Three link modes:

    1. Directed (any model): no symmetrization; the asymmetric directed flow
       passes straight through.
           B[k,l] = M[k,l] * |E_dir| / (n_k * n_l)            (k != l)
           B[k,k] = M[k,k] * |E_dir| / (n_k * (n_k - 1))      (k = l)

    2. Undirected, Non-Degree-Corrected (Bernoulli) — probabilistic "OR" of
       the two directed flows. P_{k→l} is the per-ordered-pair directed rate;
       the undirected edge exists iff k→l OR l→k.
           P_{k→l} = M[k,l] * |E| / (n_k * n_l)        (k != l)
           B[k,l]  = 1 − (1 − P_{k→l}) · (1 − P_{l→k})
           P_{k→k} = M[k,k] * |E| / (n_k * (n_k - 1))   (k = l)
           B[k,k]  = 1 − (1 − P_{k→k})²
       Bounded in [0, 1] by construction.

    3. Undirected, Degree-Corrected (Poisson) — linear sum of rates. The
       probabilistic OR of two independent Poisson processes is rate-addition:
           B[k,l] = (M[k,l] + M[l,k]) * |E| / (n_k * n_l)      (k != l)
           B[k,k] = 2 · M[k,k] * |E| / (n_k * (n_k - 1))       (k = l)
       B is a rate parameter and may legitimately exceed 1 — it is scaled
       down per-node by θ_i during likelihood evaluation, and is not clipped
       from above.

    A small floor (PROB_FLOOR) is applied for downstream log-stability.

    Args:
        M: (K, K) joint edge-type distribution from partner-choice equations.
        freqs: (K,) class frequencies (sum to 1).
        n: total number of nodes.
        num_edges: edge count. Directed edge count (= nnz of A) when
            directed=True; undirected edge count (= nnz // 2) otherwise.
        directed: if True, return the asymmetric M-scaled matrix without
            symmetrization (directed link mode above).
        degree_corrected: ignored when directed=True. For undirected graphs,
            selects between Bernoulli OR (False) and Poisson rate-sum (True).

    Returns:
        B: (K, K) edge-probability (or rate) matrix.
    """
    freqs = np.asarray(freqs, dtype=float)
    K = len(freqs)
    n_per = np.maximum(freqs * n, EPS)
    E = float(num_edges)

    if directed:
        denom = np.outer(n_per, n_per)
        np.fill_diagonal(denom, n_per * (n_per - 1.0))
        B = M * E / np.maximum(denom, EPS)
        return np.maximum(B, PROB_FLOOR)

    B = np.zeros((K, K), dtype=float)
    if degree_corrected:
        # Poisson rate addition: both directed flows contribute additively.
        for k in range(K):
            denom = n_per[k] * (n_per[k] - 1.0)
            B[k, k] = 2.0 * M[k, k] * E / max(denom, EPS)
        for i in range(K):
            for j in range(i + 1, K):
                denom = n_per[i] * n_per[j]
                rate = (M[i, j] + M[j, i]) * E / max(denom, EPS)
                B[i, j] = rate
                B[j, i] = rate
        return np.maximum(B, PROB_FLOOR)

    # Bernoulli probabilistic OR link.
    for k in range(K):
        denom = n_per[k] * (n_per[k] - 1.0)
        P_kk = M[k, k] * E / max(denom, EPS)
        P_kk = min(max(P_kk, 0.0), 1.0)
        B[k, k] = 1.0 - (1.0 - P_kk) ** 2
    for i in range(K):
        for j in range(i + 1, K):
            denom = n_per[i] * n_per[j]
            P_ij = M[i, j] * E / max(denom, EPS)
            P_ji = M[j, i] * E / max(denom, EPS)
            P_ij = min(max(P_ij, 0.0), 1.0)
            P_ji = min(max(P_ji, 0.0), 1.0)
            B_und = 1.0 - (1.0 - P_ij) * (1.0 - P_ji)
            B[i, j] = B_und
            B[j, i] = B_und
    return np.maximum(B, PROB_FLOOR)


def compute_M(b, c, x, beta, a=2.0 / 3.0, d=1.0 / 3.0,
              pinned_diag=None, gauge=None):
    """K=2 convenience wrapper (positional b, c, x, beta).

    The 2x2 payoff matrix is [[a, c], [b, d]]; row-major off-diagonal order
    is [P[0,1], P[1,0]] = [c, b].

    Gauge defaulting:
        - gauge=None   → "row_sum" (the default for K=2)
        - gauge="pin_diag" → diagonals taken from (a, d) or pinned_diag
        - gauge="row_sum"  → diagonals derived as 1-c and 1-b; a/d/pinned_diag ignored
    """
    if gauge is None:
        gauge = "row_sum"
    if gauge == "pin_diag":
        if pinned_diag is None:
            pinned_diag = [a, d]
    return compute_M_K(
        free_params=[c, b],
        freqs=[x, 1.0 - x],
        beta=beta,
        pinned_diag=pinned_diag if gauge == "pin_diag" else None,
        gauge=gauge,
    )


def M_to_B(M, x, n, num_edges, directed=False, degree_corrected=False):
    """K=2 wrapper around `M_to_B_K`.

    `directed` and `degree_corrected` select the link function — see
    `M_to_B_K` for details. The defaults give the undirected Bernoulli OR link.
    """
    return M_to_B_K(M, [x, 1.0 - x], n, num_edges,
                    directed=directed, degree_corrected=degree_corrected)


def compute_M_conditional(M):
    """Row-normalize M to obtain conditional partner-choice probs pi(J|I)."""
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return M / row_sums


def getGraphType(labels):
    """Classify a 2x2 game by the signs of (a-c) and (b-d).

    Args:
        labels: array-like [a, b, c, d] or shape (n, 4).

    Returns:
        np.array of strings, one per (a, b, c, d) row.
    """
    labels = np.asarray(labels).reshape(-1, 4)
    ac = labels[:, 0] - labels[:, 2]
    bd = labels[:, 1] - labels[:, 3]

    result = np.full(len(labels), "ambiguous", dtype=object)
    result[(ac > 0) & (bd > 0)] = "pair X"
    result[(ac < 0) & (bd > 0)] = "pair diff"
    result[(ac > 0) & (bd < 0)] = "pair same"
    result[(ac < 0) & (bd < 0)] = "pair Y"
    return result


def getGameType(labels):
    """Classify a 2x2 game by the relative ordering of payoffs.

    Classifies the 2x2 game by payoff ordering:

        delight     : a > b AND c > d
        dilemma     : b > a AND d > c
        conflict    : b > a AND c > d        (hawk-dove)
        coordination: a > b AND d > c        (stag hunt)
        boundary case: otherwise

    Args:
        labels: array-like [a, b, c, d] or shape (n, 4).

    Returns:
        np.array of strings: 'delight', 'dilemma', 'conflict',
        'coordination', or 'boundary case'.
    """
    labels = np.asarray(labels).reshape(-1, 4)
    a, b, c, d = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

    result = np.full(len(labels), "boundary case", dtype=object)
    result[(a > b) & (c > d)] = "delight"
    result[(b > a) & (d > c)] = "dilemma"
    result[(b > a) & (c > d)] = "conflict"
    result[(a > b) & (d > c)] = "coordination"
    return result
