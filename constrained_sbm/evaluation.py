"""Held-out evaluation, link prediction, and constrained-vs-free comparison."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from constrained_sbm.baseline_sbm import fit_free
from constrained_sbm.variational_em import fit as fit_constrained


def _edge_probs(pair_i, pair_j, q, B, theta=None):
    """Predicted edge probability for ordered pairs (i, j).

    If theta is provided (degree-corrected fit), p_ij = θ_i · θ_j · Σ_kl q_i(k) B[k,l] q_j(l),
    clamped to [0, 1].
    """
    qi = q[pair_i]
    qj = q[pair_j]
    base = np.einsum("mk,kl,ml->m", qi, B, qj)
    if theta is not None:
        base = base * theta[pair_i] * theta[pair_j]
    return np.clip(base, 0.0, 1.0)


def held_out_log_likelihood(test_pos, test_neg, q, B, theta=None):
    """Log-likelihood of held-out positive and negative edges under the fit."""
    if len(test_pos) > 0:
        p_pos = _edge_probs(test_pos[:, 0], test_pos[:, 1], q, B, theta=theta)
        p_pos = np.clip(p_pos, 1e-12, 1.0 - 1e-12)
        ll_pos = float(np.sum(np.log(p_pos)))
    else:
        ll_pos = 0.0
    if len(test_neg) > 0:
        p_neg = _edge_probs(test_neg[:, 0], test_neg[:, 1], q, B, theta=theta)
        p_neg = np.clip(p_neg, 1e-12, 1.0 - 1e-12)
        ll_neg = float(np.sum(np.log(1.0 - p_neg)))
    else:
        ll_neg = 0.0
    return ll_pos + ll_neg


def link_prediction_scores(test_pos, test_neg, q, B, k=500, theta=None):
    """AUROC and precision@k for held-out edges vs non-edges."""
    s_pos = _edge_probs(test_pos[:, 0], test_pos[:, 1], q, B, theta=theta)
    s_neg = _edge_probs(test_neg[:, 0], test_neg[:, 1], q, B, theta=theta)
    scores = np.concatenate([s_pos, s_neg])
    labels = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    k_eff = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k_eff]
    p_at_k = float(labels[top_idx].mean())
    return {"auroc": auroc, f"p_at_{k}": p_at_k}


def split_edges_kfold(A, n_folds=10, seed=42):
    """Split undirected edges into k folds, sampling matched negative edges.

    Returns a list of (A_train, test_pos_edges, test_neg_edges) tuples.
    A_train is symmetric with test edges removed. test edges are arrays of
    shape (m, 2) with i < j.
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    # upper triangle edges (i < j)
    coo = sp.triu(A, k=1).tocoo()
    edges = np.column_stack([coo.row, coo.col])
    num_edges = len(edges)
    perm = rng.permutation(num_edges)
    folds = np.array_split(perm, n_folds)

    # global edge set for negative sampling
    edge_set = set(map(tuple, edges.tolist()))

    splits = []
    for f in range(n_folds):
        test_idx = folds[f]
        test_pos = edges[test_idx]

        # build training adjacency by removing test edges
        keep_mask = np.ones(num_edges, dtype=bool)
        keep_mask[test_idx] = False
        train_edges = edges[keep_mask]
        rows = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
        cols = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
        data = np.ones(2 * len(train_edges), dtype=float)
        A_train = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        A_train.setdiag(0)
        A_train.eliminate_zeros()

        # sample matched negatives outside the full edge set, with i < j
        target = len(test_pos)
        negatives = []
        attempts = 0
        while len(negatives) < target and attempts < target * 50:
            cand = rng.integers(0, n, size=(target * 4, 2))
            for i, j in cand:
                if i == j:
                    continue
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if (a, b) in edge_set:
                    continue
                negatives.append((a, b))
                if len(negatives) >= target:
                    break
            attempts += len(cand)
        test_neg = np.array(negatives[:target], dtype=int)

        splits.append((A_train, test_pos, test_neg))

    return splits


def compare_models(A, n_folds=10, beta=3.0, n_restarts=10, seed=42,
                   verbose=False, K=2, degree_corrected=False, gauge=None,
                   n_spectral=1, max_iter=200, tol=1e-6):
    """k-fold comparison (default 10) of constrained vs free SBM with link prediction.

    When degree_corrected=True, both fits use the Poisson DC-SBM and the held-out
    edge scores are weighted by the recovered θ.
    """
    splits = split_edges_kfold(A, n_folds=n_folds, seed=seed)

    rows = []
    for f, (A_train, test_pos, test_neg) in enumerate(splits):
        con = fit_constrained(
            A_train, K=K, beta=beta, n_restarts=n_restarts,
            seed=seed + f, verbose=verbose,
            degree_corrected=degree_corrected,
            gauge=gauge, n_spectral=n_spectral, max_iter=max_iter, tol=tol,
        )
        free = fit_free(
            A_train, K=K, n_restarts=n_restarts,
            seed=seed + f, verbose=verbose,
            degree_corrected=degree_corrected,
            n_spectral=n_spectral, max_iter=max_iter, tol=tol,
            # Warm-start from the constrained fit so the free ELBO is >= the
            # constrained ELBO by construction.
            warm_start_q=con.get("q"),
            warm_start_theta=con.get("theta"),
        )
        con_theta = con.get("theta") if degree_corrected else None
        free_theta = free.get("theta") if degree_corrected else None
        con_ll = held_out_log_likelihood(test_pos, test_neg, con["q"], con["B"], theta=con_theta)
        free_ll = held_out_log_likelihood(test_pos, test_neg, free["q"], free["B"], theta=free_theta)
        con_lp = link_prediction_scores(test_pos, test_neg, con["q"], con["B"], theta=con_theta)
        free_lp = link_prediction_scores(test_pos, test_neg, free["q"], free["B"], theta=free_theta)
        row = {
            "fold": f,
            "constrained_elbo": con["elbo"],
            "free_elbo": free["elbo"],
            "constrained_test_ll": con_ll,
            "free_test_ll": free_ll,
            "constrained_auroc": con_lp["auroc"],
            "free_auroc": free_lp["auroc"],
            "constrained_p_at_500": con_lp["p_at_500"],
            "free_p_at_500": free_lp["p_at_500"],
        }
        if K == 2:
            row["b"] = con["b"]
            row["c"] = con["c"]
            row["x"] = con["x"]
        rows.append(row)

    def mean_std(key):
        vals = np.array([r[key] for r in rows])
        return float(vals.mean()), float(vals.std())

    summary = {}
    for k in [
        "constrained_elbo", "free_elbo",
        "constrained_test_ll", "free_test_ll",
        "constrained_auroc", "free_auroc",
        "constrained_p_at_500", "free_p_at_500",
    ]:
        m, s = mean_std(k)
        summary[k] = {"mean": m, "std": s}

    return {"folds": rows, "summary": summary}
