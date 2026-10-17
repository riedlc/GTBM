"""Graph I/O and sparse matrix helpers."""

from __future__ import annotations

import contextlib
import logging
import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def load_graph(path, symmetrize=True):
    """Load a graph from edgelist, .npz adjacency, or .graphml/.gml.

    Args:
        path: file path. Edgelists are read with `nx.read_edgelist` (undirected
            by default) and graphml/gml preserve the file's stated directedness.
        symmetrize: if True (default), symmetrize the loaded matrix via
            `A.maximum(A.T)`. Pass False to keep the file's directedness for a
            directed-mode fit.

    Returns:
        scipy.sparse.csr_matrix with binary entries and zero diagonal. Symmetric
        only when symmetrize=True (or the underlying file is symmetric).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        A = sp.load_npz(path)
    elif ext in (".graphml", ".gml"):
        if ext == ".graphml":
            G = nx.read_graphml(path)
        else:
            G = nx.read_gml(path)
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    else:
        # treat as a whitespace-separated edgelist with integer node ids
        G = nx.read_edgelist(path, nodetype=int, data=False)
        A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), format="csr", dtype=float)

    A = sp.csr_matrix(A)
    if symmetrize:
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data = np.ones_like(A.data)
    return A


def is_symmetric(A, tol=0):
    """Whether the adjacency matrix is structurally symmetric.

    With tol=0 (default), checks exact equality of nnz patterns and values via
    `(A - A.T).nnz == 0`. A nonzero tolerance compares the dense max-abs
    difference and is only intended for weighted matrices — for binary
    adjacencies the structural check is exact and fast.
    """
    A = sp.csr_matrix(A)
    diff = (A - A.T)
    if tol == 0:
        return diff.nnz == 0
    if diff.nnz == 0:
        return True
    return float(np.abs(diff.data).max()) <= tol


def is_directed(A):
    """Convenience: True iff the matrix is not structurally symmetric."""
    return not is_symmetric(A)


def num_undirected_edges(A):
    """Count distinct undirected edges in a symmetric adjacency matrix."""
    return int(A.nnz // 2)


def num_directed_edges(A):
    """Count distinct directed edges (= nnz of A)."""
    return int(A.nnz)


def edge_count(A, directed):
    """Edge count appropriate for the fit mode.

    For directed=True this is the directed nnz; for undirected, the number
    of undirected edges (nnz // 2). Supplies the |E| factor passed to
    `M_to_B_K`.
    """
    return num_directed_edges(A) if directed else num_undirected_edges(A)


def resolve_directed_mode(A, user_choice="auto"):
    """Resolve the requested directed/undirected mode against the data.

    Args:
        A: loaded adjacency matrix.
        user_choice: one of {'auto', 'yes', 'no'} (or the booleans True/False
            for explicit override). 'auto' inspects the symmetry of A.

    Returns:
        (directed: bool, banner: str). The banner is a multi-line diagnostic
        string suitable for printing at the top of a CLI run. It always
        states the chosen mode, why it was chosen, and warns if the choice
        is inconsistent with the loaded matrix (e.g., directed mode forced
        on a symmetric A).
    """
    A_sym = is_symmetric(A)
    A_kind = "symmetric" if A_sym else "asymmetric"
    if user_choice is True:
        user_choice = "yes"
    elif user_choice is False:
        user_choice = "no"

    if user_choice == "auto":
        directed = not A_sym
        source = "auto-detected from adjacency symmetry"
    elif user_choice == "yes":
        directed = True
        source = "forced by --directed yes"
    elif user_choice == "no":
        directed = False
        source = "forced by --directed no"
    else:
        raise ValueError(f"Unknown directed choice: {user_choice!r}")

    label = "DIRECTED" if directed else "UNDIRECTED"
    lines = [
        "-" * 60,
        f"Inference mode: {label}  ({source})",
        f"  adjacency is {A_kind} (n={A.shape[0]}, nnz={int(A.nnz)})",
    ]
    if directed and A_sym:
        lines.append(
            "  WARNING: directed fit on a symmetric adjacency doubles each\n"
            "           edge's contribution vs. an undirected fit; results\n"
            "           are not directly comparable to the undirected ELBO."
        )
    if (not directed) and (not A_sym):
        lines.append(
            "  Note: the asymmetric adjacency is symmetrized via A.maximum(A.T)\n"
            "        before the undirected fit."
        )
    lines.append("-" * 60)
    return directed, "\n".join(lines)


def spectral_init(A, n_classes=2, seed=None):
    """Soft initial assignments from bottom Laplacian eigenvectors.

    K=2: threshold the Fiedler vector by its median. K>=3: take the K-1
    smallest non-trivial eigenvectors and apply K-means on that spectral
    embedding.
    """
    from sklearn.cluster import KMeans

    n = A.shape[0]
    K = n_classes
    rng = np.random.default_rng(seed)

    deg = np.asarray(A.sum(axis=1)).ravel()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1.0)))
    L_norm = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = (L_norm + L_norm.T) * 0.5

    k_eig = min(K + 1, max(n - 1, 2))
    try:
        # Seed ARPACK's start vector from `seed`. Without v0, eigsh draws it from
        # numpy's *global* RNG, so two runs with the same `seed` could land in
        # different basins. The multi-eigenvector path (spectral_evec_inits)
        # seeds v0 the same way.
        vals, vecs = eigsh(L_norm, k=k_eig, which="SM", tol=1e-4, maxiter=2000,
                           v0=rng.standard_normal(n))
        order = np.argsort(vals)
        # skip the trivial eigenvector (smallest), take next K-1
        emb = vecs[:, order[1:K]] if K > 1 else vecs[:, order[:1]]
    except Exception:
        emb = rng.standard_normal((n, max(K - 1, 1)))

    if K == 2:
        fiedler = emb[:, 0]
        hard = (fiedler >= np.median(fiedler)).astype(int)
    else:
        km = KMeans(n_clusters=K, n_init=5, random_state=int(rng.integers(0, 1 << 31)))
        hard = km.fit_predict(emb)

    low = 0.1 / max(K - 1, 1)
    q = np.full((n, K), low)
    q[np.arange(n), hard] = 0.9
    q += 1e-3 * rng.standard_normal(q.shape)
    q = np.clip(q, 1e-3, 1.0)
    q /= q.sum(axis=1, keepdims=True)
    return q


def spectral_evec_inits(A, n_classes=2, n_evecs=4, seed=None):
    """Eigenvector-diverse initialisations: a list of soft assignments, one
    seeded from each of the first `n_evecs` non-trivial normalized-Laplacian
    eigenvectors.

    For some graphs the community axis is not the first non-trivial eigenvector
    (e.g. when the Fiedler vector tracks a degree or core-periphery axis). This
    helper seeds restarts from several low eigenvectors so the best-ELBO restart
    can reach the assortative basin.

    It supplements (does not replace) the random/kmeans schedule, so random
    restarts still cover non-assortative basins. The trivial (near-zero)
    eigenvectors are dropped by eigenvalue magnitude, so the result does not
    depend on whether the solver returns them.

    Returns a list of (n, K) soft-q arrays (possibly shorter than n_evecs if the
    graph is tiny).
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    n = A.shape[0]
    K = n_classes
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1.0)))
    L_norm = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = (L_norm + L_norm.T) * 0.5

    want = n_evecs + 2  # a couple extra to absorb the trivial null space
    # ARPACK which='SA' with a seeded v0 for determinism; dense fallback if it
    # fails or the graph is tiny. (Near-)zero trivial vectors are dropped below.
    w = V = None
    k_eig = min(want, n - 1)
    if k_eig >= 1 and n > k_eig + 1:
        try:
            v0 = rng.standard_normal(n)
            w, V = eigsh(L_norm, k=k_eig, which="SA", tol=1e-6, maxiter=5000, v0=v0)
        except Exception:
            w = V = None
    if w is None:
        try:
            w, V = np.linalg.eigh(L_norm.toarray())
        except Exception:
            return []
    order = np.argsort(w)
    w, V = w[order], V[:, order]

    nontrivial = np.where(w > 1e-8)[0]      # drop the trivial null space
    cols = nontrivial[:n_evecs]
    inits = []
    for j, col in enumerate(cols):
        if K == 2:
            f = V[:, col]
            hard = (f >= np.median(f)).astype(int)
        else:
            # cumulative embedding [first non-trivial .. this one], >= K-1 dims
            emb_cols = nontrivial[: max(j + 1, K - 1)]
            emb = V[:, emb_cols]
            hard = KMeans(n_clusters=K, n_init=5,
                          random_state=int(rng.integers(0, 1 << 31))).fit_predict(emb)
        low = 0.1 / max(K - 1, 1)
        q = np.full((n, K), low)
        q[np.arange(n), hard] = 0.9
        q += 1e-3 * rng.standard_normal(q.shape)
        q = np.clip(q, 1e-3, 1.0)
        q /= q.sum(axis=1, keepdims=True)
        inits.append(q)
    return inits


def random_init(n, n_classes=2, seed=None):
    """Soft assignments with mild random asymmetry to avoid collapse."""
    rng = np.random.default_rng(seed)
    K = n_classes
    if K == 2:
        q0 = rng.uniform(0.3, 0.7, size=n)
        q = np.column_stack([q0, 1.0 - q0])
        return q
    q = rng.uniform(0.2, 0.6, size=(n, K))
    q /= q.sum(axis=1, keepdims=True)
    return q


def kmeans_init(A, n_classes=2, seed=None):
    """K-means on a low-rank SVD of A for an alternative initializer."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    n = A.shape[0]
    k = min(n_classes + 2, max(n - 1, 2))
    try:
        u, _, _ = sp.linalg.svds(A.asfptype(), k=k)
    except Exception:
        u = rng.standard_normal((n, k))
    km = KMeans(n_clusters=n_classes, n_init=5, random_state=int(rng.integers(0, 1 << 31)))
    labels = km.fit_predict(u)
    q = np.full((n, n_classes), 0.1)
    q[np.arange(n), labels] = 0.9
    q /= q.sum(axis=1, keepdims=True)
    return q


def directed_spectral_init(A, n_classes=2, seed=None, gauge_order=False):
    """Directionality-preserving initial assignments via SVD co-clustering.

    Unlike `spectral_init` (which symmetrizes A and uses the bottom Laplacian
    eigenvectors, recovering only assortative communities), this embeds each
    node by BOTH its left and right singular vectors of the un-symmetrized
    adjacency. The left coordinates encode the out-profile ("who I point to")
    and the right coordinates the in-profile ("who points to me"), so the
    embedding captures the directional role that distinguishes positions in
    cyclic / bipartite / core-periphery structure (DI-SIM, Rohe-Qin-Yu 2016).
    K-means on that embedding gives the hard assignment.

    When `gauge_order` is True, blocks are relabeled in descending order of
    directed self-edge density so that label 0 is the densest self-block. This
    matches a descending pinned diagonal (e.g. 2/3 > 1/2 > 1/3) and seeds EM in
    the gauge-consistent rotation, avoiding the label-permutation ("rotation")
    lock that the weak diagonal anchor cannot otherwise break.
    """
    from sklearn.cluster import KMeans

    A = sp.csr_matrix(A).asfptype()
    n = A.shape[0]
    K = n_classes
    rng = np.random.default_rng(seed)

    k = min(K, max(min(A.shape) - 1, 1))
    try:
        U, S, Vt = sp.linalg.svds(A, k=k)
        sqrt_s = np.sqrt(np.maximum(S, 0.0))
        emb = np.hstack([U * sqrt_s, Vt.T * sqrt_s])  # (n, 2k): out- and in-profile
    except Exception:
        emb = rng.standard_normal((n, max(2 * K, 2)))

    km = KMeans(n_clusters=K, n_init=5,
                random_state=int(rng.integers(0, 1 << 31)))
    hard = km.fit_predict(emb)

    if gauge_order:
        dens = np.zeros(K)
        for kk in range(K):
            members = np.where(hard == kk)[0]
            nk = members.size
            if nk >= 2:
                sub = A[members][:, members]
                dens[kk] = sub.nnz / (nk * (nk - 1.0))
        order = np.argsort(-dens)              # densest self-block first
        relabel = np.empty(K, dtype=int)
        relabel[order] = np.arange(K)          # old label -> gauge-ordered rank
        hard = relabel[hard]

    low = 0.1 / max(K - 1, 1)
    q = np.full((n, K), low)
    q[np.arange(n), hard] = 0.9
    q += 1e-3 * rng.standard_normal(q.shape)
    q = np.clip(q, 1e-3, 1.0)
    q /= q.sum(axis=1, keepdims=True)
    return q


def type_labels(K):
    """Single-letter labels for K blocks (X, Y, Z, ...)."""
    base = "XYZWVUTSRQ"
    if K <= len(base):
        return list(base[:K])
    return [f"T{i}" for i in range(K)]


def fmt_matrix(M, K=None, decimals=4):
    """Pretty-print a KxK matrix with type-letter row/column headers."""
    if K is None:
        K = M.shape[0]
    labels = type_labels(K)
    cell_w = decimals + 4
    header = "    " + " ".join(f"{lbl:>{cell_w}}" for lbl in labels)
    rows = [header]
    for i in range(K):
        cells = " ".join(f"{M[i, j]:>{cell_w}.{decimals}f}" for j in range(K))
        rows.append(f"{labels[i]:<3} {cells}")
    return "\n".join(rows)


def fmt_payoff_matrix(P, pinned_diag=None, decimals=3):
    """Format the KxK payoff matrix. Diagonal entries are marked with `*` and
    labelled as pinned when `pinned_diag` is given, otherwise as derived."""
    K = P.shape[0]
    labels = type_labels(K)
    cell_w = decimals + 4
    header = "    " + " ".join(f"{lbl:>{cell_w}}" for lbl in labels)
    rows = [header]
    for i in range(K):
        cells = []
        for j in range(K):
            val = f"{P[i, j]:.{decimals}f}"
            val = val + ("*" if i == j else " ")  # diagonal marker
            cells.append(f"{val:>{cell_w + 1}}")
        rows.append(f"{labels[i]:<3} " + " ".join(cells))
    rows.append("(* = pinned diagonal)" if pinned_diag is not None
                else "(* = diagonal, derived from row sums)")
    return "\n".join(rows)


def fmt_freqs(freqs, decimals=4):
    """Format class frequencies as a comma-separated string."""
    labels = type_labels(len(freqs))
    return ", ".join(f"{l}={f:.{decimals}f}" for l, f in zip(labels, freqs))


def hard_cluster_counts(q):
    """Hard-assignment counts per class via argmax over q's rows."""
    K = q.shape[1]
    hard = q.argmax(axis=1)
    return np.bincount(hard, minlength=K)


def fmt_freqs_with_counts(freqs, q, decimals=3):
    """`X=0.481 (16), Y=0.372 (18), Z=0.147 (0)` — soft mean plus hard count."""
    counts = hard_cluster_counts(q)
    labels = type_labels(len(freqs))
    return ", ".join(
        f"{l}={f:.{decimals}f} ({c})"
        for l, f, c in zip(labels, freqs, counts)
    )


def get_logger(name="constrained_sbm", verbose=True):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def limit_blas_threads(n=1):
    """Temporarily pin BLAS-related thread env vars to `n`.

    Child processes spawned during the block inherit the limits. This prevents
    BLAS oversubscription when running per-chain worker processes that each
    launch their own BLAS thread pool.
    """
    old = {k: os.environ.get(k) for k in _BLAS_THREAD_ENV_VARS}
    try:
        for k in _BLAS_THREAD_ENV_VARS:
            os.environ[k] = str(n)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_chains_parallel(chain_fn, chain_args_list, n_jobs):
    """Run independent Gibbs chains, optionally in parallel processes.

    `chain_fn` must be a module-level (picklable) function taking one
    positional argument — the per-chain kwargs dict — and returning that
    chain's result. With `n_jobs <= 1`, chains run sequentially in-process.
    With `n_jobs > 1`, chains run via `ProcessPoolExecutor` using the spawn
    start method; BLAS thread env vars are pinned to 1 for the duration so
    workers don't oversubscribe cores. Results are returned in the order of
    `chain_args_list`.
    """
    n = len(chain_args_list)
    if n_jobs is None or n_jobs <= 1 or n <= 1:
        return [chain_fn(args) for args in chain_args_list]

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    workers = min(int(n_jobs), n)
    with limit_blas_threads(1):
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            return list(ex.map(chain_fn, chain_args_list))
