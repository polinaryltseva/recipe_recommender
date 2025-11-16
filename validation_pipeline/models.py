from __future__ import annotations

from typing import Sequence
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from tqdm.auto import tqdm


def fit_ease(train_matrix: sps.spmatrix, reg_weight: float = 100.0) -> np.ndarray:
    matrix = train_matrix.tocsr().astype(np.float64, copy=False)
    gram = (matrix.T @ matrix).tocoo()
    gram = gram.toarray()
    gram[np.diag_indices_from(gram)] += reg_weight
    inv_gram = np.linalg.inv(gram)
    B = -inv_gram / np.diag(inv_gram)
    np.fill_diagonal(B, 0.0)
    return B


def train_slim(
    train_matrix: sps.csr_matrix,
    l1_reg: float = 1e-3,
    l2_reg: float = 1e-4,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> sps.csr_matrix:
    num_items = train_matrix.shape[1]
    alpha = l1_reg + l2_reg
    if alpha <= 0:
        raise ValueError('l1_reg + l2_reg must sum to a positive value.')
    l1_ratio = l1_reg / alpha
    csc_matrix = train_matrix.tocsc().astype(np.float64, copy=True)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=True,
        fit_intercept=False,
        copy_X=False,
        precompute=False,
        max_iter=max_iter,
        tol=tol,
        warm_start=True,
        selection='random'
    )
    prev_coef = np.zeros(num_items, dtype=np.float64)

    iterator = tqdm(range(num_items), desc='Training SLIM', leave=False)
    for j in iterator:
        start_pos = csc_matrix.indptr[j]
        end_pos = csc_matrix.indptr[j + 1]
        if start_pos == end_pos:
            continue

        y = csc_matrix[:, j].toarray().ravel()

        original_values = csc_matrix.data[start_pos:end_pos].copy()
        csc_matrix.data[start_pos:end_pos] = 0.0

        model.coef_ = prev_coef
        model.intercept_ = 0.0

        model.fit(csc_matrix, y)
        coeffs = model.coef_.copy()
        coeffs[j] = 0.0

        nz = np.flatnonzero(coeffs)
        if nz.size > 0:
            rows.extend(nz.tolist())
            cols.extend([j] * nz.size)
            data.extend(coeffs[nz].tolist())

        prev_coef = coeffs
        csc_matrix.data[start_pos:end_pos] = original_values

    weight_matrix = sps.csr_matrix((data, (rows, cols)), shape=(num_items, num_items))
    return weight_matrix

def train_itemknn(user_item_matrix: sps.spmatrix,
                  topk: int = 200,
                  shrink: float = 100.0,
                  use_binary: bool = True) -> np.ndarray:

    X = user_item_matrix.tocsr().astype(np.float32)
    Xb = X.copy()
    if use_binary:
        Xb.data[:] = 1.0

    co_counts = (Xb.T @ Xb).astype(np.float32).toarray()
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel() + 1e-9)
    denom = (norms[:, None] * norms[None, :]) + 1e-8
    sim = co_counts / denom
    alpha = co_counts / (co_counts + shrink + 1e-9)
    sim *= alpha
    np.fill_diagonal(sim, 0.0)

    if topk and topk < sim.shape[1]:
        idx = np.argpartition(-sim, kth=topk, axis=1)
        mask = np.ones_like(sim, dtype=bool)
        rows = np.arange(sim.shape[0])[:, None]
        mask[rows, idx[:, :topk]] = False
        sim[mask] = 0.0

    return sim.astype(np.float32)


def apply_linear_model(
    item_ids: Sequence[int],
    weight_matrix: np.ndarray | sps.spmatrix,
    top_k: int = 20,
) -> list[int]:

    num_items = weight_matrix.shape[0]
    user_vector = np.zeros(num_items, dtype=np.float64)
    user_vector[list(item_ids)] = 1.0

    if sps.issparse(weight_matrix):
        scores = user_vector @ weight_matrix
        scores = np.asarray(scores).ravel()
    else:
        scores = user_vector @ weight_matrix

    scores[list(item_ids)] = -np.inf
    top_indices = np.argpartition(-scores, kth=min(top_k, num_items - 1))[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return top_indices.tolist()
