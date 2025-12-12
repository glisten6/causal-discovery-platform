import torch
import numpy as np
import logging
import sys
import os
from beartype import beartype
from copy import deepcopy
from collections import defaultdict
from functools import partial
from causallearn.utils.cit import CIT
from functools import partial
from scipy.stats import norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms.pc.pc import find_skeleton

import pybnesian as pbn
import numpy as np
import pandas as pd
import traceback

def gaussianize_rank_transform_pandas(X, clip_quantile=1e-6, keep_nan=True):
    """
    Rank-based Gaussianization using pandas.DataFrame.rank(method='average').
    Vectorized, handles ties via average rank, preserves NaNs.

    Args:
        X: numpy array shape (n, d)
        clip_quantile: float, clip u into [clip_quantile, 1-clip_quantile]
        keep_nan: if True, NaNs are preserved in output (remain NaN)

    Returns:
        Xg: numpy array shape (n, d)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D array (n, d).")
    n, d = X.shape

    # build DataFrame (pandas preserves NaNs)
    df = pd.DataFrame(X)

    # rank per column, method='average' handles ties. ranks in 1..n (na -> NaN)
    ranks = df.rank(method='average', axis=0, pct=False)
    r = ranks.to_numpy(dtype=float)  # shape (n, d)

    # count non-NaN per column
    non_na_mask = ~np.isnan(r)
    denom = np.sum(non_na_mask, axis=0)  # shape (d,)
    denom_safe = np.where(denom == 0, 1.0, denom)  # avoid divide by zero

    # compute u with broadcasting. r has NaN where missing; arithmetic preserves NaN.
    u = (r - 0.5) / denom_safe[np.newaxis, :]  # shape (n, d)
    # clip into (clip_quantile, 1-clip_quantile)
    u = np.clip(u, clip_quantile, 1.0 - clip_quantile)

    # inverse CDF (preserve NaNs)
    Xg = norm.ppf(u)
    if keep_nan:
        # ensure original NaNs remain NaN in output (norm.ppf on clipped values won't be NaN,
        # so reintroduce NaNs where original r was NaN)
        Xg[~non_na_mask] = np.nan
    return Xg



from collections import defaultdict

from collections import defaultdict, Counter


def RCOT_Test(data, x, y, z=None, *, random_fourier_xy=5, random_fourier_z=100, **kwargs):
    """Run RCoT conditional independence test using pybnesian.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input continuous data.
    x, y : int
        Column indices (zero-based) for the two variables being tested.
    z : None | int | Iterable[int]
        Conditioning set column indices.
    random_fourier_xy : int, optional
        Number of random Fourier features for X/Y variables (default 5).
    random_fourier_z : int, optional
        Number of random Fourier features for conditioning variables (default 100).
    **kwargs : dict
        Additional keyword arguments forwarded to ``pybnesian.RCoT``.

    Returns
    -------
    tuple
        A tuple ``(None, None, p_value)`` matching Castle's CI-test API.
    """
    if not hasattr(pbn, "RCoT"):
        raise ImportError("pybnesian.RCoT is unavailable; please ensure pybnesian version supports it.")

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_samples, n_features)")

    def _to_int_list(indices):
        if indices is None:
            return []
        if np.isscalar(indices):
            return [int(indices)]
        return [int(idx) for idx in indices]

    x_idx, y_idx = int(x), int(y)
    z_indices = _to_int_list(z)

    ordered_cols = []
    for idx in [x_idx, y_idx] + z_indices:
        if idx not in ordered_cols:
            ordered_cols.append(idx)

    if max(ordered_cols, default=-1) >= data_arr.shape[1] or min(ordered_cols, default=0) < 0:
        raise IndexError("RCOT_Test column indices exceed data dimensions")

    # Build DataFrame for pybnesian (use string column names)
    subset = data_arr[:, ordered_cols]
    column_names = [str(idx) for idx in ordered_cols]
    df_subset = pd.DataFrame(subset, columns=column_names)

    rcot = pbn.RCoT(
        df_subset,
        random_fourier_xy=random_fourier_xy,
        random_fourier_z=random_fourier_z,
        **kwargs,
    )
    x_name = column_names[0]
    y_name = column_names[1]
    cond_names = column_names[2:]
    try:
        if cond_names:
            p_value = float(rcot.pvalue(x_name, y_name, cond_names))
        else:
            p_value = float(rcot.pvalue(x_name, y_name))
    except Exception as exc:
        raise RuntimeError(
            f"pybnesian.RCoT.pvalue failed for x={x_name}, y={y_name}, z={cond_names}"
        ) from exc

    if not np.isfinite(p_value):
        raise ValueError(
            f"pybnesian.RCoT returned a non-finite p-value: {p_value} for x={x_name}, y={y_name}, z={cond_names}"
        )

    return None, None, p_value

    

def ci_test(data_unused, x, y, z,ci_method="kcit"):
            kci_obj = CIT(data_unused, ci_method)
            # data_unused: 来自 find_skeleton 的 data 参数，这里直接忽略，用外面的 data
            p_value = float(kci_obj(x, y, z))  # 只传索引
            # find_skeleton 只关心最后一个是 p_value，前两个可以随便占位
            return None, None, p_value

def bootstrap_pc_edge_freq(data, n_boot=20, subsample_frac=0.8,
                           alpha=0.1, max_cond_set_size=2,
                           ci_method='RCOT', threshold=0.6,
                           sep_freq_threshold=0.35):
    """
    Bootstrap PC per your rules:
      - only no_edge_freq used
      - pairs with no_edge_freq < threshold are excluded (confidence 0)
      - pairs with no_edge_freq >= threshold: collect sep element counts and empty counts:
          - if empty_cnt >= max_elem_cnt -> choose empty set
          - else choose elements with freq >= sep_freq_threshold
    Returns: res, colliders, confidence (same as get_collide_structure)
    """
    # gaussianize input (your function)
    # data = gaussianize_rank_transform_pandas(data)
    n, d = data.shape
    ci_method_func = None

    if ci_method == "RCOT":
        ci_method_func = RCOT_Test
    elif ci_method == "kci":
        ci_method_func = partial(ci_test, ci_method="kci")
    elif ci_method == "fastkci":
        ci_method_func = partial(ci_test, ci_method="fastkci")
    elif ci_method == "rcit":
        ci_method_func = partial(ci_test, ci_method="rcit")
    else:
        print("使用其他方法",ci_method)
        ci_method_func = ci_method
    if ci_method == "fisherz":
        data = gaussianize_rank_transform_pandas(data)
    # accumulate no-edge counts in upper triangle, then symmetrize
    no_edge_count = np.zeros((d, d), dtype=int)
    total_runs = 0

    # counters for sep elements and empty occurrences
    sep_element_counter = defaultdict(Counter)   # pair -> Counter(elem -> count)
    sep_empty_counter = defaultdict(int)         # pair -> empty count

    mask_upper = np.triu(np.ones((d, d), dtype=bool), k=1)

    for b in range(n_boot):
        # subsample with replacement
        idx = np.random.randint(0, n, size=int(n * subsample_frac))
        Xb = data[idx, :]

        try:
            skel, sep_set = find_skeleton(
                Xb,
                alpha=alpha,
                ci_test=ci_method_func,
                variant='stable'
            )
            print("skel")
            print(skel)
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"find_skeleton failed during bootstrap iteration {b}: {exc}") from exc

        # vectorized increment for no-edge positions (upper triangle)
        no_edge_mask = (np.abs(skel) == 0) & mask_upper
        no_edge_count[no_edge_mask] += 1

        # process sep_set if provided (must iterate; structure is irregular)
        if isinstance(sep_set, dict):
            for key, S in sep_set.items():
                # normalize key
                try:
                    i_raw, j_raw = key
                except Exception:
                    continue
                i0, j0 = int(i_raw), int(j_raw)
                if i0 == j0:
                    continue
                a, b = (i0, j0) if i0 < j0 else (j0, i0)
                pair = (a, b)

                # empty separator
                if S is None or (hasattr(S, "__len__") and len(S) == 0):
                    sep_empty_counter[pair] += 1
                    continue

                # count elements (filter endpoints)
                try:
                    s_set = {int(x) for x in S if int(x) not in pair}
                except Exception:
                    s_set = set()

                if not s_set:
                    sep_empty_counter[pair] += 1
                else:
                    sep_element_counter[pair].update(s_set)

        total_runs += 1
    # symmetrize no_edge_count
    iu, ju = np.triu_indices(d, k=1)
    for i, j in zip(iu, ju):
        no_edge_count[j, i] = no_edge_count[i, j]
    total_runs = max(1, total_runs)
    no_edge_freq = no_edge_count.astype(float) / total_runs

    # prior_knowledge as before: pairs with high no_edge_freq -> 0 (no edge), else 2
    prior_knowledge = np.where(no_edge_freq > threshold, 0, 2)

    # build sep_dict_set following your rule, but only for pairs with no_edge_freq >= threshold
    sep_dict_set = {}
    # enumerate all pairs that appeared OR all possible pairs to be safe
    all_pairs = set(list(sep_element_counter.keys()) + list(sep_empty_counter.keys()))
    for i in range(d):
        for j in range(i+1, d):
            all_pairs.add((i, j))
    print("sep_element_counter",sep_element_counter)
    print("sep_empty_counter",sep_empty_counter)
    for pair in all_pairs:
        i, j = pair
        if no_edge_freq[i, j] < threshold:
            # excluded, treat as empty / not used
            sep_dict_set[pair] = []
            continue

        empty_cnt = sep_empty_counter.get(pair, 0)
        elem_counter = sep_element_counter.get(pair, Counter())
        max_elem_cnt = max(elem_counter.values()) if elem_counter else 0

        if empty_cnt >= max_elem_cnt:
            sep_dict_set[pair] = []
            continue

        selected = [elem for elem, cnt in elem_counter.items() if (cnt / total_runs) >= sep_freq_threshold]
        sep_dict_set[pair] = sorted(selected)
    print("sep_dict_set",sep_dict_set)
    print("no_edge_freq",no_edge_freq)
    # pass no_edge_freq matrix as p_value_tracker (so get_collide_structure can index it directly)
    res, colliders, confidence = get_collide_structure(prior_knowledge, sep_dict_set, p_value_tracker=no_edge_freq)
    return res, colliders, confidence

def get_skeletion_d_sep(
    data,
    /,
    alpha=0.05,
    ci_method="kci",
    variant="stable",
    priori_knowledge=None,
    **kwargs,
):
    p_value_tracker = defaultdict(float)
    assert isinstance(data, np.ndarray), "Input data must be a numpy array."

    def _record_p_value(pair, p_value):
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        if p_value > p_value_tracker[key]:
            p_value_tracker[key] = float(p_value)

    ci_method_func = None

    if ci_method == "kci":
        # 只构造一次 KCI 对象，后面所有 CI 检验都复用它
        kci_obj = CIT(data, ci_method)

        

        ci_method_func = kci_ci_test
    else:
        if callable(ci_method):
            base_ci = ci_method
        elif isinstance(ci_method, str):
            ci_method_lower = ci_method.lower()
            if ci_method_lower == "fisherz":
                base_ci = getattr(CITest, "fisherz_test")
            elif ci_method_lower in {"g2", "gsq"}:
                base_ci = getattr(CITest, "g2_test")
            elif ci_method_lower == "chi2" or ci_method_lower == "chisq":
                base_ci = getattr(CITest, "chi2_test")
            else:
                raise ValueError(f"Unsupported ci_method: {ci_method}")
        else:
            raise TypeError("ci_method must be 'kci', a callable, or one of fisherz/g2/chi2")

        def wrapped_ci_test(data_in, x, y, z):
            result = base_ci(data_in, x, y, z)
            if isinstance(result, tuple):
                p_value = float(result[-1])
            else:
                p_value = float(result)
                result = (None, None, p_value)
            _record_p_value((x, y), p_value)
            return result

        ci_method_func = wrapped_ci_test

    skeleton, sep_set = find_skeleton(
        data,
        alpha=alpha,
        ci_test=ci_method_func,
        variant=variant,
        priori_knowledge=priori_knowledge,
        **kwargs,
    )
    return skeleton, sep_set, dict(p_value_tracker)


def get_collide_structure(skeleton, sep_set, p_value_tracker=None):
    """
    skeleton: np.ndarray (d,d), 0 = no-edge, non-zero = edge (we expect 2 for undirected in your convention)
    sep_set: dict mapping (i,j) -> list_of_nodes (d-sep), keys should be pairs (order may be either i<j or not)
    p_value_tracker: either
        - None or dict mapping (i,j)->value (symmetrically stored or not), or
        - numpy array shape (d,d) giving values directly.
    Returns: (res, colliders, confidence)
    - res: int matrix with 0=no edge, 1=directed endpoint, 2=undirected edge
    - colliders: list of (i,k,j)
    - confidence: float matrix same shape as skeleton with assigned confidences (0 if none)
    """
    d = skeleton.shape[0]
    print("骨架")
    print(skeleton)
    print()

    # 初始化 res：非零位置视为有边（2）
    res = np.where(skeleton != 0, 2, 0).astype(int)
    # print("初始矩阵")
    # print(res)
    np.fill_diagonal(res, 0)
    # print("d分离集")
    # print(sep_set)

    confidence = np.zeros_like(skeleton, dtype=float)
    # p_value_matrix kept for potential debugging / return if wanted
    p_value_matrix = np.zeros_like(skeleton, dtype=float)

    cpdag = deepcopy(np.abs(skeleton))
    colliders = []

    # helper to read p_value_tracker robustly (dict or ndarray)
    def _get_pval(i, j):
        if p_value_tracker is None:
            return 0.0
        # numpy-like
        if hasattr(p_value_tracker, "shape"):
            try:
                return float(p_value_tracker[i, j])
            except Exception:
                return 0.0
        # assume dict-like
        return float(p_value_tracker.get((i, j), p_value_tracker.get((j, i), 0.0)))

    # 主体：遍历 sep_set 中每个 pair
    for key, S_ij in sep_set.items():
        # normalize key to ints
        try:
            i, j = int(key[0]), int(key[1])
        except Exception:
            # skip malformed key
            continue
        if S_ij is None:
            S_ij = []
        for k in range(d):
            if k == i or k == j:
                continue
            if (cpdag[i, k] + cpdag[k, i] != 0 and
                cpdag[k, j] + cpdag[j, k] != 0):

                if k in S_ij:
                    continue

                pval_ij = _get_pval(i, j)
                # 若 i-k 双向存在，则把 k->i 指向 i (即 i is endpoint of edge i<-k)
                if cpdag[i, k] + cpdag[k, i] == 2:
                    cpdag[k, i] = 0
                    res[i, k] = 1
                    res[k, i] = 0
                    confidence[i, k] = pval_ij
                    confidence[k, i] = pval_ij

                if cpdag[j, k] + cpdag[k, j] == 2:
                    cpdag[k, j] = 0
                    res[j, k] = 1
                    res[k, j] = 0
                    confidence[j, k] = pval_ij
                    confidence[k, j] = pval_ij

                triple = (i, k, j)
                if triple not in colliders:
                    colliders.append(triple)

    # print("cp_dag")
    # print(cpdag)
    # print("最终res矩阵（0=无边, 1=有向边终点, 2=未定向边）")
    # print(res)
    # print(f"未定向边数量（值为2的元素）: {np.sum(res == 2) // 2}")
    # print(f"有向边数量: {np.sum((res == 1) & (res.T == 0))}")
    return res, colliders, confidence


def get_collide_constraint( data,
    /,
    alpha=0.05,
    ci_method="kci",
    variant="stable",
    priori_knowledge=None,
    **kwargs
    ):
    skeleton, d_sep, p_tracker = get_skeletion_d_sep(
        data,
        alpha=alpha,
        ci_method=ci_method,
        variant=variant,
        priori_knowledge=priori_knowledge,
        **kwargs,
    )
    return get_collide_structure(
        skeleton=skeleton,
        sep_set=d_sep,
        p_value_tracker=p_tracker,
    )
