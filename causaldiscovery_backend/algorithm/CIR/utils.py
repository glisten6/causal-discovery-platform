
import numpy as np

import torch 
def get_candidate(true_w,
                  edge_frac=0.4,
                  only_positive=True,
                  seed=2025,
                  false_p=0.2,
                  use_transitive_closure=False,
                  alpha=0.5):

    if isinstance(true_w, np.matrix):
        true_w = np.asarray(true_w)

    d = true_w.shape[0]

    # 可选：传递闭包
    if use_transitive_closure:
        m = d
        # 如果只关心可达性，建议先二值化（可选）
        B = (true_w != 0).astype(float)
        I = np.eye(d)
        M = I + alpha * B

        def matrix_power_fast(M, m: int):
            result = np.eye(d)
            base = M
            exp = m
            while exp > 0:
                if exp & 1:
                    result = result @ base
                base = base @ base
                exp >>= 1
            return result

        C = matrix_power_fast(M, m)   # C = (I + alpha B)^m
        closure_soft = C - I          # 去掉 0 阶项
        true_w = (closure_soft > 0).astype(int)

    assert true_w.shape == (d, d)

    # 1) 真实边
    mask = (true_w > 0) if only_positive else (true_w != 0)
    rng = np.random.default_rng(seed)

    edge_idx = np.array(mask.nonzero()).T   # [num_true_edges, 2]
    has_edges_num = edge_idx.shape[0]

    if has_edges_num == 0:
        # 看你需求，可以选择 raise 或者直接返回全 0
        return np.zeros_like(true_w)

    candidate_num = int(has_edges_num * edge_frac)
    candidate_num = max(candidate_num, 1)   # 至少保留 1 条真实边

    edge_index_perm = rng.permutation(has_edges_num)
    edge_index_candidate = edge_index_perm[:candidate_num]
    edge_candidate = edge_idx[edge_index_candidate]

    w_candidate = np.zeros_like(true_w)
    w_candidate[edge_candidate[:, 0], edge_candidate[:, 1]] = 1

    # 2) 虚假边
    zero_mask = (true_w == 0)
    false_edges = zero_mask & zero_mask.T  # 无向意义下为 0 的对

    i, j = np.triu_indices_from(true_w, 1)  # 上三角无向 pair
    false_edges_candidate = np.column_stack([i, j])[false_edges[i, j]]
    total_false = false_edges_candidate.shape[0]

    false_n = int(candidate_num * false_p)
    if total_false == 0 or false_n == 0:
        # 没有可注入的虚假边或不需要注入
        return w_candidate

    # 最多选 total_false 个虚假边
    false_n = min(false_n, total_false)

    false_edges_candidate_perm = rng.permutation(total_false)
    false_edges_candidate = false_edges_candidate[false_edges_candidate_perm[:false_n]]

    def rand_bool_vector(n, p=0.5, seed=2025):
        _rng = np.random.default_rng(seed=seed)
        return _rng.random(n) < p

    tmp = rand_bool_vector(false_n, seed=seed)
    false_edges_candidate = np.where(
        tmp[..., None],
        false_edges_candidate,
        false_edges_candidate[:, ::-1]
    )

    w_candidate[false_edges_candidate[:, 0], false_edges_candidate[:, 1]] = 1
    return w_candidate

    
    
def rand_40pct_zero(n, low=-1, high=1, seed=None):
    """
    返回 n×n 矩阵：
    - 对角线恒为 0
    - 其余位置约 40 % 为 0，60 % 为 [low, high) 均匀随机数
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(low, high, size=(n, n))
    mask = rng.random((n, n)) < 0.4          # 40 % 位置置零
    A[mask] = 0
    np.fill_diagonal(A, 0)                   # 强制对角线为零
    return A


if __name__ == "__main__":
    d = 10
    true_w = rand_40pct_zero(d,seed = 2025)
    w_candidate = get_candidate(true_w)



