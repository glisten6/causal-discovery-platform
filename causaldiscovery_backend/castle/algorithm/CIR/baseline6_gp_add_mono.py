
# -*- coding: utf-8 -*-
"""
gp-mono-add 数据生成 + NOTEARS 实验（含符号先验）
- 生成带符号的 DAG（行=来源 i，列=指向 j，W_sign∈{-1,0,+1}）
- 用“单调 GP 可加模型”采样 X：x_j = sum_{i∈Pa(j)} s_ij * g_ij(x_i) + ε_j，且 g_ij 单调↑
- NOTEARS-MLP 学习：无先验 vs 含 MonoConstraints（雅可比符号软约束）
- 评估：SHD/TPR/FPR/Precision/Recall + sign_acc（仅在真边∩估计边上比较符号）
"""
import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.integrate import cumulative_trapezoid

import torch
import torch.nn.functional as F
from torch.func import vmap, jacrev
# ---------- 中文字体 ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
# gcastle / castle
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG
from castle.algorithms import NotearsNonlinear
from algorithm.utils.constraints import ActiveConstraints,InactiveConstraints,OrientationConstraints,MonoConstraints

# =========================
# 1) 生成带符号的 DAG + 非负参数（仅供留档；本实验只用 W_sign）
# =========================
def build_signed_dag(
    d: int = 40,
    n_edges: int | None = None,
    frac_pos: float = 0.6,
    frac_neg: float = 0.4,
    frac_neu: float = 0.0,
    seed: int | None = 2025,
):
    """
    返回：
      A:       (d,d) 0/1 结构（DAG，行=来源、列=指向）
      W_sign:  (d,d) ∈{-1,0,+1}，决定边的正/负/无
      order:   拓扑序（生成数据时使用）
    """
    assert abs(frac_pos + frac_neg + frac_neu - 1.0) < 1e-8
    rng = np.random.default_rng(seed)
    if n_edges is None:
        n_edges = int(3 * d)

    # 上三角随机，再随机置换（确保 DAG）
    p = (2 * n_edges) / (d * (d - 1))
    upper = np.triu((rng.random((d, d)) < p).astype(int), k=1)
    P = np.eye(d)[rng.permutation(d)]
    A = (P.T @ upper @ P).astype(int)

    # 符号分配
    edges = np.argwhere(A == 1)
    m = len(edges)
    cats = rng.choice([+1, -1, 0], size=m, p=[frac_pos, frac_neg, frac_neu])
    W_sign = np.zeros((d, d), dtype=int)
    for k, (i, j) in enumerate(edges):
        W_sign[i, j] = int(cats[k])

    # 拓扑序
    indeg = A.sum(0)
    order, A_tmp = [], A.copy().astype(int)
    selectable = np.where(indeg == 0)[0].tolist()
    while selectable:
        v = selectable.pop(0)
        order.append(v)
        children = np.where(A_tmp[v, :] == 1)[0]
        A_tmp[v, children] = 0
        indeg[children] -= 1
        for c in children:
            if indeg[c] == 0:
                selectable.append(c)
    if len(order) != d:
        order = list(range(d))

    return A, W_sign, order


# =========================
# 2) 单调 GP 可加：g'(x)=softplus(h(x))，g(x)=∫g'(t)dt
# =========================
def _monotone_gp_1d(x_query, rng, kernel=None, x_min=-3.0, x_max=3.0, n_grid=512):
    if kernel is None:
        kernel = C(1.0) * RBF(length_scale=1.2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=rng.integers(1<<31))

    x_grid = np.linspace(x_min, x_max, n_grid).reshape(-1, 1)
    h_grid = gp.sample_y(x_grid, random_state=rng.integers(1<<31)).ravel()

    deriv = np.log1p(np.exp(h_grid))           # softplus ≥ 0
    g_grid = cumulative_trapezoid(deriv, x_grid.ravel(), initial=0.0)

    xq = np.clip(np.asarray(x_query), x_min, x_max)
    gq = np.interp(xq, x_grid.ravel(), g_grid) # 线性插值
    return gq


def simulate_gp_add_signed(W_sign, n=1000, noise_scale=0.5, seed=42):
    """
    x_j = sum_{i∈Pa(j)} s_ij * g_ij(x_i) + ε_j
    其中 g_ij 单调递增，∂x_j/∂x_i 的符号恒为 s_ij，s_ij ∈ {-1,0,+1}
    """
    rng = np.random.default_rng(seed)
    d = W_sign.shape[0]
    B = (W_sign != 0).astype(int)
    G = nx.DiGraph(B)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("W_sign 对应结构需为 DAG")

    X = np.zeros((n, d))
    topo = list(nx.topological_sort(G))
    kernel = C(1.0) * RBF(length_scale=1.2)

    edge_rng = {(i, j): np.random.default_rng(rng.integers(1<<31))
                for i, j in zip(*np.where(B == 1))}

    for j in topo:
        pa = list(G.predecessors(j))
        if len(pa) == 0:
            X[:, j] = rng.normal(0.0, noise_scale, size=n)  # 根
        else:
            agg = np.zeros(n)
            for i in pa:
                s_ij = np.sign(W_sign[i, j])  # ±1
                g_ij = _monotone_gp_1d(X[:, i], edge_rng[(i, j)], kernel=kernel)
                agg += s_ij * g_ij
            X[:, j] = agg + rng.normal(0.0, noise_scale, size=n)
    return X



# =========================
# 4) 评估：符号准确率（只在真边∩估计边上比较）
# =========================
def sign_metrics(W_est, W_true):
    est_mask  = (np.abs(W_est)  > 1e-8)
    true_mask = (np.abs(W_true) > 1e-8)
    inter = est_mask & true_mask
    if inter.sum() == 0:
        return {"sign_acc": np.nan}
    sign_est  = np.sign(W_est[inter])
    sign_true = np.sign(W_true[inter])
    acc = (sign_est == sign_true).mean()
    return {"sign_acc": float(acc)}


# =========================
# 5) 主实验
# =========================
def run_experiment():
    out_dir = "algorithm/CIR/exp/gp_mono_add_baseline5"
    os.makedirs(out_dir, exist_ok=True)

    # 实验配置：40 节点，边数 = 3*d
    node_list = [40]
    h_list    = [3,4]  # n_edges = h * n_nodes
    x_labels  = []

    results = {
        'shd': {'no_prior': [], 'with_prior': []},
        'recall': {'no_prior': [], 'with_prior': []},
        'precision': {'no_prior': [], 'with_prior': []},
        'fdr': {'no_prior': [], 'with_prior': []},
        'tpr': {'no_prior': [], 'with_prior': []},
        'fpr': {'no_prior': [], 'with_prior': []},
        'sign_acc': {'no_prior': [], 'with_prior': []}
    }

    # NOTEARS 配置：开启 mono 先验；其余关闭（可自行打开 active/orient）
    config = {
        "active":   {"use": False, "method": "max", "threshold": 0.6, "lamb": 0.07, "name": "active"},
        "inactive": {"use": False, "lamb": 0.01, "name": "inactive"},
        "plus_minus":{"use": False, "lamb": 0.01, "name": "plus_minus"},
        "orient":   {"use": False, "l2_lambda": 0.01, "l1_lambda": 0.1, "name": "orient", "alpha": "max", "use_cumulative": True},
        "mono":     {"use": True,  "l1_lambda": 0.01, "name": "mono", "model": MonoConstraints}
    }

    for n_nodes in node_list:
        for h in h_list:
            n_edges = n_nodes * h
            tag = f"n={n_nodes},e={n_edges}"
            x_labels.append(tag)
            print(f"\n配置：{tag}")

            # 1) 生成真值带符号的 DAG
            A_true, W_sign_true, order = build_signed_dag(
                d=n_nodes, n_edges=n_edges,
                frac_pos=0.6, frac_neg=0.4, frac_neu=0.0, seed=n_edges
            )

            # 2) 用单调 GP 可加模型采样数据
            X = simulate_gp_add_signed(W_sign_true, n=1200, noise_scale=0.6, seed=123)

            # 3) 从真值抽 40% 的符号先验（可加噪声）
            rng = np.random.default_rng(2025)
            w_mono_adj = np.zeros_like(W_sign_true)
            idx_true = np.argwhere(A_true == 1)
            keep_mask = rng.random(len(idx_true)) < 0.4
            for flag, (i, j) in zip(keep_mask, idx_true):
                if flag:
                    w_mono_adj[i, j] = np.sign(W_sign_true[i, j])  # ±1

            # 4) 含先验的 NOTEARS
            sub_dir = os.path.join(out_dir, f"{n_nodes}_{h}")
            os.makedirs(sub_dir, exist_ok=True)

            candidate_dict = {
                "active": (A_true * 0),   # 这里示例不启用，可传部分结构先验
                "orient": (A_true * 0),   # 方向先验同理
                "mono":   w_mono_adj      # 关键：符号先验（二维）
            }

            al1 = NotearsNonlinear(config=config, candidate_dict=candidate_dict, device_type="gpu")
            al1.learn(X)
            GraphDAG(al1.causal_matrix, A_true, show=False, save_name=os.path.join(sub_dir, "with_priors.jpg"))
            met1 = MetricsDAG(al1.causal_matrix, A_true)
            sm1  = sign_metrics(al1.causal_matrix, W_sign_true)
            print("with priors:", met1.metrics, sm1)

            results['shd']['with_prior'].append(met1.metrics['shd'])
            results['recall']['with_prior'].append(met1.metrics['recall'])
            results['precision']['with_prior'].append(met1.metrics['precision'])
            results['fdr']['with_prior'].append(met1.metrics['fdr'])
            results['tpr']['with_prior'].append(met1.metrics['tpr'])
            results['fpr']['with_prior'].append(met1.metrics['fpr'])
            results['sign_acc']['with_prior'].append(sm1['sign_acc'])

            # 5) 无先验的 NOTEARS
            al0 = NotearsNonlinear(device_type="gpu")
            al0.learn(X)
            GraphDAG(al0.causal_matrix, A_true, show=False, save_name=os.path.join(sub_dir, "no_prior.jpg"))
            met0 = MetricsDAG(al0.causal_matrix, A_true)
            sm0  = sign_metrics(al0.causal_matrix, W_sign_true)
            print("no priors :", met0.metrics, sm0)

            results['shd']['no_prior'].append(met0.metrics['shd'])
            results['recall']['no_prior'].append(met0.metrics['recall'])
            results['precision']['no_prior'].append(met0.metrics['precision'])
            results['fdr']['no_prior'].append(met0.metrics['fdr'])
            results['tpr']['no_prior'].append(met0.metrics['tpr'])
            results['fpr']['no_prior'].append(met0.metrics['fpr'])
            results['sign_acc']['no_prior'].append(sm0['sign_acc'])

    # 6) 作图
    metrics = ['shd', 'recall', 'precision', 'fdr', 'tpr', 'fpr', 'sign_acc']
    metric_names = {
        'shd': 'SHD', 'recall': '召回率', 'precision': '精确率',
        'fdr': 'FDR', 'tpr': 'TPR', 'fpr': 'FPR', 'sign_acc': '符号准确率'
    }
    method_name_map = {'no_prior': '无先验', 'with_prior': '含符号先验(mono)'}
    plt.figure(figsize=(20, 22))
    plt.style.use('ggplot')

    for i, metric in enumerate(metrics):
        plt.subplot(4, 2, i+1)
        for key in ['no_prior', 'with_prior']:
            ys = results[metric][key]
            if len(ys) != len(x_labels): continue
            plt.plot(range(len(x_labels)), ys, marker='o', linewidth=2, markersize=8,
                     label=method_name_map[key])
        plt.title(metric_names[metric], fontsize=18, fontweight='bold')
        plt.xlabel('配置 (节点数, 边数)', fontsize=14)
        plt.ylabel(metric_names[metric], fontsize=14)
        plt.xticks(range(len(x_labels)), x_labels, rotation=0, ha='center', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(out_dir, "metrics_gp_mono_add.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=180)
    plt.show()
    print(f"完成。图已保存：{save_path}")


if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(123)
    run_experiment()
