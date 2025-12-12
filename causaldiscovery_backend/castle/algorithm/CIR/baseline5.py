# coding: utf-8

import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------- gCastle / Castle ----------
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG
from castle.algorithms import NotearsNonlinear
from algorithm.utils.constraints import ActiveConstraints,InactiveConstraints,OrientationConstraints,MonoConstraints

# ---------- torch.func ----------
try:
    from torch.func import vmap, jacrev
except Exception as e:
    raise ImportError("需要 PyTorch>=2.0 以使用 torch.func.vmap/jacrev") from e





# ============================================================
# 2) 数据生成：带 正/负/无明显方向 的加权 DAG
# ============================================================
def build_signed_dag(n_nodes, n_edges,
                     frac_pos=0.4, frac_neg=0.4, frac_neu=0.2,
                     w_range_pos=(0.5, 2.0),
                     w_range_neg=(-2.0, -0.5),
                     w_range_neu=(-0.05, 0.05),
                     seed=None):
    """
    返回:
      W_true: [d,d] 加权邻接（i->j 的权重，可正可负，可近0）
      A_true: [d,d] 结构0/1
      S_true: [d,d] 符号(+1/-1/0)
    “无明显方向”用接近0的权重近似（也可直接置0，看你需求）
    """
    assert abs(frac_pos + frac_neg + frac_neu - 1.0) < 1e-8
    if seed is not None:
        np.random.seed(seed)

    d = n_nodes
    A_struct = DAG.erdos_renyi(n_nodes=d, n_edges=n_edges,
                               weight_range=(1., 1.), seed=seed)
    edges = np.argwhere(A_struct != 0)  # (i,j)

    W_true = np.zeros((d, d), dtype=float)
    S_true = np.zeros((d, d), dtype=int)

    cats = np.random.choice([+1, -1, 0], size=len(edges),
                            p=[frac_pos, frac_neg, frac_neu])
    for k, (i, j) in enumerate(edges):
        c = cats[k]
        if c > 0:
            w = np.random.uniform(*w_range_pos)
            W_true[i, j] = w
            S_true[i, j] = +1
        elif c < 0:
            w = np.random.uniform(*w_range_neg)
            W_true[i, j] = w
            S_true[i, j] = -1
        else:
            w = np.random.uniform(*w_range_neu)  # 近0
            W_true[i, j] = w
            S_true[i, j] = 0

    A_true = (np.abs(W_true) > 0).astype(int)
    return W_true, A_true, S_true


# ============================================================
# 3) 从真值抽取“部分正确方向/符号”作为候选先验
# ============================================================
def make_priors_from_truth(W_true, A_true, S_true,
                           keep_rate_active=0.4,
                           keep_rate_orient=0.4,
                           keep_rate_mono=0.4,
                           allow_noise=False, noise_rate=0.1,
                           rng=None):
    """
    返回：
      candidate_active: [d,d] 0/1
      candidate_orient: [d,d] 0/1
      w_mono_adj      : [d,d] 邻接型符号先验（行=来源i，列=指向j）：+1/-1/0
    """
    d = W_true.shape[0]
    rnd = np.random.RandomState(None) if rng is None else rng

    idx_true = np.argwhere(A_true == 1)
    m = len(idx_true)

    sel_active = rnd.rand(m) < keep_rate_active
    sel_orient = rnd.rand(m) < keep_rate_orient
    sel_mono   = rnd.rand(m) < keep_rate_mono

    candidate_active = np.zeros((d, d), dtype=int)
    candidate_orient = np.zeros((d, d), dtype=int)
    w_mono_adj = np.zeros((d, d), dtype=float)

    for k, (i, j) in enumerate(idx_true):
        if sel_active[k]:
            candidate_active[i, j] = 1
        if sel_orient[k]:
            candidate_orient[i, j] = 1
        if sel_mono[k]:
            w_mono_adj[i, j] = np.sign(W_true[i, j])  # +1/-1/0

    if allow_noise and keep_rate_mono > 0:
        idx_mono = np.argwhere(w_mono_adj != 0)
        if len(idx_mono) > 0:
            flip_mask = rnd.rand(len(idx_mono)) < noise_rate
            for t, (i, j) in enumerate(idx_mono):
                if flip_mask[t]:
                    w_mono_adj[i, j] = -w_mono_adj[i, j]

    return candidate_active, candidate_orient, w_mono_adj


# ============================================================
# 4) 符号评估（只在双方都认为有边的位置上比较符号一致性）
# ============================================================
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


# ============================================================
# 5) 主实验
# ============================================================
def main():
    # ==== 配置 ====
    method   = 'nonlinear'
    sem_type = 'gp'
    out_dir  = "algorithm/CIR/exp/compare_signed"
    os.makedirs(out_dir, exist_ok=True)

    # NOTEARS 的 config（示例把 mono 打开；active/orient 你可按需也打开）
    config = {
        "active":  {"use": False, "method": "max", "threshold": 0.6, "lamb": 0.07, "name": "active"},
        "inactive":{"use": False, "lamb": 0.01, "name": "inactive"},
        "plus_minus":{"use": False, "lamb": 0.01, "name": "plus_minus"},
        "orient":  {"use": False, "l2_lambda": 0.01, "l1_lambda": 0.1, "name": "orient", "alpha": "max", "use_cumulative": True},
        "mono":    {"use": True,  "l1_lambda": 0.01, "name": "mono", "model": MonoConstraints}
    }

    results = {
        'shd': {'no_prior': [], 'with_prior': []},
        'recall': {'no_prior': [], 'with_prior': []},
        'precision': {'no_prior': [], 'with_prior': []},
        'fdr': {'no_prior': [], 'with_prior': []},
        'tpr': {'no_prior': [], 'with_prior': []},
        'fpr': {'no_prior': [], 'with_prior': []},
        'sign_acc': {'no_prior': [], 'with_prior': []}
    }

    node_list = [30, 35, 40]
    h_list    = [3, 4]  # n_edges = h * n_nodes
    x_labels  = []

    for n_nodes in node_list:
        for h in h_list:
            n_edges = h * n_nodes
            tag = f"n={n_nodes},e={n_edges}"
            x_labels.append(tag)
            print(f"\n配置：{tag}")

            # ---- 生成真值带符号的 DAG ----
            W_true, A_true, S_true = build_signed_dag(
                n_nodes=n_nodes, n_edges=n_edges,
                frac_pos=0.4, frac_neg=0.4, frac_neu=0.2,
                w_range_pos=(0.5, 2.0),
                w_range_neg=(-2.0, -0.5),
                w_range_neu=(-0.05, 0.05),
                seed=n_edges
            )

            # ---- 采样数据 ----
            dataset = IIDSimulation(W=W_true, n=120, method=method, sem_type=sem_type)
            X = dataset.X
            true_dag = A_true

            # ---- 从真值抽部分正确的先验 ----
            candidate_active, candidate_orient, w_mono_adj = make_priors_from_truth(
                W_true, A_true, S_true,
                keep_rate_active=0.4,
                keep_rate_orient=0.4,
                keep_rate_mono=0.4,
                allow_noise=False, noise_rate=0.1
            )

            # ---- 带先验训练（这里演示只用 mono）----
            candidate_dict = {
                "active": candidate_active,  # 若 active/use=True 才会被用到
                "orient": candidate_orient,  # 若 orient/use=True 才会被用到
                "mono":   w_mono_adj         # 邻接型 [d,d]，行=来源、列=指向
            }

            sub_dir = os.path.join(out_dir, f"{n_nodes}_{h}")
            os.makedirs(sub_dir, exist_ok=True)

            # with priors
            al1 = NotearsNonlinear(config=config, candidate_dict=candidate_dict, device_type="gpu")
            al1.learn(X)
            GraphDAG(al1.causal_matrix, true_dag, show=False, save_name=os.path.join(sub_dir, "with_priors.jpg"))
            met1 = MetricsDAG(al1.causal_matrix, true_dag)
            sm1  = sign_metrics(al1.causal_matrix, W_true)
            print("with priors:", met1.metrics, sm1)

            results['shd']['with_prior'].append(met1.metrics['shd'])
            results['recall']['with_prior'].append(met1.metrics['recall'])
            results['precision']['with_prior'].append(met1.metrics['precision'])
            results['fdr']['with_prior'].append(met1.metrics['fdr'])
            results['tpr']['with_prior'].append(met1.metrics['tpr'])
            results['fpr']['with_prior'].append(met1.metrics['fpr'])
            results['sign_acc']['with_prior'].append(sm1['sign_acc'])

            # no priors
            al0 = NotearsNonlinear(device_type="gpu")
            al0.learn(X)
            GraphDAG(al0.causal_matrix, true_dag, show=False, save_name=os.path.join(sub_dir, "no_prior.jpg"))
            met0 = MetricsDAG(al0.causal_matrix, true_dag)
            sm0  = sign_metrics(al0.causal_matrix, W_true)
            print("no priors :", met0.metrics, sm0)

            results['shd']['no_prior'].append(met0.metrics['shd'])
            results['recall']['no_prior'].append(met0.metrics['recall'])
            results['precision']['no_prior'].append(met0.metrics['precision'])
            results['fdr']['no_prior'].append(met0.metrics['fdr'])
            results['tpr']['no_prior'].append(met0.metrics['tpr'])
            results['fpr']['no_prior'].append(met0.metrics['fpr'])
            results['sign_acc']['no_prior'].append(sm0['sign_acc'])

    # ---- 画图 ----
    metrics = ['shd', 'recall', 'precision', 'fdr', 'tpr', 'fpr', 'sign_acc']
    metric_names = {
        'shd': 'SHD', 'recall': '召回率', 'precision': '精确率',
        'fdr': 'FDR', 'tpr': 'TPR', 'fpr': 'FPR', 'sign_acc': '符号准确率'
    }
    method_name_map = {'no_prior': '无先验', 'with_prior': '含先验(示例: mono)'}

    plt.figure(figsize=(22, 24))
    plt.style.use('ggplot')

    for i, metric in enumerate(metrics):
        plt.subplot(4, 2, i+1)
        for j, key in enumerate(['no_prior', 'with_prior']):
            ys = results[metric][key]
            if len(ys) != len(x_labels):  # 容错
                continue
            label = method_name_map[key]
            plt.plot(range(len(x_labels)), ys, marker='o', linewidth=2, markersize=8, label=label)
        plt.title(f'{metric_names[metric]} 指标对比', fontsize=18, fontweight='bold')
        plt.xlabel('配置 (节点数,边数)', fontsize=14)
        plt.ylabel(metric_names[metric], fontsize=14)
        plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best', frameon=True, facecolor='white', edgecolor='gray', framealpha=0.8)

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(out_dir, "causal_discovery_metrics_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=180)
    plt.show()
    print(f"分析完成，结果图已保存：{save_path}")


if __name__ == "__main__":
    # 复现实验可设定全局种子
    np.random.seed(123)
    torch.manual_seed(123)
    main()
