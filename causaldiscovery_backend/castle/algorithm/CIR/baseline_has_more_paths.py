import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
import pandas as pd
import seaborn as sns

from algorithm.utils.constraints import (
    ActiveConstraints,
    InactiveConstraints,
    OrientationConstraints,
    MonoConstraints,
)

from algorithm.utils.checkout import (
    get_paths,
    is_acyclic,
)

# 从本地 gcastle 库导入
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG

from torch.optim import lr_scheduler
from torch.autograd import Variable
from algorithm.CIR.utils.data_process import get_candidate
from collections import defaultdict


# =========================
# 字体相关：保证中文正常显示
# =========================
def _ensure_cjk_fonts():
    """动态注册中文字体，避免退回到 Arial 导致中文方块。"""
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",      # Microsoft YaHei
        r"C:\\Windows\\Fonts\\msyhbd.ttc",    # Microsoft YaHei Bold
        r"C:\\Windows\\Fonts\\simhei.ttf",    # SimHei
        r"C:\\Windows\\Fonts\\simsun.ttc",    # SimSun
        r"C:\\Windows\\Fonts\\NSimSun.ttf",   # NSimSun
    ]
    added = False
    for p in candidates:
        try:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                added = True
        except Exception:
            pass
    if added:
        try:
            fm._rebuild()
        except Exception:
            pass


def _apply_cjk_font():
    """优先选择可用的中文字体。"""
    preferred = [
        'Microsoft YaHei',
        'Microsoft YaHei UI',
        'SimHei',
        'SimSun',
        'NSimSun',
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei',
        'Source Han Sans CN',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams['font.family'] = [name]
            matplotlib.rcParams['font.sans-serif'] = [name]
            return name
    # 若都不可用，至少用 DejaVu Sans（不含 CJK，但避免回退到 Arial）
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


_ensure_cjk_fonts()
_apply_cjk_font()
plt.rcParams['axes.unicode_minus'] = False

# 再次确保项目根目录在路径中（以防从别处调用时出错）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def _to_numpy_matrix(W):
    """将 torch.Tensor 或其他类型统一转换成 numpy.ndarray。"""
    if isinstance(W, np.ndarray):
        return W
    if torch is not None and isinstance(W, torch.Tensor):
        return W.detach().cpu().numpy()
    return np.asarray(W)


def _analyze_structure(W):
    """返回图结构的多项属性信息。

    返回值依次为：
    1. 是否无环
    2. 是否存在任意节点对之间的多条简单路径
    3. 满足多路径条件的节点对列表（包含路径条数）
    4. 是否存在直接路径
    5. 直接路径节点对列表（包含边权）
    6. 既存在直接边，又存在至少包含两个中间节点的其他路径的节点对详情
    """
    W_np = _to_numpy_matrix(W)
    acyclic = bool(is_acyclic(W_np))
    d = W_np.shape[0]
    multi_pairs = []
    direct_pairs = []
    direct_multi_pairs = []
    for src in range(d):
        for dst in range(d):
            if src == dst:
                continue
            paths = get_paths(src, dst, W_np)
            path_count = len(paths)
            if path_count > 1:
                multi_pairs.append({
                    "src": int(src),
                    "dst": int(dst),
                    "path_count": int(path_count),
                })
            if W_np[src, dst] != 0:
                direct_pairs.append({
                    "src": int(src),
                    "dst": int(dst),
                    "weight": float(W_np[src, dst]),
                })

                long_paths = []
                for path in paths:
                    path_nodes = [int(node) for node in path]
                    intermediate_nodes = path_nodes[1:-1]
                    if len(intermediate_nodes) > 1:
                        long_paths.append({
                            "full_path": path_nodes,
                            "intermediate_nodes": intermediate_nodes,
                        })
                if long_paths:
                    direct_multi_pairs.append({
                        "src": int(src),
                        "dst": int(dst),
                        "direct_weight": float(W_np[src, dst]),
                        "long_path_count": len(long_paths),
                        "paths": long_paths,
                    })
    has_multi = len(multi_pairs) > 0
    has_direct = len(direct_pairs) > 0
    return acyclic, has_multi, multi_pairs, has_direct, direct_pairs, direct_multi_pairs


if __name__ == "__main__":
    method = 'nonlinear'
    sem_type = 'gp-add'
    m = 4
    out_dir = f"algorithm/CIR/exp/active_constraint/ex4/{m}"
    os.makedirs(out_dir, exist_ok=True)
    alpha = 0.8  # 先验构造 / 约束里会用到的 alpha

    node_list = [15, 30]
    h_list = [2]  # n_edges = h * n_nodes
    x_labels = []
    seeds = [3407, 7331, 104729, 8675309]
    evaluations_name = ["shd", "recall", "precision", "fdr", "fpr"]
    
    # 运行控制：是否执行实验、是否从 CSV 作图
    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True  # 若只想从已有 CSV 画图，设为 True，且 RUN_EXPERIMENTS=False

    # fdr: (reverse + FP) / (TP + FP)
    # tpr: TP/(TP + FN)
    # fpr: (reverse + FP) / (TN + FP)

    config = {
        "active": {
            "use": True,
            "method": "max",
            "threshold": 0.6,
            "lamb": 0.1,
            "name": "active",
            "model": ActiveConstraints,
            "use_transitive_closure": False,  # 默认不用传递闭包
            "alpha": alpha,                  # 这里保存 alpha，方便到处统一读取
        },
        "inactive": {
            "use": False,
            "lamb": 0.01,
            "model": InactiveConstraints,
            "name": "inactive",
        },
        "plus_minus": {
            "use": False,
            "lamb": 0.01,
            "name": "plus_minus",
        },
        "orient": {
            "use": False,
            "l2_lambda": 0.01,
            "l1_lambda": 0.1,
            "model": OrientationConstraints,
            "name": "orient",
            "alpha": "max",
            "use_cumulative": True,
        },
        "mono": {
            "use": False,
            "l1_lambda": 0.01,
            "model": MonoConstraints,
            "name": "mono",
        },
    }

    # 收集每个 seed 的原始指标行
    results_rows = []
    true_ps = [0.4, 0.6]

    # ==============================
    # 一、实验部分：生成数据 + 训练
    # ==============================
    if RUN_EXPERIMENTS:
        for n in [300, 1200]:
            n_out_dir = os.path.join(out_dir, f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            for n_nodes in node_list:
                for h in h_list:
                    for true_p in true_ps:
                        for false_p in [0.2, 0.0]:
                            for seed in seeds:
                                n_edges = h * n_nodes
                                print(
                                    f"\n测试配置: n={n}, 节点数量={n_nodes}, edge={n_nodes * h}, "
                                    f"true_p={true_p}, false_p={false_p}, seed={seed}"
                                )

                                x_labels.append(f"n={n_nodes},e={n_nodes * h}")

                                # ---- 生成真值带权 DAG ----
                                weighted_random_dag = DAG.erdos_renyi(
                                    n_nodes=n_nodes,
                                    n_edges=n_edges,
                                    weight_range=(0.5, 2.0),
                                    seed=seed,
                                )
                                dataset = IIDSimulation(
                                    W=weighted_random_dag,
                                    n=n,
                                    method=method,
                                    sem_type=sem_type
                                )
                                true_dag, X = dataset.B, dataset.X

                                # ---- 获得先验集合（active 约束）----
                                alpha_for_candidate = config.get("active", {}).get("alpha", alpha)
                                candidate_active = get_candidate(
                                    true_dag,
                                    seed=seed,
                                    false_p=false_p,
                                    edge_frac=true_p,
                                    alpha=alpha_for_candidate,
                                    use_transitive_closure=True,
                                    m=m
                                )
                                candidate_dict = {
                                    "active": candidate_active,
                                }

                                # 创建输出目录
                                base_dir = os.path.join(
                                    n_out_dir,
                                    f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}"
                                )
                                os.makedirs(base_dir, exist_ok=True)

                                # ===========================
                                # 1) 普通约束：邻接矩阵先验
                                # ===========================
                                print("普通约束（邻接矩阵，非传递闭包）")
                                config["active"]["use_transitive_closure"] = False

                                al_adj = NotearsNonlinear(
                                    config=config,
                                    candidate_dict=candidate_dict,
                                    device_type="cpu",
                                )
                                al_adj.learn(X)

                                GraphDAG(
                                    al_adj.causal_matrix,
                                    true_dag,
                                    show=False,
                                    save_name=os.path.join(base_dir, "active_adj.jpg"),
                                )
                                met_adj = MetricsDAG(al_adj.causal_matrix, true_dag)
                                print("普通约束(邻接):", met_adj.metrics)

                                (
                                    adj_is_dag,
                                    adj_has_multi_path,
                                    adj_pairs,
                                    adj_has_direct,
                                    adj_direct_pairs,
                                    adj_direct_multi_pairs,
                                ) = _analyze_structure(al_adj.causal_matrix)
                                adj_has_direct_multi = len(adj_direct_multi_pairs) > 0
                                results_rows.append({
                                    "n": n,
                                    "true_p": true_p,
                                    "false_p": false_p,
                                    "n_nodes": n_nodes,
                                    "h": h,
                                    "seed": seed,
                                    "method": "with_constraint_adj",  # 邻接矩阵约束
                                    "shd": met_adj.metrics["shd"],
                                    "recall": met_adj.metrics["recall"],
                                    "precision": met_adj.metrics["precision"],
                                    "fdr": met_adj.metrics["fdr"],
                                    "fpr": met_adj.metrics["fpr"],
                                    "acyclic": adj_is_dag,
                                    "multi_paths": adj_has_multi_path,
                                    "multi_path_pairs": json.dumps(adj_pairs, ensure_ascii=False),
                                    "has_direct_path": adj_has_direct,
                                    "direct_path_pairs": json.dumps(adj_direct_pairs, ensure_ascii=False),
                                    "has_direct_multi_path": adj_has_direct_multi,
                                    "direct_multi_path_pairs": json.dumps(adj_direct_multi_pairs, ensure_ascii=False),
                                    "adjacency_matrix": al_adj.causal_matrix.detach().cpu().tolist(),
                                })

                                # ===========================
                                # 2) 传递闭包约束
                                # ===========================
                                print("使用传递闭包的有约束")
                                config["active"]["use_transitive_closure"] = True

                                al_tc = NotearsNonlinear(
                                    config=config,
                                    candidate_dict=candidate_dict,
                                    device_type="cpu",
                                )
                                al_tc.learn(X)

                                GraphDAG(
                                    al_tc.causal_matrix,
                                    true_dag,
                                    show=False,
                                    save_name=os.path.join(base_dir, "active_tc.jpg"),
                                )
                                met_tc = MetricsDAG(al_tc.causal_matrix, true_dag)
                                print("传递闭包约束:", met_tc.metrics)

                                (
                                    tc_is_dag,
                                    tc_has_multi_path,
                                    tc_pairs,
                                    tc_has_direct,
                                    tc_direct_pairs,
                                    tc_direct_multi_pairs,
                                ) = _analyze_structure(al_tc.causal_matrix)
                                tc_has_direct_multi = len(tc_direct_multi_pairs) > 0
                                results_rows.append({
                                    "n": n,
                                    "true_p": true_p,
                                    "false_p": false_p,
                                    "n_nodes": n_nodes,
                                    "h": h,
                                    "seed": seed,
                                    "method": "with_constraint_tc",  # 传递闭包约束
                                    "shd": met_tc.metrics["shd"],
                                    "recall": met_tc.metrics["recall"],
                                    "precision": met_tc.metrics["precision"],
                                    "fdr": met_tc.metrics["fdr"],
                                    "fpr": met_tc.metrics["fpr"],
                                    "acyclic": tc_is_dag,
                                    "multi_paths": tc_has_multi_path,
                                    "multi_path_pairs": json.dumps(tc_pairs, ensure_ascii=False),
                                    "has_direct_path": tc_has_direct,
                                    "direct_path_pairs": json.dumps(tc_direct_pairs, ensure_ascii=False),
                                    "has_direct_multi_path": tc_has_direct_multi,
                                    "direct_multi_path_pairs": json.dumps(tc_direct_multi_pairs, ensure_ascii=False),
                                    "adjacency_matrix": al_tc.causal_matrix.detach().cpu().tolist(),
                                })

                                # 结束后复位（可选）
                                config["active"]["use_transitive_closure"] = False

                                # ===========================
                                # 3) 无约束
                                # ===========================
                                print("无约束")
                                al_none = NotearsNonlinear(device_type="cpu")
                                al_none.learn(X)

                                no_dir = os.path.join(
                                    n_out_dir,
                                    f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}_no_constraints"
                                )
                                os.makedirs(no_dir, exist_ok=True)

                                GraphDAG(
                                    al_none.causal_matrix,
                                    true_dag,
                                    show=False,
                                    save_name=os.path.join(no_dir, "no_constraint.jpg"),
                                )
                                met_none = MetricsDAG(al_none.causal_matrix, true_dag)
                                print("无约束:", met_none.metrics)

                                (
                                    none_is_dag,
                                    none_has_multi_path,
                                    none_pairs,
                                    none_has_direct,
                                    none_direct_pairs,
                                    none_direct_multi_pairs,
                                ) = _analyze_structure(al_none.causal_matrix)
                                none_has_direct_multi = len(none_direct_multi_pairs) > 0
                                results_rows.append({
                                    "n": n,
                                    "true_p": true_p,
                                    "false_p": false_p,
                                    "n_nodes": n_nodes,
                                    "h": h,
                                    "seed": seed,
                                    "method": "no_constraint",
                                    "shd": met_none.metrics["shd"],
                                    "recall": met_none.metrics["recall"],
                                    "precision": met_none.metrics["precision"],
                                    "fdr": met_none.metrics["fdr"],
                                    "fpr": met_none.metrics["fpr"],
                                    "acyclic": none_is_dag,
                                    "multi_paths": none_has_multi_path,
                                    "multi_path_pairs": json.dumps(none_pairs, ensure_ascii=False),
                                    "has_direct_path": none_has_direct,
                                    "direct_path_pairs": json.dumps(none_direct_pairs, ensure_ascii=False),
                                    "has_direct_multi_path": none_has_direct_multi,
                                    "direct_multi_path_pairs": json.dumps(none_direct_multi_pairs, ensure_ascii=False),
                                    "adjacency_matrix": al_none.causal_matrix.detach().cpu().tolist(),
                                })

            # 每个 n 保存一次 CSV（也可以统一到最外层再保存）
            df_all_tmp = pd.DataFrame(results_rows)
            df_n = df_all_tmp[df_all_tmp["n"] == n].copy()
            csv_path_n = os.path.join(n_out_dir, "results_by_seed.csv")
            df_n.to_csv(csv_path_n, index=False, encoding="utf-8-sig")

    print("___________________________________________")

    # ========================================
    # 二、画图：自动根据 df 中的 n / false_p 画图
    # ========================================
    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
        print("画图")

        # 1）准备 df_all：要么直接用刚跑完的 results_rows，要么从 CSV 汇总
        if RUN_EXPERIMENTS:
            if not results_rows:
                print("没有任何实验结果，跳过画图")
                sys.exit(0)
            df_all = pd.DataFrame(results_rows)
        else:
            dfs = []
            if os.path.exists(out_dir):
                for name in os.listdir(out_dir):
                    subdir = os.path.join(out_dir, name)
                    if not os.path.isdir(subdir):
                        continue
                    csv_path = os.path.join(subdir, "results_by_seed.csv")
                    if os.path.exists(csv_path):
                        dfs.append(pd.read_csv(csv_path))
            if not dfs:
                print("未找到任何 CSV 结果文件，跳过画图")
                sys.exit(0)
            df_all = pd.concat(dfs, ignore_index=True)

        # ★ 自动识别所有出现过的 n
        n_values = sorted(df_all["n"].unique())
        print("自动识别到的 n 值:", n_values)

        for n in n_values:
            n_int = int(n)
            n_out_dir = os.path.join(out_dir, f"{n_int}")
            os.makedirs(n_out_dir, exist_ok=True)
            csv_path = os.path.join(n_out_dir, "results_by_seed.csv")

            # 当前 n 的子集
            df = df_all[df_all["n"] == n].copy()
            if df.empty:
                continue

            # 覆盖 / 生成当前 n 的 CSV（方便以后只从 CSV 画图）
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            print(f"n={n_int} 的结构检测结果")

            df["acyclic"] = df["acyclic"].astype(bool)
            df["multi_paths"] = df["multi_paths"].astype(bool)
            df["has_cycle"] = ~df["acyclic"]
            df["has_multi_path"] = df["multi_paths"]
            df["has_direct_path"] = df["has_direct_path"].astype(bool)
            if "has_direct_multi_path" in df.columns:
                df["has_direct_multi_path"] = df["has_direct_multi_path"].astype(bool)
            else:
                df["has_direct_multi_path"] = False

            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            group_cols = ["n", "n_nodes", "h", "true_p", "false_p", "method"]
            summary = (
                df.groupby(group_cols, as_index=False)
                .agg(
                    total_runs=("seed", "count"),
                    cycle_count=("has_cycle", "sum"),
                    multi_path_count=("has_multi_path", "sum"),
                    direct_path_count=("has_direct_path", "sum"),
                    direct_multi_path_count=("has_direct_multi_path", "sum"),
                )
            )
            summary["has_cycle"] = summary["cycle_count"] > 0
            summary["has_multi_path"] = summary["multi_path_count"] > 0
            summary["has_direct_path"] = summary["direct_path_count"] > 0
            summary["has_direct_multi_path"] = summary["direct_multi_path_count"] > 0

            summary = summary[
                group_cols
                + [
                    "total_runs",
                    "cycle_count",
                    "has_cycle",
                    "multi_path_count",
                    "has_multi_path",
                    "direct_path_count",
                    "has_direct_path",
                    "direct_multi_path_count",
                    "has_direct_multi_path",
                ]
            ].sort_values(["n_nodes", "h", "true_p", "false_p", "method"])

            excel_path = os.path.join(n_out_dir, "structure_summary.xlsx")
            csv_summary_path = os.path.join(n_out_dir, "structure_summary.csv")
            summary.to_excel(excel_path, index=False)
            summary.to_csv(csv_summary_path, index=False, encoding="utf-8-sig")

            print(summary.to_string(index=False))
