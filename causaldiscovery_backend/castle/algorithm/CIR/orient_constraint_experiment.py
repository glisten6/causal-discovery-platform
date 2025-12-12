import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

# 从本地 gcastle 库导入
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG

from torch.optim import lr_scheduler
from torch.autograd import Variable
from algorithm.CIR.utils.data_process import get_candidate


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
use_pytorch_optimizer = False


def _compute_f1(recall: float, precision: float, eps: float = 1e-12) -> float:
    """根据 recall 与 precision 计算 F1 分数。"""
    denom = recall + precision
    if denom <= eps:
        return 0.0
    return 2.0 * recall * precision / denom


def _plot_metrics_for_n(df_n: pd.DataFrame, n_int: int, out_dir: str,
                        metrics: list[str], method_map: dict[str, str]) -> None:
    if df_n.empty:
        return

    df_n = df_n.copy()
    df_n["method_cn"] = df_n["method"].map(method_map).fillna(df_n["method"])

    n_out_dir = os.path.join(out_dir, f"{n_int}")
    os.makedirs(n_out_dir, exist_ok=True)

    def _format_prob(value) -> str:
        if isinstance(value, (int, float, np.floating)):
            return f"{value:g}"
        return str(value)

    df_n["config"] = (
        "n" + df_n["n_nodes"].astype(int).astype(str)
        + "_h" + df_n["h"].astype(int).astype(str)
        + "_tp" + df_n["true_p"].map(_format_prob)
        + "_fp" + df_n["false_p"].map(_format_prob)
    )

    config_order = (
        df_n[["n_nodes", "h", "true_p", "false_p"]]
        .drop_duplicates()
        .sort_values(["n_nodes", "h", "true_p", "false_p"])
        .apply(
            lambda r: (
                f"n{int(r['n_nodes'])}_h{int(r['h'])}_tp{_format_prob(r['true_p'])}_fp{_format_prob(r['false_p'])}"
            ),
            axis=1,
        )
        .tolist()
    )
    df_n["config"] = pd.Categorical(df_n["config"], categories=config_order, ordered=True)
    df_n = df_n.sort_values("config")

    hue_order = (
        ["方向约束", "无约束"]
        if set(df_n["method_cn"]) >= {"方向约束", "无约束"}
        else sorted(df_n["method_cn"].unique())
    )

    for metric in metrics:
        plt.figure(figsize=(12, 5))
        ax = sns.lineplot(
            data=df_n,
            x="config",
            y=metric,
            hue="method_cn",
            estimator="mean",
            errorbar="sd",
            marker="o",
            sort=False,
            hue_order=hue_order,
        )
        ax.set_xlabel("配置 (n_nodes, h, true_p, false_p)")
        ax.set_ylabel(metric)
        ax.set_title(f"指标折线图（按种子取均值）：{metric} (n={n_int})")

        tick_positions = ax.get_xticks()
        raw_labels = [label.get_text() for label in ax.get_xticklabels()]
        formatted_labels = ["\n".join(lbl.split("_")) if lbl else lbl for lbl in raw_labels]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_positions))
        ax.set_xticklabels(formatted_labels)
        ax.tick_params(axis="x", labelrotation=0)

        plt.tight_layout()

        save_path = os.path.join(n_out_dir, f"n_{n_int}_{metric}_comparison.png")
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close()
        print(f"已保存图表: {save_path}")


if __name__ == "__main__":
    method = 'nonlinear'
    m = 4
    sem_type = 'gp-add'
    out_dir = f"algorithm/CIR/exp/orient_constraint/study/m_{m}"
    os.makedirs(out_dir, exist_ok=True)

    node_list = [15, 30]
    h_list = [2]  # n_edges = h * n_nodes
    x_labels = []
    seeds = [3407, 7331, 104729]
    evaluations_name = ["shd", "recall", "precision", "f1_score", "fdr", "fpr"]
    orient_penalties = [
        (0.4, 0.1),
    ]  # (l1_lambda, l2_lambda)

    # 运行控制：是否执行实验、是否从 CSV 作图
    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True  # 若只想从已有 CSV 画图，设为 True，且 RUN_EXPERIMENTS=False

    config = {
        "active": {
            "use": False,
            "method": "max",
            "threshold": 0.6,
            "lamb": 0.1,
            "name": "active",
            "model": ActiveConstraints,
            "use_transitive_closure": False,
            "alpha": 0.8,
            "beta": 10,
        },
        "inactive": {
            "use": False,
            "lamb": 0.01,
            "model": InactiveConstraints,
            "name": "inactive",
        },
        # "plus_minus": {
        #     "use": False,
        #     "lamb": 0.01,
        #     "name": "plus_minus",
        # },
        "orient": {
            "use": True,
            "l2_lambda": 0.3,
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
    method_map = {
        "orient": "方向约束",
        "no_constraint": "无约束",
    }

    # ==============================
    # 一、实验部分：生成数据 + 训练
    # ==============================
    if RUN_EXPERIMENTS:
        for n in [200, 300]:
            n_out_dir = os.path.join(out_dir, f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            n_rows = []

            for l1_lambda, l2_lambda in orient_penalties:
                l1_display = f"{l1_lambda:.3f}".rstrip('0').rstrip('.')
                if l1_display == "":
                    l1_display = "0"
                l1_tag = l1_display.replace('.', 'p')

                l2_display = f"{l2_lambda:.3f}".rstrip('0').rstrip('.')
                if l2_display == "":
                    l2_display = "0"
                l2_tag = l2_display.replace('.', 'p')

                lambda_out_dir = os.path.join(n_out_dir, f"l1_{l1_tag}_l2_{l2_tag}")
                os.makedirs(lambda_out_dir, exist_ok=True)
                config["orient"]["l1_lambda"] = l1_lambda
                config["orient"]["l2_lambda"] = l2_lambda

                for n_nodes in node_list:
                    for h in h_list:
                        for true_p in true_ps:
                            for false_p in [0.2, 0.0]:
                                for seed in seeds:
                                    n_edges = h * n_nodes
                                    print(
                                        f"\n测试配置: n={n}, 节点数量={n_nodes}, edge={n_nodes * h}, "
                                        f"true_p={true_p}, false_p={false_p}, seed={seed}, "
                                        f"l1={l1_display}, l2={l2_display}"
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

                                    # ---- 获得先验集合（orient 约束）----
                                    candidate_orient = get_candidate(
                                        true_dag,
                                        seed=seed,
                                        false_p=false_p,
                                        edge_frac=true_p,
                                        alpha=1,
                                        use_transitive_closure=False,
                                        m=m
                                    )
                                    candidate_dict = {
                                        "orient": candidate_orient,
                                    }

                                    # 创建输出目录
                                    base_dir = os.path.join(
                                        lambda_out_dir,
                                        f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}"
                                    )
                                    os.makedirs(base_dir, exist_ok=True)

                                    # ===========================
                                    # 1) 方向约束
                                    # ===========================
                                    print("方向约束 (orient)")
                                    orient_model = NotearsNonlinear(
                                        config=config,
                                        candidate_dict=candidate_dict,
                                        use_pytorch_optimizer=use_pytorch_optimizer,
                                        device_type="cpu",
                                    )
                                    orient_model.learn(X)

                                    GraphDAG(
                                        orient_model.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "orient.jpg"),
                                    )
                                    met_orient = MetricsDAG(orient_model.causal_matrix, true_dag)
                                    print("方向约束指标:", met_orient.metrics)

                                    row_orient = {
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "l1_lambda": l1_lambda,
                                        "l2_lambda": l2_lambda,
                                        "method": "orient",
                                        "shd": met_orient.metrics["shd"],
                                        "recall": met_orient.metrics["recall"],
                                        "precision": met_orient.metrics["precision"],
                                        "f1_score": _compute_f1(
                                            met_orient.metrics["recall"],
                                            met_orient.metrics["precision"],
                                        ),
                                        "fdr": met_orient.metrics["fdr"],
                                        "fpr": met_orient.metrics["fpr"],
                                    }
                                    results_rows.append(row_orient)
                                    n_rows.append(row_orient)

                                    # ===========================
                                    # 2) 无约束基线（不传候选）
                                    # ===========================
                                    print("无约束 (baseline)")
                                    baseline_model = NotearsNonlinear(
                                        use_pytorch_optimizer=use_pytorch_optimizer,
                                        device_type="cpu",
                                    )
                                    baseline_model.learn(X)

                                    GraphDAG(
                                        baseline_model.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "no_constraint.jpg"),
                                    )
                                    met_baseline = MetricsDAG(baseline_model.causal_matrix, true_dag)
                                    print("无约束指标:", met_baseline.metrics)

                                    row_baseline = {
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "l1_lambda": np.nan,
                                        "l2_lambda": np.nan,
                                        "method": "no_constraint",
                                        "shd": met_baseline.metrics["shd"],
                                        "recall": met_baseline.metrics["recall"],
                                        "precision": met_baseline.metrics["precision"],
                                        "f1_score": _compute_f1(
                                            met_baseline.metrics["recall"],
                                            met_baseline.metrics["precision"],
                                        ),
                                        "fdr": met_baseline.metrics["fdr"],
                                        "fpr": met_baseline.metrics["fpr"],
                                    }
                                    results_rows.append(row_baseline)
                                    n_rows.append(row_baseline)

            if n_rows:
                df_n = pd.DataFrame(n_rows)
                csv_path_n = os.path.join(n_out_dir, "results_by_seed.csv")
                df_n.to_csv(csv_path_n, index=False, encoding="utf-8-sig")
                print(f"n={n} 的结果已保存到: {csv_path_n}")

                if PLOT_FROM_CSV:
                    _plot_metrics_for_n(df_n, int(n), out_dir, evaluations_name, method_map)

    print("___________________________________________")

    # ========================================
    # 二、画图：自动根据 df 中的 n / false_p 画图
    # ========================================
    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
        print("开始绘图")

        # 如果刚跑完实验，直接使用 results_rows；否则从 CSV 汇总
        if RUN_EXPERIMENTS and results_rows:
            df_all = pd.DataFrame(results_rows)
        else:
            # 读取每个 n 的 CSV
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
                print("未找到任何 results_by_seed.csv，跳过绘图")
                sys.exit(0)

            df_all = pd.concat(dfs, ignore_index=True)

        if df_all.empty:
            print("没有可用数据，跳过绘图")
            sys.exit(0)

        sns.set(style="whitegrid")
        _ensure_cjk_fonts()
        _apply_cjk_font()
        plt.rcParams['axes.unicode_minus'] = False

        for n in sorted(df_all["n"].unique()):
            df_n = df_all[df_all["n"] == n]
            _plot_metrics_for_n(df_n, int(n), out_dir, evaluations_name, method_map)

        print("绘图完成")
