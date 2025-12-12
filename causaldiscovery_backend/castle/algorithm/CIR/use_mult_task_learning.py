import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
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
from RoaDs.RoaDs.PSL.NOTEARS_MLP import NotearsNonlinear_PSL
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
use_pytorch_optimizer = False


def _compute_f1(recall: float, precision: float, eps: float = 1e-12) -> float:
    """根据 recall 与 precision 计算 F1 分数。"""
    denom = recall + precision
    if denom <= eps:
        return 0.0
    return 2.0 * recall * precision / denom

if __name__ == "__main__":
    method = 'nonlinear'
    m = 4
    sem_type = 'gp-add'
    out_dir = rf"D:\data\python_project\causal_discovery\algorithm\CIR\exp\mdlearning_project2\ex1"
    os.makedirs(out_dir, exist_ok=True)
    alpha = 0.8  # 先验构造 / 约束里会用到的 alpha

    node_list = [15, 30]
    h_list = [2]  # n_edges = h * n_nodes
    x_labels = []
    # seeds = [3407, 7331, 104729, 8675309]   
    seeds = [3407, 7331, 104729，8675309]     
    evaluations_name = ["shd", "recall", "precision", "f1_score", "fdr", "fpr"]
    lambda_values = [0.1]  # 可按需调整不同的约束强度

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
            "alpha": alpha, # 这里保存 alpha，方便到处统一读取
            "beta":10                 
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
        for n in [200, 300]:
            n_out_dir = os.path.join(out_dir, f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            for lambda_val in lambda_values:
                lambda_display = f"{lambda_val:.3f}".rstrip('0').rstrip('.')
                lambda_tag = lambda_display.replace('.', 'p')
                lambda_out_dir = os.path.join(n_out_dir, f"lambda_{lambda_tag}")
                os.makedirs(lambda_out_dir, exist_ok=True)
                config["active"]["lamb"] = lambda_val

                for n_nodes in node_list:
                    for h in h_list:
                        for true_p in true_ps:
                            for false_p in [0.4, 0.2]:
                                for seed in seeds:
                                    n_edges = h * n_nodes
                                    print(
                                        f"\n测试配置: n={n}, 节点数量={n_nodes}, edge={n_nodes * h}, "
                                        f"true_p={true_p}, false_p={false_p}, seed={seed}, lambda={lambda_display}"
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
                                        alpha=1,
                                        use_transitive_closure=False,
                                        m=m
                                    )
                                    candidate_dict = {
                                        "active": candidate_active,
                                    }

                                    # 创建输出目录
                                    base_dir = os.path.join(
                                        lambda_out_dir,
                                        f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}"
                                    )
                                    os.makedirs(base_dir, exist_ok=True)

                                    # ===========================
                                    # 1) 普通约束：邻接矩阵先验
                                    # ===========================
                                    print("max_soft_regulation")
                                    config["active"]["use_transitive_closure"] = False
                                    config["active"]["method"] = "max"

                                    al_max = NotearsNonlinear(
                                        config=config,
                                        candidate_dict=candidate_dict,
                                        use_pytorch_optimizer=use_pytorch_optimizer,
                                        device_type="cpu",
                                    )
                                    al_max.learn(X)

                                    GraphDAG(
                                        al_max.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "max_soft_regulation.jpg"),
                                    )
                                    met_max = MetricsDAG(al_max.causal_matrix, true_dag)
                                    print("max_soft_regulation:", met_max.metrics)

                                    results_rows.append({
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "lambda_param": lambda_val,
                                        "method": "max_soft_regulation",
                                        "shd": met_max.metrics["shd"],
                                        "recall": met_max.metrics["recall"],
                                        "precision": met_max.metrics["precision"],
                                        "f1_score": _compute_f1(met_max.metrics["recall"], met_max.metrics["precision"]),
                                        "fdr": met_max.metrics["fdr"],
                                        "fpr": met_max.metrics["fpr"],
                                    })

                                    # ===========================
                                    # 2) NotearsNonlinear_PSL with prior (DSGP)
                                    # ===========================
                                    print("dsgp")
                                    W_m_prior = candidate_active.astype(float)
                                    constraint_prior = W_m_prior.copy()

                                    psl_prior = NotearsNonlinear_PSL(
                                        lambda1=lambda_val,
                                        lambda2=lambda_val,
                                        use_orignal_solver=False,
                                    )
                                    psl_prior.learn(
                                        data=X,
                                        method='RoaDs',
                                        constraint_dag=constraint_prior,
                                        Wm=W_m_prior,
                                        normalization_type='loss+'
                                    )

                                    GraphDAG(
                                        psl_prior.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "dsgp.jpg"),
                                    )
                                    met_psl_prior = MetricsDAG(psl_prior.causal_matrix, true_dag)
                                    print("dsgp:", met_psl_prior.metrics)

                                    results_rows.append({
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "lambda_param": lambda_val,
                                        "method": "dsgp",
                                        "shd": met_psl_prior.metrics["shd"],
                                        "recall": met_psl_prior.metrics["recall"],
                                        "precision": met_psl_prior.metrics["precision"],
                                        "f1_score": _compute_f1(met_psl_prior.metrics["recall"], met_psl_prior.metrics["precision"]),
                                        "fdr": met_psl_prior.metrics["fdr"],
                                        "fpr": met_psl_prior.metrics["fpr"],
                                    })

                                    # ===========================
                                    # 3) NotearsNonlinear_PSL without prior
                                    # ===========================
                                    print("无先验")
                                    zero_prior = np.zeros_like(W_m_prior)

                                    no_prior = NotearsNonlinear(
                                        config=config,
                                        candidate_dict=candidate_dict,
                                        use_pytorch_optimizer=use_pytorch_optimizer,
                                        device_type="cpu",
                                    )
                                    no_prior.learn(X)


                                    GraphDAG(
                                        no_prior.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "no_prior.jpg"),
                                    )
                                    met_psl_no_prior = MetricsDAG(no_prior.causal_matrix, true_dag)
                                    print("无先验:", met_psl_no_prior.metrics)

                                    results_rows.append({
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "lambda_param": lambda_val,
                                        "method": "无先验",
                                        "shd": met_psl_no_prior.metrics["shd"],
                                        "recall": met_psl_no_prior.metrics["recall"],
                                        "precision": met_psl_no_prior.metrics["precision"],
                                        "f1_score": _compute_f1(met_psl_no_prior.metrics["recall"], met_psl_no_prior.metrics["precision"]),
                                        "fdr": met_psl_no_prior.metrics["fdr"],
                                        "fpr": met_psl_no_prior.metrics["fpr"],
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
                print(out_dir)
                print(os.listdir(out_dir))
                for name in os.listdir(out_dir):
                    subdir = os.path.join(out_dir, name)
                    print(subdir)
                    if not os.path.isdir(subdir):
                        continue
                    csv_path = os.path.join(subdir, "results_by_seed.csv")
                    print(csv_path)
                    if os.path.exists(csv_path):
                        dfs.append(pd.read_csv(csv_path))
            if not dfs:
                print("未找到任何 CSV 结果文件，跳过画图")
                sys.exit(0)
            df_all = pd.concat(dfs, ignore_index=True)

        if "f1_score" not in df_all.columns:
            df_all["f1_score"] = df_all.apply(
                lambda row: _compute_f1(row.get("recall", 0.0), row.get("precision", 0.0)),
                axis=1,
            )
        else:
            mask_nan = df_all["f1_score"].isna()
            if mask_nan.any():
                df_all.loc[mask_nan, "f1_score"] = df_all.loc[mask_nan].apply(
                    lambda row: _compute_f1(row.get("recall", 0.0), row.get("precision", 0.0)),
                    axis=1,
                )

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

            print(f"n={n_int} 的对比图")

            if "lambda_param" not in df.columns:
                fallback_lambda = config.get("active", {}).get("lamb", 0.0)
                df["lambda_param"] = float(fallback_lambda)
            else:
                fallback_lambda = config.get("active", {}).get("lamb", 0.0)
                df["lambda_param"] = df["lambda_param"].fillna(float(fallback_lambda))
            df["lambda_param"] = df["lambda_param"].astype(float)
            lambda_values_in_df = sorted(df["lambda_param"].unique())

            sns.set(style="whitegrid")
            _ensure_cjk_fonts()
            _apply_cjk_font()
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_theme()
            _apply_cjk_font()
            plt.rcParams['axes.unicode_minus'] = False

            # 组合横轴标签（不含 fp，用于单方法图）
            def make_cfg_base(n_nodes_val, h_val, true_p_val):
                try:
                    nn = int(n_nodes_val)
                except Exception:
                    nn = n_nodes_val
                try:
                    hh = int(h_val)
                except Exception:
                    hh = h_val
                return f"n{nn}_h{hh}_tp{true_p_val}"

            df["config_base"] = df.apply(
                lambda r: make_cfg_base(r["n_nodes"], r["h"], r["true_p"]), axis=1
            )
            order_base = (
                df[["n_nodes", "h", "true_p"]]
                .drop_duplicates()
                .sort_values(["n_nodes", "h", "true_p"])
                .apply(lambda r: f"n{int(r['n_nodes'])}_h{int(r['h'])}_tp{r['true_p']}", axis=1)
                .tolist()
            )
            df["config_base"] = pd.Categorical(
                df["config_base"],
                categories=order_base,
                ordered=True,
            )

            # 多行标签，避免太挤
            def make_label(cfg: str) -> str:
                return cfg.replace("_h", "\nh").replace("_tp", "\ntp")

            label_order_base = [make_label(x) for x in order_base]
            mapping_base = {o: l for o, l in zip(order_base, label_order_base)}
            df["config_base_label"] = df["config_base"].astype(str).map(mapping_base)
            df["config_base_label"] = pd.Categorical(
                df["config_base_label"],
                categories=label_order_base,
                ordered=True,
            )

            # 构造含 fp 的完整横轴 config：先基础，再拼接 fp
            def make_cfg_full(n_nodes_val, h_val, true_p_val, false_p_val):
                base = make_cfg_base(n_nodes_val, h_val, true_p_val)
                fp_str = f"{float(false_p_val):.1f}"
                return f"{base}_fp{fp_str}"

            df["config"] = df.apply(
                lambda r: make_cfg_full(r["n_nodes"], r["h"], r["true_p"], r["false_p"]),
                axis=1,
            )

            global_fp_order = sorted(df["false_p"].unique())
            order_full = [
                f"{b}_fp{float(fp):.1f}" for b in order_base for fp in global_fp_order
            ]
            if not order_full:
                order_full = sorted(df["config"].unique())
            df["config"] = pd.Categorical(
                df["config"],
                categories=order_full,
                ordered=True,
            )

            def make_label_full(cfg: str) -> str:
                return (
                    cfg.replace("_h", "\nh")
                    .replace("_tp", "\ntp")
                    .replace("_fp", "\nfp")
                )

            label_order_full = [make_label_full(x) for x in order_full]
            mapping_full = {o: l for o, l in zip(order_full, label_order_full)}
            df["config_label"] = df["config"].astype(str).map(mapping_full)
            df["config_label"] = pd.Categorical(
                df["config_label"],
                categories=label_order_full,
                ordered=True,
            )

            method_map = {
                "max_soft_regulation": "max_soft_regulation",
                "dsgp": "dsgp",
                "无先验": "无先验",
            }
            df["method_cn"] = df["method"].map(method_map).fillna(df["method"])

            single_method_list = [
                ("max_soft_regulation", method_map["max_soft_regulation"]),
                ("dsgp", method_map["dsgp"]),
                ("无先验", method_map["无先验"]),
            ]

            method_palette = {
                method_map["max_soft_regulation"]: "#1f77b4",
                method_map["dsgp"]: "#ff7f0e",
                method_map["无先验"]: "#2ca02c",
            }

            for lambda_val in lambda_values_in_df:
                df_lambda = df[df["lambda_param"] == lambda_val].copy()
                if df_lambda.empty:
                    continue
                lambda_display = f"{lambda_val:.3f}".rstrip('0').rstrip('.')
                if lambda_display == "":
                    lambda_display = "0"
                lambda_tag = lambda_display.replace('.', 'p')

                fp_order_lambda = sorted(df_lambda["false_p"].unique())
                palette_lambda = sns.color_palette(
                    "colorblind", n_colors=len(fp_order_lambda)
                ) if fp_order_lambda else []
                fp_palette_lambda = {
                    fp: color for fp, color in zip(fp_order_lambda, palette_lambda)
                }

                for method_key, method_title in single_method_list:
                    df_m = df_lambda[df_lambda["method"] == method_key]
                    if df_m.empty:
                        continue

                    for metric in evaluations_name:
                        plt.figure(figsize=(12, 4.5))
                        sns.lineplot(
                            data=df_m,
                            x="config_base_label",
                            y=metric,
                            hue="false_p",
                            hue_order=fp_order_lambda if fp_order_lambda else None,
                            palette=fp_palette_lambda if fp_palette_lambda else None,
                            estimator="mean",
                            errorbar="sd",
                            marker='o',
                            linewidth=2,
                        )
                        plt.xticks(rotation=0)
                        plt.xlabel("config_param")
                        plt.ylabel(metric)
                        plt.title(
                            f"{method_title} 方法：{metric}（n={n_int}，m={m}，λ={lambda_display}，均值±方差）"
                        )
                        if fp_order_lambda:
                            plt.legend(title="fp")
                        else:
                            legend = plt.legend()
                            if legend:
                                legend.remove()
                        plt.tight_layout()

                        save_path = os.path.join(
                            n_out_dir,
                            f"lambda_{lambda_tag}_{method_key}_{metric}_line_by_config.png",
                        )
                        plt.savefig(save_path, dpi=220, bbox_inches='tight')
                        plt.close()

                for metric in evaluations_name:
                    plt.figure(figsize=(12, 4.8))
                    sns.lineplot(
                        data=df_lambda,
                        x="config_label",
                        y=metric,
                        hue="method_cn",
                        palette=method_palette,
                        estimator="mean",
                        errorbar="sd",
                        marker='o',
                        linewidth=2,
                    )
                    plt.xticks(rotation=0)
                    plt.xlabel("config_param")
                    plt.ylabel(metric)
                    plt.title(
                        f"对比：{metric}（n={n_int}，m={m}，λ={lambda_display}，均值±方差）"
                    )
                    plt.legend(title="方法")
                    plt.tight_layout()

                    cmp_path = os.path.join(
                        n_out_dir,
                        f"lambda_{lambda_tag}_compare_{metric}_line_by_config.png",
                    )
                    plt.savefig(cmp_path, dpi=220, bbox_inches='tight')
                    plt.close()
