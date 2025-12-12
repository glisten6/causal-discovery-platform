import os
import sys
import copy
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

from algorithm.utils.constraints import ActiveConstraints
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG

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
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


_ensure_cjk_fonts()
_apply_cjk_font()
plt.rcParams['axes.unicode_minus'] = False

# 再次确保项目根目录在路径中（以防从别处调用时出错）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == "__main__":
    method = 'nonlinear'
    m = 4
    sem_type = 'gp-add'
    out_dir = os.path.join(
        "algorithm",
        "CIR",
        "exp",
        "active_constraint",
        "closure_ablation_max"
    )
    os.makedirs(out_dir, exist_ok=True)

    alpha = 0.95
    node_list = [15,30]
    h_list = [2]  # n_edges = h * n_nodes
    seeds = [3407, 7329, 104729,2015]
    lambda_values = [1]
    evaluations_name = ["shd", "recall", "precision", "fdr", "fpr", "F1"]

    # 运行控制
    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True

    # 三种对比设置（均使用 ActiveConstraints 的 max 方法）
    BASE_ACTIVE_CONFIG = {
        "use": True,
        "method": "max",
        "threshold": 0.5,
        "lamb": 0.05,
        "name": "active",
        "model": ActiveConstraints,
        "alpha": alpha,
    }

    VARIANTS = {
        "closure_inverse": {
            "label": "传递闭包 + 逆矩阵",
            "palette": "#1f77b4",
            "config_patch": {
                "use_transitive_closure": True,
                "use_inverse_matrix": True,
            },
        },
        "closure_power": {
            "label": "传递闭包 + 幂法",
            "palette": "#ff7f0e",
            "config_patch": {
                "use_transitive_closure": True,
                "use_inverse_matrix": False,
            },
        },
        "no_constraint": {
            "label": "无约束",
            "palette": "#2ca02c",
            "config_patch": None,
        },
    }

    results_rows = []
    true_ps = [0.4, 0.6]

    if RUN_EXPERIMENTS:
        for n in [300]:
            n_out_dir = os.path.join(out_dir, f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            for lambda_val in lambda_values:
                lambda_display = f"{lambda_val:.3f}".rstrip('0').rstrip('.')
                lambda_tag = lambda_display.replace('.', 'p')
                lambda_out_dir = os.path.join(n_out_dir, f"lambda_{lambda_tag}")
                os.makedirs(lambda_out_dir, exist_ok=True)

                for n_nodes in node_list:
                    for h in h_list:
                        for true_p in true_ps:
                            for false_p in [0.2, 0.0]:
                                for seed in seeds:
                                    n_edges = h * n_nodes
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

                                    candidate_active = get_candidate(
                                        true_dag,
                                        seed=seed,
                                        false_p=false_p,
                                        edge_frac=true_p,
                                        alpha=1,
                                        use_transitive_closure=True,
                                        m=m
                                    )
                                    base_dir = os.path.join(
                                        lambda_out_dir,
                                        f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}"
                                    )
                                    os.makedirs(base_dir, exist_ok=True)

                                    for variant_key, variant_info in VARIANTS.items():
                                        label = variant_info["label"]
                                        print(
                                            f"Variant={label}, n={n}, 节点={n_nodes}, 边={n_edges}, "
                                            f"true_p={true_p}, false_p={false_p}, seed={seed}, λ={lambda_display}"
                                        )

                                        if variant_key == "no_constraint":
                                            learner = NotearsNonlinear(
                                                use_pytorch_optimizer=False,
                                                device_type="cpu",
                                            )
                                        else:
                                            cfg = copy.deepcopy({"active": BASE_ACTIVE_CONFIG})
                                            cfg["active"].update({"lamb": lambda_val})
                                            cfg["active"].update(variant_info["config_patch"])

                                            candidate_dict = {
                                                "active": candidate_active,
                                            }
                                            learner = NotearsNonlinear(
                                                config=cfg,
                                                candidate_dict=candidate_dict,
                                                use_pytorch_optimizer=False,
                                                device_type="cpu",
                                            )

                                        start_time = monotonic()
                                        learner.learn(X)
                                        train_time = monotonic() - start_time

                                        GraphDAG(
                                            learner.causal_matrix,
                                            true_dag,
                                            show=False,
                                            save_name=os.path.join(base_dir, f"{variant_key}.jpg"),
                                        )
                                        metrics = MetricsDAG(learner.causal_matrix, true_dag)
                                        print("variant_key:", variant_key)
                                        print(metrics.metrics)
                                        print(f"train_time_sec: {train_time:.3f}")
                                        row = {
                                            "n": n,
                                            "true_p": true_p,
                                            "false_p": false_p,
                                            "n_nodes": n_nodes,
                                            "h": h,
                                            "seed": seed,
                                            "lambda_param": lambda_val,
                                            "variant": variant_key,
                                            "variant_label": label,
                                            "train_time_sec": train_time,
                                        }
                                        for metric_name in evaluations_name:
                                            row[metric_name] = metrics.metrics[metric_name]
                                        results_rows.append(row)

            df_all_tmp = pd.DataFrame(results_rows)
            df_n = df_all_tmp[df_all_tmp["n"] == n].copy()
            csv_path_n = os.path.join(n_out_dir, "results_by_variant.csv")
            df_n.to_csv(csv_path_n, index=False, encoding="utf-8-sig")

    print("___________________________________________")

    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
        if RUN_EXPERIMENTS:
            if not results_rows:
                print("没有实验结果，跳过画图")
                sys.exit(0)
            df_all = pd.DataFrame(results_rows)
        else:
            dfs = []
            if os.path.exists(out_dir):
                for name in os.listdir(out_dir):
                    subdir = os.path.join(out_dir, name)
                    if not os.path.isdir(subdir):
                        continue
                    csv_path = os.path.join(subdir, "results_by_variant.csv")
                    if os.path.exists(csv_path):
                        dfs.append(pd.read_csv(csv_path))
            if not dfs:
                print("未找到任何 CSV 结果文件，跳过画图")
                sys.exit(0)
            df_all = pd.concat(dfs, ignore_index=True)

        n_values = sorted(df_all["n"].unique())
        print("自动识别到的 n 值:", n_values)

        palette = {key: info["palette"] for key, info in VARIANTS.items()}

        for n in n_values:
            n_int = int(n)
            n_out_dir = os.path.join(out_dir, f"{n_int}")
            os.makedirs(n_out_dir, exist_ok=True)
            csv_path = os.path.join(n_out_dir, "results_by_variant.csv")

            df = df_all[df_all["n"] == n].copy()
            if df.empty:
                continue
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 构造横轴标签
            def make_cfg_label(n_nodes_val, h_val, true_p_val, false_p_val):
                base = f"n{int(n_nodes_val)}_h{int(h_val)}_tp{true_p_val}"
                return f"{base}\nfp{float(false_p_val):.1f}"

            df["config"] = df.apply(
                lambda r: make_cfg_label(r["n_nodes"], r["h"], r["true_p"], r["false_p"]),
                axis=1,
            )
            config_order = (
                df[["n_nodes", "h", "true_p", "false_p"]]
                .drop_duplicates()
                .sort_values(["n_nodes", "h", "true_p", "false_p"])
                .apply(lambda r: make_cfg_label(r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3]), axis=1)
                .tolist()
            )
            df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)

            for metric in evaluations_name:
                plt.figure(figsize=(12, 4.8))
                sns.lineplot(
                    data=df,
                    x="config",
                    y=metric,
                    hue="variant",
                    palette=palette,
                    estimator="mean",
                    errorbar="sd",
                    marker='o',
                    linewidth=2,
                )
                plt.xticks(rotation=0)
                plt.xlabel("配置参数")
                plt.ylabel(metric)
                plt.title(f"Max 方法：{metric}（n={n_int}，均值±方差）")
                handles, labels = plt.gca().get_legend_handles_labels()
                label_map = {k: v["label"] for k, v in VARIANTS.items()}
                labels = [label_map.get(k, k) for k in labels]
                plt.legend(handles, labels, title="约束设置")
                plt.tight_layout()

                save_path = os.path.join(
                    n_out_dir,
                    f"compare_{metric}_max_closure_ablation.png",
                )
                plt.savefig(save_path, dpi=220, bbox_inches='tight')
                plt.close()
