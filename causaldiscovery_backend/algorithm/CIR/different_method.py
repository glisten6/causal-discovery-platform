import os
import sys
from copy import deepcopy

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


def _ensure_cjk_fonts():
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\NSimSun.ttf",
    ]
    added = False
    for path in candidates:
        try:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                added = True
        except Exception:
            pass
    if added:
        try:
            fm._rebuild()
        except Exception:
            pass


def _apply_cjk_font():
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "NSimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            matplotlib.rcParams["font.sans-serif"] = [name]
            return name
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return "DejaVu Sans"


_ensure_cjk_fonts()
_apply_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


if __name__ == "__main__":
    method = "nonlinear"
    m = 4
    sem_type = "gp-add"
    out_dir = f"algorithm/CIR/exp/active_constraint/different_method/{m}"
    os.makedirs(out_dir, exist_ok=True)

    node_list = [15, 30]
    h_list = [2]
    seeds = [3407, 7331, 104729]
    true_ps = [0.4, 0.6]
    false_ps = [0.2, 0.0]

    beta_values = [1.0, 5.0, 10.0]
    methods_to_compare = {
        "swish": {
            "threshold": 0.6,
            "threshold1": 0.6,
            "sigma": 0.1,
        },
        "potential": {
            "threshold": 0.6,
            "threshold1": 0.6,
            "sigma": 0.1,
        },
    }

    RUN_EXPERIMENTS = True
    PLOT_FROM_CSV = True

    base_config = {
        "active": {
            "use": True,
            "method": "max",
            "threshold": 0.6,
            "threshold1": 0.6,
            "lamb": 0.1,
            "name": "active",
            "model": ActiveConstraints,
            "use_transitive_closure": False,
            "alpha": 0.8,
            "beta": 5.0,
            "sigma": 0.1,
        }
    }

    results_rows = []

    if RUN_EXPERIMENTS:
        for n in [200, 300]:
            n_out_dir = os.path.join(out_dir, f"n_{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            for method_name, overrides in methods_to_compare.items():
                for beta in beta_values:
                    for n_nodes in node_list:
                        for h in h_list:
                            for true_p in true_ps:
                                for false_p in false_ps:
                                    for seed in seeds:
                                        config = deepcopy(base_config)
                                        active_cfg = config["active"]
                                        active_cfg["method"] = method_name
                                        active_cfg["beta"] = beta
                                        active_cfg.update(overrides)

                                        n_edges = h * n_nodes
                                        print(
                                            f"\n配置: n={n}, nodes={n_nodes}, edges={n_edges}, "
                                            f"true_p={true_p}, false_p={false_p}, seed={seed}, "
                                            f"method={method_name}, beta={beta}"
                                        )

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
                                            sem_type=sem_type,
                                        )
                                        true_dag, X = dataset.B, dataset.X

                                        candidate_active = get_candidate(
                                            true_dag,
                                            seed=seed,
                                            false_p=false_p,
                                            edge_frac=true_p,
                                            alpha=1,
                                            use_transitive_closure=True,
                                            m=m,
                                        )
                                        candidate_dict = {"active": candidate_active}

                                        base_dir = os.path.join(
                                            n_out_dir,
                                            method_name,
                                            f"beta_{str(beta).replace('.', 'p')}",
                                            f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}",
                                        )
                                        os.makedirs(base_dir, exist_ok=True)

                                        model_active = NotearsNonlinear(
                                            config=config,
                                            candidate_dict=candidate_dict,
                                            device_type="cpu",
                                        )
                                        model_active.learn(X)

                                        GraphDAG(
                                            model_active.causal_matrix,
                                            true_dag,
                                            show=False,
                                            save_name=os.path.join(base_dir, "active.jpg"),
                                        )
                                        metrics = MetricsDAG(model_active.causal_matrix, true_dag).metrics

                                        results_rows.append(
                                            {
                                                "n": n,
                                                "n_nodes": n_nodes,
                                                "h": h,
                                                "true_p": true_p,
                                                "false_p": false_p,
                                                "seed": seed,
                                                "method": method_name,
                                                "beta": beta,
                                                "shd": metrics["shd"],
                                                "recall": metrics["recall"],
                                                "precision": metrics["precision"],
                                                "fdr": metrics["fdr"],
                                                "fpr": metrics["fpr"],
                                            }
                                        )

        if results_rows:
            df_all = pd.DataFrame(results_rows)
            csv_all = os.path.join(out_dir, "results_all.csv")
            df_all.to_csv(csv_all, index=False, encoding="utf-8-sig")
            print(f"保存结果: {csv_all}")

            for n in sorted(df_all["n"].unique()):
                df_n = df_all[df_all["n"] == n].copy()
                n_out_dir = os.path.join(out_dir, f"n_{int(n)}")
                os.makedirs(n_out_dir, exist_ok=True)
                csv_path = os.path.join(n_out_dir, "results_by_seed.csv")
                df_n.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"保存 {n=} 到 {csv_path}")

    if PLOT_FROM_CSV:
        if not results_rows:
            csv_all = os.path.join(out_dir, "results_all.csv")
            if not os.path.exists(csv_all):
                print("未找到 results_all.csv，跳过画图")
                sys.exit(0)
            df_all = pd.read_csv(csv_all)
        else:
            df_all = pd.DataFrame(results_rows)

        if df_all.empty:
            print("没有可用数据，跳过画图")
            sys.exit(0)

        sns.set(style="whitegrid")
        _ensure_cjk_fonts()
        _apply_cjk_font()
        plt.rcParams["axes.unicode_minus"] = False

        def build_config_label(row):
            return (
                f"n{int(row['n_nodes'])}_h{int(row['h'])}_tp{row['true_p']}_fp{row['false_p']}"
            )

        df_all["config_label"] = df_all.apply(build_config_label, axis=1)

        metrics_to_plot = ["shd", "recall", "precision", "fdr", "fpr"]
        palette = sns.color_palette("colorblind", n_colors=len(beta_values))
        beta_palette = {beta: palette[idx] for idx, beta in enumerate(beta_values)}

        for n in sorted(df_all["n"].unique()):
            df_n = df_all[df_all["n"] == n].copy()
            if df_n.empty:
                continue

            n_out_dir = os.path.join(out_dir, f"n_{int(n)}")
            os.makedirs(n_out_dir, exist_ok=True)

            for method_name in methods_to_compare.keys():
                df_method = df_n[df_n["method"] == method_name].copy()
                if df_method.empty:
                    continue

                order = (
                    df_method[["n_nodes", "h", "true_p", "false_p"]]
                    .drop_duplicates()
                    .sort_values(["n_nodes", "h", "true_p", "false_p"])
                    .apply(
                        lambda r: f"n{int(r['n_nodes'])}_h{int(r['h'])}_tp{r['true_p']}_fp{r['false_p']}",
                        axis=1,
                    )
                    .tolist()
                )
                df_method["config_label"] = pd.Categorical(
                    df_method["config_label"], categories=order, ordered=True
                )

                for metric in metrics_to_plot:
                    plt.figure(figsize=(12, 5))
                    sns.lineplot(
                        data=df_method,
                        x="config_label",
                        y=metric,
                        hue="beta",
                        palette=beta_palette,
                        estimator="mean",
                        errorbar="sd",
                        marker="o",
                    )
                    plt.xlabel("配置 (n_nodes, h, true_p, false_p)")
                    plt.ylabel(metric)
                    plt.title(
                        f"{method_name} 方法 {metric} 指标对比 (n={int(n)})"
                    )
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()

                    save_path = os.path.join(
                        n_out_dir,
                        f"{method_name}_{metric}_beta_compare.png",
                    )
                    plt.savefig(save_path, dpi=220, bbox_inches="tight")
                    plt.close()
                    print(f"保存图表: {save_path}")

        print("绘图完成")
