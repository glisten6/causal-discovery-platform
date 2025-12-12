import os
import sys
from copy import deepcopy


# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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


if __name__ == "__main__":
    method = 'nonlinear'
    m = 4
    sem_type = 'gp-add'
    alpha_values = [0.8, 0.85]
    out_dir = f"algorithm/CIR/exp/active_constraint/alpha_compare/{m}/ex2"
    os.makedirs(out_dir, exist_ok=True)

    node_list = [15,30]
    h_list = [2]
    seeds = [3407, 7331, 104729]
    evaluations_name = ["shd", "recall", "precision", "fdr", "fpr"]

    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True

    base_config = {
        "active": {
            "use": True,
            "method": "max",
            "threshold": 0.6,
            "lamb": 0.1,
            "name": "active",
            "model": ActiveConstraints,
            "use_transitive_closure": True,
            "alpha": None,
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

    results_rows = []
    true_ps = [0.4]

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

                                for alpha in alpha_values:
                                    print(f"使用传递闭包约束，alpha={alpha}")
                                    config = deepcopy(base_config)
                                    config["active"]["alpha"] = alpha

                                    candidate_active = get_candidate(
                                        true_dag,
                                        seed=seed,
                                        false_p=false_p,
                                        edge_frac=true_p,
                                        alpha=alpha,
                                        use_transitive_closure=True,
                                        m=m,
                                    )
                                    candidate_dict = {
                                        "active": candidate_active,
                                    }

                                    base_dir = os.path.join(
                                        n_out_dir,
                                        (
                                            f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_"
                                            f"seed{seed}_alpha{alpha}"
                                        ),
                                    )
                                    os.makedirs(base_dir, exist_ok=True)

                                    al_tc = NotearsNonlinear(
                                        config=config,
                                        candidate_dict=candidate_dict,
                                        device_type="gpu",
                                    )
                                    al_tc.learn(X)

                                    GraphDAG(
                                        al_tc.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "active_tc.jpg"),
                                    )
                                    met_tc = MetricsDAG(al_tc.causal_matrix, true_dag)
                                    print(f"alpha={alpha} 指标:", met_tc.metrics)

                                    results_rows.append({
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "alpha": alpha,
                                        "shd": met_tc.metrics["shd"],
                                        "recall": met_tc.metrics["recall"],
                                        "precision": met_tc.metrics["precision"],
                                        "fdr": met_tc.metrics["fdr"],
                                        "fpr": met_tc.metrics["fpr"],
                                    })

            df_all_tmp = pd.DataFrame(results_rows)
            df_n = df_all_tmp[df_all_tmp["n"] == n].copy()
            csv_path_n = os.path.join(n_out_dir, "results_by_seed.csv")
            df_n.to_csv(csv_path_n, index=False, encoding="utf-8-sig")

    print("___________________________________________")

    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
        print("画图")

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

        n_values = sorted(df_all["n"].unique())
        alpha_order = sorted(alpha_values)
        alpha_palette = {
            alpha: color
            for alpha, color in zip(alpha_order, sns.color_palette("colorblind", len(alpha_order)))
        }

        for n in n_values:
            n_int = int(n)
            n_out_dir = os.path.join(out_dir, f"{n_int}")
            os.makedirs(n_out_dir, exist_ok=True)
            csv_path = os.path.join(n_out_dir, "results_by_seed.csv")

            df = df_all[df_all["n"] == n].copy()
            if df.empty:
                continue

            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            sns.set(style="whitegrid")
            _ensure_cjk_fonts()
            _apply_cjk_font()
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_theme()
            _apply_cjk_font()
            plt.rcParams['axes.unicode_minus'] = False

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

            fp_values = sorted(df["false_p"].unique(), reverse=True)

            def make_cfg_full(n_nodes_val, h_val, true_p_val, false_p_val):
                base = make_cfg_base(n_nodes_val, h_val, true_p_val)
                fp_str = f"{float(false_p_val):.1f}"
                return f"{base}_fp{fp_str}"

            df["config"] = df.apply(
                lambda r: make_cfg_full(r["n_nodes"], r["h"], r["true_p"], r["false_p"]),
                axis=1,
            )

            config_order = [
                f"{base}_fp{float(fp):.1f}"
                for base in order_base
                for fp in fp_values
            ]
            df["config"] = pd.Categorical(
                df["config"],
                categories=config_order,
                ordered=True,
            )

            def make_label_full(cfg: str) -> str:
                return (
                    cfg.replace("_h", "\nh")
                    .replace("_tp", "\ntp")
                    .replace("_fp", "\nfp")
                )

            label_order_full = [make_label_full(x) for x in config_order]
            mapping_full = {o: l for o, l in zip(config_order, label_order_full)}
            df["config_label"] = df["config"].astype(str).map(mapping_full)
            df["config_label"] = pd.Categorical(
                df["config_label"],
                categories=label_order_full,
                ordered=True,
            )

            for metric in evaluations_name:
                plt.figure(figsize=(12, 4.8))
                sns.lineplot(
                    data=df,
                    x="config_label",
                    y=metric,
                    hue="alpha",
                    hue_order=alpha_order,
                    palette=alpha_palette,
                    estimator="mean",
                    errorbar="sd",
                    marker='o',
                    linewidth=2,
                )
                plt.xticks(rotation=0)
                plt.xlabel("config_param")
                plt.ylabel(metric)
                plt.title(
                    f"传递闭包约束：{metric}（n={n_int}，m={m}，fp降序，均值±方差）"
                )
                plt.legend(title="alpha")
                plt.tight_layout()

                save_path = os.path.join(
                    n_out_dir,
                    f"compare_alpha_{metric}_line_by_config_fp.png",
                )
                plt.savefig(save_path, dpi=220, bbox_inches='tight')
                plt.close()

