import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
import pandas as pd
import seaborn as sns

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def _format_lambda(value: float):
    display = f"{value:.3f}".rstrip('0').rstrip('.')
    if display == "":
        display = "0"
    tag = display.replace('.', 'p')
    return display, tag


def _make_cfg_base(n_nodes_val, h_val, true_p_val):
    try:
        nn = int(n_nodes_val)
    except Exception:
        nn = n_nodes_val
    try:
        hh = int(h_val)
    except Exception:
        hh = h_val
    return f"n{nn}_h{hh}_tp{true_p_val}"


def _make_cfg_full(n_nodes_val, h_val, true_p_val, false_p_val):
    base = _make_cfg_base(n_nodes_val, h_val, true_p_val)
    fp_str = f"{float(false_p_val):.1f}"
    return f"{base}_fp{fp_str}"


def _make_label(text: str) -> str:
    return text.replace("_h", "\nh").replace("_tp", "\ntp")


def _make_label_full(text: str) -> str:
    return text.replace("_h", "\nh").replace("_tp", "\ntp").replace("_fp", "\nfp")


def _plot_for_n(df: pd.DataFrame, n_int: int, out_dir: str, m: int, evaluations):
    if df.empty:
        return

    sns.set(style="whitegrid")
    _ensure_cjk_fonts()
    _apply_cjk_font()
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme()
    _apply_cjk_font()
    plt.rcParams['axes.unicode_minus'] = False

    df = df.copy()
    df["false_p"] = df["false_p"].astype(float)
    df["lambda_param"] = df["lambda_param"].astype(float)

    df["config_base"] = df.apply(
        lambda r: _make_cfg_base(r["n_nodes"], r["h"], r["true_p"]), axis=1
    )
    order_base = (
        df[["n_nodes", "h", "true_p"]]
        .drop_duplicates()
        .sort_values(["n_nodes", "h", "true_p"])
        .apply(lambda r: f"n{int(r['n_nodes'])}_h{int(r['h'])}_tp{r['true_p']}", axis=1)
        .tolist()
    )
    df["config_base"] = pd.Categorical(df["config_base"], categories=order_base, ordered=True)

    label_order_base = [_make_label(x) for x in order_base]
    mapping_base = dict(zip(order_base, label_order_base))
    df["config_base_label"] = df["config_base"].astype(str).map(mapping_base)
    df["config_base_label"] = pd.Categorical(
        df["config_base_label"], categories=label_order_base, ordered=True
    )

    fp_values = sorted(df["false_p"].unique(), reverse=True)
    config_order = [
        f"{base}_fp{float(fp):.1f}" for base in order_base for fp in fp_values
    ]
    df["config"] = df.apply(
        lambda r: _make_cfg_full(r["n_nodes"], r["h"], r["true_p"], r["false_p"]),
        axis=1,
    )
    df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)

    label_order_full = [_make_label_full(x) for x in config_order]
    mapping_full = dict(zip(config_order, label_order_full))
    df["config_label"] = df["config"].astype(str).map(mapping_full)
    df["config_label"] = pd.Categorical(
        df["config_label"], categories=label_order_full, ordered=True
    )

    lambda_values = sorted(df["lambda_param"].unique())
    lambda_display_map = {val: _format_lambda(val)[0] for val in lambda_values}
    df["lambda_label"] = df["lambda_param"].map(lambda_display_map)

    lambda_palette = {
        val: color
        for val, color in zip(
            lambda_values,
            sns.color_palette("colorblind", n_colors=len(lambda_values))
        )
    }
    lambda_color_map = {lambda_display_map[val]: lambda_palette[val] for val in lambda_values}

    lambda_order_labels = [lambda_display_map[val] for val in lambda_values]
    df["lambda_label"] = pd.Categorical(
        df["lambda_label"], categories=lambda_order_labels, ordered=True
    )

    for metric in evaluations:
        plt.figure(figsize=(12, 4.8))
        sns.lineplot(
            data=df,
            x="config_label",
            y=metric,
            hue="lambda_label",
            hue_order=lambda_order_labels,
            palette=lambda_color_map,
            estimator="mean",
            errorbar="sd",
            marker='o',
            linewidth=2,
        )
        plt.xticks(rotation=0)
        plt.xlabel("config_param")
        plt.ylabel(metric)
        plt.title(
            f"传递闭包约束 λ 对比：{metric}（n={n_int}，m={m}，均值±方差）"
        )
        plt.legend(title="λ")
        plt.tight_layout()

        save_path = os.path.join(
            out_dir,
            f"n{n_int}_compare_lambda_{metric}.png",
        )
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    method = 'nonlinear'
    m = 3
    sem_type = 'gp-add'
    lambda_values = [0.01,0.05, 0.1, 0.5]

    out_dir = f"algorithm/CIR/exp/active_constraint/lambda_tc_compare/{m}"
    os.makedirs(out_dir, exist_ok=True)

    node_list = [20, 35]
    h_list = [2]
    seeds = [3407, 7331, 104729]
    true_ps = [0.4, 0.6]
    false_ps = [0.2, 0.0]
    sample_sizes = [300, 1200]
    evaluations_name = ["shd", "recall", "precision", "fdr", "fpr"]

    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True

    base_config = {
        "active": {
            "use": True,
            "method": "max",
            "threshold": 0.6,
            "lamb": lambda_values[0],  # will be overwritten in loop
            "name": "active",
            "model": ActiveConstraints,
            "use_transitive_closure": True,
            "alpha": 0.8,
        },
    }

    results_rows = []

    if RUN_EXPERIMENTS:
        for n in sample_sizes:
            n_out_dir = os.path.join(out_dir, f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)

            results_rows_n = []

            for lambda_val in lambda_values:
                lambda_display, lambda_tag = _format_lambda(lambda_val)
                base_config_lambda = deepcopy(base_config)
                base_config_lambda["active"]["lamb"] = lambda_val

                for n_nodes in node_list:
                    for h in h_list:
                        for true_p in true_ps:
                            for false_p in false_ps:
                                for seed in seeds:
                                    n_edges = h * n_nodes
                                    print(
                                        f"\n测试配置: n={n}, 节点数量={n_nodes}, edge={n_nodes * h}, "
                                        f"true_p={true_p}, false_p={false_p}, seed={seed}, λ={lambda_display}"
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
                                        f"lambda_{lambda_tag}",
                                        f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}",
                                    )
                                    os.makedirs(base_dir, exist_ok=True)

                                    configur = deepcopy(base_config_lambda)

                                    learner = NotearsNonlinear(
                                        config=configur,
                                        candidate_dict=candidate_dict,
                                        device_type="gpu",
                                    )
                                    learner.learn(X)

                                    GraphDAG(
                                        learner.causal_matrix,
                                        true_dag,
                                        show=False,
                                        save_name=os.path.join(base_dir, "active_tc.jpg"),
                                    )

                                    metrics = MetricsDAG(learner.causal_matrix, true_dag)
                                    print(f"λ={lambda_display} 指标:", metrics.metrics)

                                    row = {
                                        "n": n,
                                        "true_p": true_p,
                                        "false_p": false_p,
                                        "n_nodes": n_nodes,
                                        "h": h,
                                        "seed": seed,
                                        "lambda_param": lambda_val,
                                    }
                                    for metric_name in evaluations_name:
                                        row[metric_name] = metrics.metrics[metric_name]

                                    results_rows.append(row)
                                    results_rows_n.append(row)

            df_n = pd.DataFrame(results_rows_n)
            csv_path_n = os.path.join(n_out_dir, "results_by_seed.csv")
            df_n.to_csv(csv_path_n, index=False, encoding="utf-8-sig")

            _plot_for_n(df_n, n_int=int(n), out_dir=n_out_dir, m=m, evaluations=evaluations_name)

    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
        dfs = []
        for name in os.listdir(out_dir):
            subdir = os.path.join(out_dir, name)
            if not os.path.isdir(subdir):
                continue
            csv_path = os.path.join(subdir, "results_by_seed.csv")
            if os.path.exists(csv_path):
                dfs.append((int(name), pd.read_csv(csv_path)))

        if RUN_EXPERIMENTS:
            by_n = {int(row["n"]): [] for row in results_rows}
            for row in results_rows:
                by_n[int(row["n"])].append(row)
            for n_val, rows in by_n.items():
                n_dir = os.path.join(out_dir, f"{n_val}")
                os.makedirs(n_dir, exist_ok=True)
                df_tmp = pd.DataFrame(rows)
                df_tmp.to_csv(
                    os.path.join(n_dir, "results_by_seed.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

        if dfs:
            dfs.sort(key=lambda x: x[0])
            for n_val, df_n in dfs:
                print(f"加载 CSV 绘图：n={n_val}")
                _plot_for_n(df_n, n_int=int(n_val), out_dir=os.path.join(out_dir, f"{n_val}"), m=m, evaluations=evaluations_name)
        else:
            print("未找到任何 CSV 结果文件，跳过画图")
