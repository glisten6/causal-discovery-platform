

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
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
from algorithm.utils.constraints import ActiveConstraints,InactiveConstraints,OrientationConstraints,MonoConstraints
import pandas as pd
import seaborn as sns

# 动态注册并选择可用的中文字体，避免回退到 Arial 导致缺字
def _ensure_cjk_fonts():
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

# 动态选择可用的中文字体，避免回退到 Arial 导致缺字
def _apply_cjk_font():
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

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 从本地gcastle库导入
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG
from torch.optim import lr_scheduler
from torch.autograd import Variable
from utils.data_process import get_candidate
from collections import defaultdict

if __name__ == "__main__": 
    method   = 'nonlinear'
    sem_type = 'gp-add'
    out_dir  = "algorithm/CIR/exp/active_constraint/ex1"
    os.makedirs(out_dir, exist_ok=True)
    node_list = [10,30]
    h_list    = [2]  # n_edges = h * n_nodes
    x_labels  = []
    seeds = [3407, 7331, 104729,8675309]
    evaluations_name = ["shd","recall","precision","fdr","fpr"]
    # 运行控制：是否执行实验、是否从 CSV 作图（当不运行实验时）
    RUN_EXPERIMENTS = False
    PLOT_FROM_CSV = True
   
    # fdr: (reverse + FP) / (TP + FP)
    # tpr: TP/(TP + FN)
    # fpr: (reverse + FP) / (TN + FP)
    

    config = {

    "active":{
        "use":True,
        "method":"max",
        "threshold":0.6,
        "lamb":0.07,
        "name":"active",
        "model":ActiveConstraints
    },
    "inactive":{
        "use":False,
        "lamb":0.01,
        "model":InactiveConstraints,
        "name":"inactive"
    },
    "plus_minus":{
        "use":False,
        "lamb":0.01,
        "name":"plus_minus"
       
    },
    "orient":{
        "use":False,
        "l2_lambda":0.01,
        "l1_lambda":0.1,
         "model":OrientationConstraints,
         "name":"orient",
         "alpha":"max",
         "use_cumulative":True
    },
    "mono":{
        "use":False,
        "l1_lambda":0.01,
        "model":MonoConstraints,
        "name":"mono"
    }
}

        
        
    # 收集每个 seed 的原始指标行
    results_rows = []
    true_ps = [0.4,0.6]
    
    if RUN_EXPERIMENTS:
        for n in [300,1200]:
            n_out_dir = os.path.join(out_dir,f"{n}")
            os.makedirs(n_out_dir, exist_ok=True)
            for n_nodes in node_list:
                for h in h_list:
                    for true_p in true_ps:
                        for false_p in [0.2, 0]:
                            evaluations = defaultdict(int)
                            for seed in seeds:
                                n_edges = h * n_nodes
                                # ---- 生成真值带符号的 DAG ----
                                print(f"\n测试配置: 节点数量 = {n_nodes}, edge = {n_nodes*h}")
                                x_labels.append(f"n={n_nodes},e={n_nodes*h}")
                            
                                weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                                                    weight_range=(0.5, 2.0), seed=seed)

                                dataset = IIDSimulation(W=weighted_random_dag, n=n,
                                                        method=method, sem_type=sem_type)
                                true_dag, X = dataset.B, dataset.X

                                # ---- 获得先验集合 ----
                                candidate_active = get_candidate(true_dag, seed = seed,false_p=false_p,edge_frac=true_p)
                                candidate_dict = {
                                    "active": candidate_active,  # 若 active/use=True 才会被用到
                                }
                                print("有约束")
                                al1 = NotearsNonlinear(config = config,candidate_dict=candidate_dict,device_type="cpu")
                                al1.learn(X)
                                # 确保当前配置输出目录存在（修复未定义 file_dir）
                                file_dir = os.path.join(n_out_dir, f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}")
                                os.makedirs(file_dir, exist_ok=True)
                                GraphDAG(al1.causal_matrix, true_dag,show=False,save_name=file_dir + "/active.jpg")
                                met1 = MetricsDAG(al1.causal_matrix, true_dag)
                                print(met1.metrics)
                                for name in evaluations_name:
                                    evaluations[name] += met1.metrics[name]

                                # 记录单次 seed 的原始指标（有约束）
                                results_rows.append({
                                    "n": n,
                                    "true_p": true_p,
                                    "false_p": false_p,
                                    "n_nodes": n_nodes,
                                    "h": h,
                                    "seed": seed,
                                    "method": "with_constraint",
                                    "shd": met1.metrics["shd"],
                                    "recall": met1.metrics["recall"],
                                    "precision": met1.metrics["precision"],
                                    "fdr": met1.metrics["fdr"],
                                    "fpr": met1.metrics["fpr"],
                                })


                                print("无约束")

                                al = NotearsNonlinear(device_type="cpu")
                                al.learn(X)
                                # 确保当前配置输出目录存在（修复未定义 file_dir）
                                file_dir = os.path.join(n_out_dir, f"n{n_nodes}_h{h}_tp{true_p}_fp{false_p}_seed{seed}_no_constraints")
                                os.makedirs(file_dir, exist_ok=True)
                                GraphDAG(al.causal_matrix, true_dag,show=False,save_name=file_dir + "/active.jpg")
                                met0 = MetricsDAG(al.causal_matrix, true_dag)
                                print(met0.metrics)
                                # 记录单次 seed 的原始指标（无约束）
                                results_rows.append({
                                    "n": n,
                                    "true_p": true_p,
                                    "false_p": false_p,
                                    "n_nodes": n_nodes,
                                    "h": h,
                                    "seed": seed,
                                    "method": "no_constraint",
                                    "shd": met0.metrics["shd"],
                                    "recall": met0.metrics["recall"],
                                    "precision": met0.metrics["precision"],
                                    "fdr": met0.metrics["fdr"],
                                    "fpr": met0.metrics["fpr"],
                                })
                            
                            for name in evaluations_name:
                                evaluations[name] /= len(seeds)
                                        
                
            # # ---- 带先验训练（这里演示只用 mono）----
            # candidate_dict = {
            #     "active": candidate_active,  # 若 active/use=True 才会被用到
            #     "orient": candidate_orient,  # 若 orient/use=True 才会被用到
            #     "mono":   w_mono_adj         # 邻接型 [d,d]，行=来源、列=指向
            # }

            # sub_dir = os.path.join(n_out_dir, f"{n_nodes}_{h}")
            # os.makedirs(sub_dir, exist_ok=True)

            # # with priors
            # al1 = NotearsNonlinear(config=config, candidate_dict=candidate_dict, device_type="gpu")
            # al1.learn(X)
            # GraphDAG(al1.causal_matrix, true_dag, show=False, save_name=os.path.join(sub_dir, "with_priors.jpg"))
            # met1 = MetricsDAG(al1.causal_matrix, true_dag)
            # sm1  = sign_metrics(al1.causal_matrix, W_true)
            # print("with priors:", met1.metrics, sm1)

            # results['shd']['with_prior'].append(met1.metrics['shd'])
            # results['recall']['with_prior'].append(met1.metrics['recall'])
            # results['precision']['with_prior'].append(met1.metrics['precision'])
            # results['fdr']['with_prior'].append(met1.metrics['fdr'])
            # results['tpr']['with_prior'].append(met1.metrics['tpr'])
            # results['fpr']['with_prior'].append(met1.metrics['fpr'])
            # results['sign_acc']['with_prior'].append(sm1['sign_acc'])

            # # no priors
            # al0 = NotearsNonlinear(device_type="gpu")
            # al0.learn(X)
            # GraphDAG(al0.causal_matrix, true_dag, show=False, save_name=os.path.join(sub_dir, "no_prior.jpg"))
            # met0 = MetricsDAG(al0.causal_matrix, true_dag)
            # sm0  = sign_metrics(al0.causal_matrix, W_true)
            # print("no priors :", met0.metrics, sm0)

            # results['shd']['no_prior'].append(met0.metrics['shd'])
            # results['recall']['no_prior'].append(met0.metrics['recall'])
            # results['precision']['no_prior'].append(met0.metrics['precision'])
            # results['fdr']['no_prior'].append(met0.metrics['fdr'])
            # results['tpr']['no_prior'].append(met0.metrics['tpr'])
            # results['fpr']['no_prior'].append(met0.metrics['fpr'])
            # results['sign_acc']['no_prior'].append(sm0['sign_acc'])


    print("___________________________________________")
    # ================== 汇总与绘图（按 seed 求均值并显示方差） ==================
    if RUN_EXPERIMENTS or PLOT_FROM_CSV:
            # 按样本量分别绘图：标题与保存路径包含 n，不合并在一张图
            n_values = [300, 1200]
            print("画图")
            for n in n_values:
                n_out_dir = os.path.join(out_dir, f"{n}")
                os.makedirs(n_out_dir, exist_ok=True)
                csv_path = os.path.join(n_out_dir, "results_by_seed.csv")
                print("对比图")
                if RUN_EXPERIMENTS:
                    df_all = pd.DataFrame(results_rows)
                    df = df_all[df_all["n"] == n].copy()
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                else:
                    if not os.path.exists(csv_path):
                        print("找不到文件")
                        continue
                    df = pd.read_csv(csv_path)
                print("根据csv")
                sns.set(style="whitegrid")
                _ensure_cjk_fonts()
                _apply_cjk_font()
                plt.rcParams['axes.unicode_minus'] = False
                sns.set_theme()
                _apply_cjk_font()
                plt.rcParams['axes.unicode_minus'] = False
                # 组合横轴标签（不含 fp，用于单方法图）
                def make_cfg_base(n_nodes, h, true_p):
                    try:
                        nn = int(n_nodes)
                    except Exception:
                        nn = n_nodes
                    try:
                        hh = int(h)
                    except Exception:
                        hh = h
                    return f"n{nn}_h{hh}_tp{true_p}"

                # 基础配置（不含 fp）
                df["config_base"] = df.apply(lambda r: make_cfg_base(r["n_nodes"], r["h"], r["true_p"]), axis=1)
                order_base = (df[["n_nodes","h","true_p"]]
                         .drop_duplicates()
                         .sort_values(["n_nodes","h","true_p"])
                         .apply(lambda r: f"n{int(r['n_nodes'])}_h{int(r['h'])}_tp{r['true_p']}", axis=1)
                         .tolist())
                df["config_base"] = pd.Categorical(df["config_base"], categories=order_base, ordered=True)

                # 多行标签，避免斜着显示
                def make_label(cfg: str) -> str:
                    return cfg.replace("_h", "\nh").replace("_tp", "\ntp")
                label_order_base = [make_label(x) for x in order_base]
                df["config_base_label"] = df["config_base"].astype(str).map({o: l for o, l in zip(order_base, label_order_base)})
                df["config_base_label"] = pd.Categorical(df["config_base_label"], categories=label_order_base, ordered=True)

                # 更易区分的调色板
                fp_order = [0.2, 0.0]
                fp_palette = {0.2: "#0173b2", 0.0: "#de8f05"}  # colorblind 方案中的蓝/橙

                # 单方法图：x 为 4 个配置，hue=false_p 两条线
                for method_key, method_title in [("with_constraint", "有约束"), ("no_constraint", "无约束")]:
                    df_m = df[df["method"] == method_key]
                    if df_m.empty:
                        continue
                    for metric in evaluations_name:
                        plt.figure(figsize=(12, 4.5))
                        sns.lineplot(
                            data=df_m,
                            x="config_base_label",
                            y=metric,
                            hue="false_p",
                            hue_order=fp_order,
                            palette=fp_palette,
                            estimator="mean",
                            errorbar="sd",
                            marker='o',
                            linewidth=2,
                        )
                        plt.xticks(rotation=0)
                        plt.xlabel("config_param")
                        plt.ylabel(metric)
                        plt.title(f"{method_title}：{metric}（n={n}，均值±方差）")
                        plt.legend(title="fp")
                        plt.tight_layout()
                        save_path = os.path.join(n_out_dir, f"{method_key}_{metric}_line_by_config.png")
                        plt.savefig(save_path, dpi=220, bbox_inches='tight')
                        plt.close()

                # 对比图：x 为 8 个刻度（先 fp=0.2 的 4 个配置，再 fp=0 的 4 个配置），仅两条线（方法）
                method_map = {"with_constraint": "有约束", "no_constraint": "无约束"}
                df["method_cn"] = df["method"].map(method_map).fillna(df["method"])  # 兜底

                # 构造包含 fp 的横轴：将 fp 作为 config 的一部分
                def make_cfg_full(n_nodes, h, true_p, false_p):
                    base = make_cfg_base(n_nodes, h, true_p)
                    fp_str = f"{float(false_p):.1f}"
                    return f"{base}_fp{fp_str}"
                df["config"] = df.apply(lambda r: make_cfg_full(r["n_nodes"], r["h"], r["true_p"], r["false_p"]), axis=1)
                # 顺序：按基础配置遍历，每个配置内先 fp=0.2 再 fp=0.0（统一一位小数）
                order_full = [f"{b}_fp{fp:.1f}" for b in order_base for fp in [0.2, 0.0]]
                df["config"] = pd.Categorical(df["config"], categories=order_full, ordered=True)

                def make_label_full(cfg: str) -> str:
                    return cfg.replace("_h", "\nh").replace("_tp", "\ntp").replace("_fp", "\nfp")
                label_order_full = [make_label_full(x) for x in order_full]
                df["config_label"] = df["config"].astype(str).map({o: l for o, l in zip(order_full, label_order_full)})
                df["config_label"] = pd.Categorical(df["config_label"], categories=label_order_full, ordered=True)

                method_palette = {"有约束": "#0072B2", "无约束": "#D55E00"}  # 明显区分的两色

                for metric in evaluations_name:
                    plt.figure(figsize=(12, 4.8))
                    sns.lineplot(
                        data=df,
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
                    plt.title(f"对比：{metric}（n={n}，均值±方差）")
                    plt.legend(title="方法")
                    plt.tight_layout()
                    cmp_path = os.path.join(n_out_dir, f"compare_{metric}_line_by_config.png")
                    plt.savefig(cmp_path, dpi=220, bbox_inches='tight')
                    plt.close()
