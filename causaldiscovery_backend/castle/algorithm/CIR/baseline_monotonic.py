"""
使用单调性约束的因果发现实验

与 baseline4 类似，但使用带单调性的数据生成

支持两种数据生成模式：
- 'simple': 使用预定义函数（快速）
- 'mlp': 使用多层神经网络（更复杂、更真实）
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.utils.constraints import OrientationConstraints
from monotonic_data_generation import MonotonicDAGGenerator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from castle.algorithms import NotearsNonlinear

# ==================== 实验配置 ====================
# 数据生成模式: 'simple' 或 'mlp'
GENERATION_MODE = 'mlp'  # 改为 'simple' 使用简单函数

# 配置约束
config = {
    "orient": {
        "use": True,
        "l2_lambda": 0.01,
        "l1_lambda": 0.01,
        "model": OrientationConstraints,
        "name": "orient",
        "alpha": 3.0,
        "use_cumulative": True
    }
}

# 存储结果
results = {
    'shd': {'no_prior': [], 'monotonic_prior': []},
    'recall': {'no_prior': [], 'monotonic_prior': []},
    'precision': {'no_prior': [], 'monotonic_prior': []},
    'fdr': {'no_prior': [], 'monotonic_prior': []},
    'tpr': {'no_prior': [], 'monotonic_prior': []},
    'fpr': {'no_prior': [], 'monotonic_prior': []}
}

# 实验配置
node_list = [20, 30]
edge_multiplier_list = [2, 3]
x_labels = []

print("=" * 80)
print("单调性约束因果发现实验")
print(f"数据生成模式: {GENERATION_MODE.upper()}")
print("=" * 80)

# 遍历不同配置
for n_nodes in node_list:
    for h in edge_multiplier_list:
        n_edges = n_nodes * h
        print(f"\n{'='*60}")
        print(f"配置: 节点={n_nodes}, 边={n_edges} (倍数={h}), 模式={GENERATION_MODE}")
        print(f"{'='*60}")
        
        x_labels.append(f"n={n_nodes},e={n_edges}")
        
        # 1. 生成带单调性的 DAG 和数据
        print(f"\n[1/4] 生成单调性 DAG (模式: {GENERATION_MODE})...")
        gen = MonotonicDAGGenerator(
            n_nodes=n_nodes,
            n_edges=n_edges,
            monotonic_ratio=0.6,  # 60% 边是单调的
            seed=n_edges,  # 使用 n_edges 作为种子确保可复现
            generation_mode=GENERATION_MODE  # 使用指定的生成模式
        )
        gen.print_summary()
        
        print("\n[2/4] 生成数据...")
        X = gen.generate_data(n_samples=500, noise_scale=1.0)
        true_dag = gen.W
        
        print(f"  数据形状: {X.shape}")
        print(f"  真实边数: {int((true_dag != 0).sum())}")
        
        # 2. 提取单调边作为先验（采样50%的单调边）
        print("\n[3/4] 提取单调性先验...")
        monotonic_constraint = gen.sample_monotonic_edges(sample_ratio=0.5)
        candidate_dict = {"orient": monotonic_constraint}
        
        n_total_edges = int((true_dag != 0).sum())
        n_monotonic_edges = int((np.abs(gen.monotonic_matrix) == 1).sum())
        n_prior_edges = int(monotonic_constraint.sum())
        
        print(f"  总边数: {n_total_edges}")
        print(f"  单调边数: {n_monotonic_edges} ({n_monotonic_edges/n_total_edges*100:.1f}%)")
        print(f"  先验边数: {n_prior_edges} ({n_prior_edges/n_total_edges*100:.1f}%)")
        print(f"  先验覆盖: {n_prior_edges/n_monotonic_edges*100:.1f}% 的单调边")
        
        # 创建输出目录
        file_dir = f"algorithm/CIR/exp/monotonic/{n_nodes}_{h}"
        os.makedirs(file_dir, exist_ok=True)
        
        # 3. 无先验实验
        print("\n[4/4] 运行实验...")
        print("  (a) 无先验 NOTEARS...")
        model_no_prior = NotearsNonlinear(device_type="gpu", max_iter=50)
        model_no_prior.learn(X)
        
        GraphDAG(model_no_prior.causal_matrix, true_dag, 
                show=False, save_name=file_dir + "/no_prior.jpg")
        met_no_prior = MetricsDAG(model_no_prior.causal_matrix, true_dag)
        
        print(f"    无先验结果: {met_no_prior.metrics}")
        
        # 存储结果
        for metric_name in results.keys():
            results[metric_name]['no_prior'].append(met_no_prior.metrics[metric_name])
        
        # 4. 带单调性先验实验
        print("  (b) 带单调性先验 NOTEARS...")
        model_with_prior = NotearsNonlinear(
            config=config,
            candidate_dict=candidate_dict,
            device_type="gpu",
            max_iter=50
        )
        model_with_prior.learn(X)
        
        GraphDAG(model_with_prior.causal_matrix, true_dag,
                show=False, save_name=file_dir + "/monotonic_prior.jpg")
        met_with_prior = MetricsDAG(model_with_prior.causal_matrix, true_dag)
        
        print(f"    单调性先验结果: {met_with_prior.metrics}")
        
        # 存储结果
        for metric_name in results.keys():
            results[metric_name]['monotonic_prior'].append(met_with_prior.metrics[metric_name])
        
        # 保存此配置的数据和约束
        np.save(file_dir + '/true_dag.npy', true_dag)
        np.save(file_dir + '/monotonic_matrix.npy', gen.monotonic_matrix)
        np.save(file_dir + '/constraint.npy', monotonic_constraint)
        np.save(file_dir + '/data.npy', X)
        
        # 可视化此配置
        gen.visualize(save_path=file_dir + '/dag_structure.png')
        
        print(f"\n  结果已保存到 {file_dir}/")

# 绘制对比图表
print("\n" + "=" * 80)
print("绘制结果对比图...")
print("=" * 80)

metrics = ['shd', 'recall', 'precision', 'fdr', 'tpr', 'fpr']
metric_names = {
    'shd': 'SHD', 
    'recall': '召回率', 
    'precision': '精确率', 
    'fdr': 'FDR', 
    'tpr': 'TPR', 
    'fpr': 'FPR'
}

method_name_map = {
    'no_prior': '无先验',
    'monotonic_prior': '单调性先验'
}
color_palette = ['blue', 'green']

plt.figure(figsize=(22, 20))
plt.style.use('ggplot')

for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    
    # 检查数据可用性
    available_methods = []
    for method_key, series in results[metric].items():
        if len(series) == len(x_labels) and len(series) > 0:
            available_methods.append(method_key)
    
    if not available_methods:
        print(f"跳过 {metric}：无数据")
        continue
    
    # 绘制
    for j, method_key in enumerate(available_methods):
        color = color_palette[j % len(color_palette)]
        label = method_name_map.get(method_key, method_key)
        plt.plot(
            range(len(x_labels)),
            results[metric][method_key],
            marker='o', linewidth=2, markersize=8,
            color=color, label=label
        )
    
    plt.title(f'{metric_names[metric]} 指标对比', fontsize=18, fontweight='bold')
    plt.xlabel('配置 (节点数,边数)', fontsize=14)
    plt.ylabel(metric_names[metric], fontsize=14)
    plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', frameon=True, 
              facecolor='white', edgecolor='gray', framealpha=0.8)

plt.tight_layout(pad=2.0)

# 保存图表
os.makedirs('algorithm/CIR/exp/monotonic', exist_ok=True)
plt.savefig('algorithm/CIR/exp/monotonic/comparison.png', bbox_inches='tight', dpi=150)
print("对比图已保存到 algorithm/CIR/exp/monotonic/comparison.png")

plt.show()

print("\n" + "=" * 80)
print("实验完成!")
print("=" * 80)

