import os
import sys
from time import monotonic

# Ensure project root is on sys.path so `import algorithm` works when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.utils.constraints import ActiveConstraints,InactiveConstraints,OrientationConstraints,MonoConstraints

# constraints_name = ["active", "inactive", "plus_minus", "orient"]

config = {

    "active":{
        "use":False,
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
        "use":True,
        "l1_lambda":0.01,
        "model":MonoConstraints,
        "name":"mono",



    }
}


import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



# 从本地gcastle库导入
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG
from torch.optim import lr_scheduler
from torch.autograd import Variable

type = 'ER'  # or `SF`
method = 'nonlinear'
sem_type = 'gp'

# 存储不同配置的结果
results = {
    'shd': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []},
    'recall': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []},
    'precision': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []},
    'fdr': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []},
    'tpr': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []},
    'fpr': {'no_prior': [], 'active': [], 'inactive': [], 'orient': []}
}

# 要测试的节点数量
node_list = [30,35,40]
# 要测试的h值（边与节点的倍数）
h_list = [3,4]

# 存储x轴标签
x_labels = []

# 遍历不同的节点数量和h值
for n_nodes in node_list:
    for h in h_list:
        print(f"\n测试配置: 节点数量 = {n_nodes}, edge = {n_nodes*h}")
        x_labels.append(f"n={n_nodes},e={n_nodes*h}")
        
        # 设置随机种子
        n_edges = h * n_nodes
        weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                            weight_range=(0.5, 2.0), seed=n_edges)

        dataset = IIDSimulation(W=weighted_random_dag, n=120,
                                method=method, sem_type=sem_type)
        true_dag, X = dataset.B, dataset.X

        matrix = np.zeros(true_dag.shape)

        # 生成一个与矩阵相同大小的随机数矩阵，范围在[0, 1)
        random_matrix = np.random.rand(*true_dag.shape)

        # 找出随机数小于采样概率的位置
        sampled_indices = random_matrix <= 0.4
        sampled_indices = sampled_indices.astype(int)
        # 对这些位置进行随机采样赋值
        candidate = true_dag * sampled_indices
        candidate_dict = {"active":candidate,"orient":candidate,"mono":}
        file_dir = f"algorithm/CIR/exp/compare_1/{n_nodes}_{h}"
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        # print("使用先验因果图进行修正的结果")
        # print("将先验因果图转化为惩罚约束")
        # print("有向边约束")

        al1 = NotearsNonlinear(config = config,candidate_dict=candidate_dict,device_type="gpu")

        al1.learn(X)
        
        GraphDAG(al1.causal_matrix, true_dag,show=False,save_name=file_dir + "/active.jpg")
        met1 = MetricsDAG(al1.causal_matrix, true_dag)
        print(met1.metrics)
        
        # 存储结果
        results['shd']['orient'].append(met1.metrics['shd'])
        results['recall']['orient'].append(met1.metrics['recall'])
        results['precision']['orient'].append(met1.metrics['precision'])
        results['fdr']['orient'].append(met1.metrics['fdr'])
        results['tpr']['orient'].append(met1.metrics['tpr'])
        results['fpr']['orient'].append(met1.metrics['fpr'])


        # print("没有先验因果图修正的结果")
        al = NotearsNonlinear(device_type="gpu")
        al.learn(X)
        GraphDAG(al.causal_matrix, true_dag,show=False,save_name=file_dir + "/no_prior.jpg")
        met = MetricsDAG(al.causal_matrix, true_dag)
        print(met.metrics)
        
        # 存储结果
        results['shd']['no_prior'].append(met.metrics['shd'])
        results['recall']['no_prior'].append(met.metrics['recall'])
        results['precision']['no_prior'].append(met.metrics['precision'])
        results['fdr']['no_prior'].append(met.metrics['fdr'])
        results['tpr']['no_prior'].append(met.metrics['tpr'])
        results['fpr']['no_prior'].append(met.metrics['fpr'])

      
        

# 绘制结果图表
metrics = ['shd', 'recall', 'precision', 'fdr', 'tpr', 'fpr']
metric_names = {'shd': 'SHD', 'recall': '召回率', 'precision': '精确率', 'fdr': 'FDR', 'tpr': 'TPR', 'fpr': 'FPR'}

# 方法名称映射（按可用结果自动选择要绘制的方法）
method_name_map = {
    'no_prior': '无先验',
    'orient': '有向边约束',
    'active': '有边约束',
    'inactive': '无边约束',
    'both': '同时约束'
}
color_palette = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

# 设置图表大小和样式
plt.figure(figsize=(22, 20))
plt.style.use('ggplot')

# 为每个指标创建子图
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)

    # 选择具有有效数据的方法：长度与 x_labels 一致且非空
    available_methods = []
    for method_key, series in results[metric].items():
        if len(series) == len(x_labels) and len(series) > 0:
            available_methods.append(method_key)

    if not available_methods:
        print(f"跳过绘制 {metric}：无可用数据（x={len(x_labels)}）")
        continue

    # 绘制每种可用方法的结果
    for j, method_key in enumerate(available_methods):
        color = color_palette[j % len(color_palette)]
        label = method_name_map.get(method_key, method_key)
        plt.plot(
            range(len(x_labels)),
            results[metric][method_key],
            marker='o', linewidth=2, markersize=8,
            color=color, label=label
        )
    
    # 设置图表标题和标签
    plt.title(f'{metric_names[metric]} 指标对比', fontsize=18, fontweight='bold')
    plt.xlabel('配置 (节点数,边数)', fontsize=14)
    plt.ylabel(metric_names[metric], fontsize=14)
    
    # 优化横坐标显示
    plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', frameon=True, facecolor='white', edgecolor='gray', framealpha=0.8)

# 调整子图之间的间距
plt.tight_layout(pad=2.0)
print("保存图表")

# 确保目录存在
os.makedirs('algorithm/CIR/exp/compare_1', exist_ok=True)

# 保存图表 - 使用更高DPI确保中文清晰显示
plt.savefig('algorithm/CIR/exp/compare_1/causal_discovery_metrics_comparison.png', bbox_inches='tight')

# 显示图表
plt.show()

print("分析完成，结果已保存到 causal_discovery_metrics_comparison.png")






