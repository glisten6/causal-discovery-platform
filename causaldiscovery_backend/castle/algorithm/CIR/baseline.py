import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
    'shd': {'no_prior': [], 'active': [], 'inactive': [], 'both': []},
    'recall': {'no_prior': [], 'active': [], 'inactive': [], 'both': []},
    'precision': {'no_prior': [], 'active': [], 'inactive': [], 'both': []},
    'fdr': {'no_prior': [], 'active': [], 'inactive': [], 'both': []},
    'tpr': {'no_prior': [], 'active': [], 'inactive': [], 'both': []},
    'fpr': {'no_prior': [], 'active': [], 'inactive': [], 'both': []}
}

# 要测试的节点数量
node_list = [15,20]
# 要测试的h值（边与节点的倍数）
h_list = [4,6]

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

        dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                                method=method, sem_type=sem_type)
        true_dag, X = dataset.B, dataset.X

        matrix = np.zeros(true_dag.shape)

        # 生成一个与矩阵相同大小的随机数矩阵，范围在[0, 1)
        random_matrix = np.random.rand(*true_dag.shape)

        

        
        # 生成inactive_matrix，条件是true_dag为0且inactive_samples_indices为1的位置
        inactive_matrix = np.zeros_like(true_dag)
        inactive_positions = (true_dag == 0) & (inactive_samples_indices == 1)
        inactive_matrix[inactive_positions] = 1

        file_dir = f"algorithm/CIR/exp/compare_1/{n_nodes}_{h}"
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        print("没有先验因果图修正的结果")
        al = NotearsNonlinear()
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

        print("使用先验因果图进行修正的结果")
        print("将先验因果图转化为惩罚约束")
        print("有向边约束")
        al1 = NotearsNonlinear(active_constraints=active_matrix, active_constraint_lambda=0.01, active_method="max")

        al1.learn(X)
        
        GraphDAG(al1.causal_matrix, true_dag,show=False,save_name=file_dir + "/active.jpg")
        met1 = MetricsDAG(al1.causal_matrix, true_dag)
        print(met1.metrics)
        
        # 存储结果
        results['shd']['active'].append(met1.metrics['shd'])
        results['recall']['active'].append(met1.metrics['recall'])
        results['precision']['active'].append(met1.metrics['precision'])
        results['fdr']['active'].append(met1.metrics['fdr'])
        results['tpr']['active'].append(met1.metrics['tpr'])
        results['fpr']['active'].append(met1.metrics['fpr'])

        print("使用先验因果图进行修正的结果")
        print("将先验因果图转化为惩罚约束")
        print("无边约束")
        al2 = NotearsNonlinear(inactive_constraints=inactive_matrix)
        al2.learn(X)
        GraphDAG(al2.causal_matrix, true_dag,show=False,save_name=file_dir+"/inactive.jpg")
        met2 = MetricsDAG(al2.causal_matrix, true_dag)
        print(met2.metrics)
        
        # 存储结果
        results['shd']['inactive'].append(met2.metrics['shd'])
        results['recall']['inactive'].append(met2.metrics['recall'])
        results['precision']['inactive'].append(met2.metrics['precision'])
        results['fdr']['inactive'].append(met2.metrics['fdr'])
        results['tpr']['inactive'].append(met2.metrics['tpr'])
        results['fpr']['inactive'].append(met2.metrics['fpr'])

        print("使用先验因果图进行修正的结果")
        print("将先验因果图转化为惩罚约束")
        print("同时存在有边约束和无边约束")
       
        # al = NotearsNonlinear(active_constraints=constraint_matrix,active_constraint_lambda=0.01,active_method="max",inactive_constraints=indicator)
        al3 = NotearsNonlinear(active_constraints=active_matrix, active_constraint_lambda=0.01, active_method="max",inactive_constraints=inactive_matrix)

        al3.learn(X)
        GraphDAG(al3.causal_matrix, true_dag,show=False,save_name=file_dir+"/both.jpg")

        met3 = MetricsDAG(al3.causal_matrix, true_dag)
        print(met3.metrics)
        
        # 存储结果
        results['shd']['both'].append(met3.metrics['shd'])
        results['recall']['both'].append(met3.metrics['recall'])
        results['precision']['both'].append(met3.metrics['precision'])
        results['fdr']['both'].append(met3.metrics['fdr'])
        results['tpr']['both'].append(met3.metrics['tpr'])
        results['fpr']['both'].append(met3.metrics['fpr'])

# 绘制结果图表
metrics = ['shd', 'recall', 'precision', 'fdr', 'tpr', 'fpr']
metric_names = {'shd': 'SHD', 'recall': '召回率', 'precision': '精确率', 'fdr': 'FDR', 'tpr': 'TPR', 'fpr': 'FPR'}
method_names = ['无先验', '有向边约束', '无边约束', '同时约束']
method_keys = ['no_prior', 'active', 'inactive', 'both']
colors = ['blue', 'green', 'red', 'purple']

# 设置图表大小和样式
plt.figure(figsize=(22, 20))
plt.style.use('ggplot')

# 为每个指标创建子图
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    
    # 绘制每种方法的结果
    for j, method in enumerate(method_keys):
        plt.plot(range(len(x_labels)), results[metric][method], marker='o', linewidth=2, markersize=8, color=colors[j], label=method_names[j])
    
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





