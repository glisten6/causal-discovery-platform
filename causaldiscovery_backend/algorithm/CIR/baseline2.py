import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd

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

# 存储不同方法之间的SHD差异以及与真实图的比较
method_comparison = {
    # 方法之间的比较
    'no_prior_vs_active': [],
    'no_prior_vs_inactive': [],
    'no_prior_vs_both': [],
    'active_vs_inactive': [],
    'active_vs_both': [],
    'inactive_vs_both': [],
    # 与真实图的比较
    'true_vs_no_prior': [],
    'true_vs_active': [],
    'true_vs_inactive': [],
    'true_vs_both': []
}

# 要测试的节点数量
node_list = [15, 20]
# 要测试的h值（边与节点的倍数）
h_list = [4, 6]

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

        # 生成一个与矩阵相同大小的随机数矩阵，范围在[0, 1)
        random_matrix = np.random.rand(*true_dag.shape)

        # 找出随机数小于采样概率的位置
        active_sampled_indices = random_matrix <= 0.4
        active_sampled_indices = active_sampled_indices.astype(int)
        # 对这些位置进行随机采样赋值
        active_matrix = true_dag * active_sampled_indices

        random_matrix = np.random.rand(*true_dag.shape)
        inactive_samples_indices = random_matrix <= 0.6
        inactive_samples_indices = inactive_samples_indices.astype(int)
        
        # 生成inactive_matrix，条件是true_dag为0且inactive_samples_indices为1的位置
        inactive_matrix = np.zeros_like(true_dag)
        inactive_positions = (true_dag == 0) & (inactive_samples_indices == 1)
        inactive_matrix[inactive_positions] = 1

        file_dir = f"algorithm/CIR/exp/compare_2/{n_nodes}_{h}"
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        print("没有先验因果图修正的结果")
        al = NotearsNonlinear()
        al.learn(X)
        no_prior_matrix = al.causal_matrix
        
        print("使用先验因果图进行修正的结果 - 有向边约束")
        al1 = NotearsNonlinear(active_constraints=active_matrix, active_constraint_lambda=0.01, active_method="max")
        al1.learn(X)
        active_matrix_result = al1.causal_matrix
        
        print("使用先验因果图进行修正的结果 - 无边约束")
        al2 = NotearsNonlinear(inactive_constraints=inactive_matrix)
        al2.learn(X)
        inactive_matrix_result = al2.causal_matrix
        
        print("使用先验因果图进行修正的结果 - 同时存在有边约束和无边约束")
        al3 = NotearsNonlinear(active_constraints=active_matrix, active_constraint_lambda=0.01, active_method="max", inactive_constraints=inactive_matrix)
        al3.learn(X)
        both_matrix_result = al3.causal_matrix

        # 生成不同方法之间的比较图
        print("\n生成不同方法之间的比较图:")
        
        # 无先验 vs 有向边约束
        GraphDAG(no_prior_matrix, active_matrix_result, show=False, save_name=file_dir + "/no_prior_vs_active.jpg", est_graph="No_Prior", true_graph="Active_Constraints")
        
        # 无先验 vs 无边约束
        GraphDAG(no_prior_matrix, inactive_matrix_result, show=False, save_name=file_dir + "/no_prior_vs_inactive.jpg", est_graph="No_Prior", true_graph="Inactive_Constraints")
        
        # 无先验 vs 同时约束
        GraphDAG(no_prior_matrix, both_matrix_result, show=False, save_name=file_dir + "/no_prior_vs_both.jpg", est_graph="No_Prior", true_graph="Both_Constraints")
        
        # 有向边约束 vs 无边约束
        GraphDAG(active_matrix_result, inactive_matrix_result, show=False, save_name=file_dir + "/active_vs_inactive.jpg", est_graph="Active_Constraints", true_graph="Inactive_Constraints")
        
        # 有向边约束 vs 同时约束
        GraphDAG(active_matrix_result, both_matrix_result, show=False, save_name=file_dir + "/active_vs_both.jpg", est_graph="Active_Constraints", true_graph="Both_Constraints")
        
        # 无边约束 vs 同时约束
        GraphDAG(inactive_matrix_result, both_matrix_result, show=False, save_name=file_dir + "/inactive_vs_both.jpg", est_graph="Inactive_Constraints", true_graph="Both_Constraints")
        
        # 与真实图的比较
        print("\n生成与真实图的比较图:")
        
        # 真实图 vs 无先验
        GraphDAG(true_dag=true_dag, est_dag=no_prior_matrix, show=False, save_name=file_dir + "/true_vs_no_prior.jpg", est_graph="No_Prior", true_graph="True_Graph")
        
        # 真实图 vs 有向边约束
        GraphDAG(true_dag=true_dag, est_dag=active_matrix_result, show=False, save_name=file_dir + "/true_vs_active.jpg", est_graph="Active_Constraints", true_graph="True_Graph")
        
        # 真实图 vs 无边约束
        GraphDAG(true_dag=true_dag, est_dag=inactive_matrix_result, show=False, save_name=file_dir + "/true_vs_inactive.jpg", est_graph="Inactive_Constraints", true_graph="True_Graph")
        
        # 真实图 vs 同时约束
        GraphDAG(true_dag=true_dag, est_dag=both_matrix_result, show=False, save_name=file_dir + "/true_vs_both.jpg", est_graph="Both_Constraints", true_graph="True_Graph")

        # 首先计算各方法与真实图的指标
        print("\n计算各方法与真实图的指标:")
        
        # 真实图 vs 无先验
        met_true_vs_no_prior = MetricsDAG(B_true = true_dag, B_est=no_prior_matrix)
        shd_true_vs_no_prior = met_true_vs_no_prior.metrics['shd']
        precision_true_vs_no_prior = met_true_vs_no_prior.metrics['precision']
        recall_true_vs_no_prior = met_true_vs_no_prior.metrics['recall']
        f1_true_vs_no_prior = met_true_vs_no_prior.metrics['F1']
        method_comparison['true_vs_no_prior'].append(shd_true_vs_no_prior)
        print(f"True_Graph vs No_Prior: SHD = {shd_true_vs_no_prior}, Precision = {precision_true_vs_no_prior:.4f}, Recall = {recall_true_vs_no_prior:.4f}, F1 = {f1_true_vs_no_prior:.4f}")
        
        # 真实图 vs 有向边约束
        met_true_vs_active = MetricsDAG(B_true=true_dag, B_est=active_matrix_result)
        shd_true_vs_active = met_true_vs_active.metrics['shd']
        precision_true_vs_active = met_true_vs_active.metrics['precision']
        recall_true_vs_active = met_true_vs_active.metrics['recall']
        f1_true_vs_active = met_true_vs_active.metrics['F1']
        method_comparison['true_vs_active'].append(shd_true_vs_active)
        print(f"True_Graph vs Active_Constraints: SHD = {shd_true_vs_active}, Precision = {precision_true_vs_active:.4f}, Recall = {recall_true_vs_active:.4f}, F1 = {f1_true_vs_active:.4f}")
        
        # 真实图 vs 无边约束
        met_true_vs_inactive = MetricsDAG(B_true=true_dag,B_est= inactive_matrix_result)
        shd_true_vs_inactive = met_true_vs_inactive.metrics['shd']
        precision_true_vs_inactive = met_true_vs_inactive.metrics['precision']
        recall_true_vs_inactive = met_true_vs_inactive.metrics['recall']
        f1_true_vs_inactive = met_true_vs_inactive.metrics['F1']
        method_comparison['true_vs_inactive'].append(shd_true_vs_inactive)
        print(f"True_Graph vs Inactive_Constraints: SHD = {shd_true_vs_inactive}, Precision = {precision_true_vs_inactive:.4f}, Recall = {recall_true_vs_inactive:.4f}, F1 = {f1_true_vs_inactive:.4f}")
        
        # 真实图 vs 同时约束
        met_true_vs_both = MetricsDAG(B_true=true_dag, B_est=both_matrix_result)
        shd_true_vs_both = met_true_vs_both.metrics['shd']
        precision_true_vs_both = met_true_vs_both.metrics['precision']
        recall_true_vs_both = met_true_vs_both.metrics['recall']
        f1_true_vs_both = met_true_vs_both.metrics['F1']
        method_comparison['true_vs_both'].append(shd_true_vs_both)
        print(f"True_Graph vs Both_Constraints: SHD = {shd_true_vs_both}, Precision = {precision_true_vs_both:.4f}, Recall = {recall_true_vs_both:.4f}, F1 = {f1_true_vs_both:.4f}")
        
        # 然后计算不同方法之间的SHD和指标差异（通过相减得到）
        print("\n计算不同方法之间的SHD和指标差异:")
        
        # 无先验 vs 有向边约束
        shd_no_prior_vs_active = abs(shd_true_vs_no_prior - shd_true_vs_active)
        precision_diff_no_prior_vs_active = abs(precision_true_vs_no_prior - precision_true_vs_active)
        recall_diff_no_prior_vs_active = abs(recall_true_vs_no_prior - recall_true_vs_active)
        f1_diff_no_prior_vs_active = abs(f1_true_vs_no_prior - f1_true_vs_active)
        method_comparison['no_prior_vs_active'].append(shd_no_prior_vs_active)
        print(f"No_Prior vs Active_Constraints: SHD Diff = {shd_no_prior_vs_active}, Precision Diff = {precision_diff_no_prior_vs_active:.4f}, Recall Diff = {recall_diff_no_prior_vs_active:.4f}, F1 Diff = {f1_diff_no_prior_vs_active:.4f}")
        
        # 无先验 vs 无边约束
        shd_no_prior_vs_inactive = abs(shd_true_vs_no_prior - shd_true_vs_inactive)
        precision_diff_no_prior_vs_inactive = abs(precision_true_vs_no_prior - precision_true_vs_inactive)
        recall_diff_no_prior_vs_inactive = abs(recall_true_vs_no_prior - recall_true_vs_inactive)
        f1_diff_no_prior_vs_inactive = abs(f1_true_vs_no_prior - f1_true_vs_inactive)
        method_comparison['no_prior_vs_inactive'].append(shd_no_prior_vs_inactive)
        print(f"No_Prior vs Inactive_Constraints: SHD Diff = {shd_no_prior_vs_inactive}, Precision Diff = {precision_diff_no_prior_vs_inactive:.4f}, Recall Diff = {recall_diff_no_prior_vs_inactive:.4f}, F1 Diff = {f1_diff_no_prior_vs_inactive:.4f}")
        
        # 无先验 vs 同时约束
        shd_no_prior_vs_both = abs(shd_true_vs_no_prior - shd_true_vs_both)
        precision_diff_no_prior_vs_both = abs(precision_true_vs_no_prior - precision_true_vs_both)
        recall_diff_no_prior_vs_both = abs(recall_true_vs_no_prior - recall_true_vs_both)
        f1_diff_no_prior_vs_both = abs(f1_true_vs_no_prior - f1_true_vs_both)
        method_comparison['no_prior_vs_both'].append(shd_no_prior_vs_both)
        print(f"No_Prior vs Both_Constraints: SHD Diff = {shd_no_prior_vs_both}, Precision Diff = {precision_diff_no_prior_vs_both:.4f}, Recall Diff = {recall_diff_no_prior_vs_both:.4f}, F1 Diff = {f1_diff_no_prior_vs_both:.4f}")
        
        # 有向边约束 vs 无边约束
        shd_active_vs_inactive = abs(shd_true_vs_active - shd_true_vs_inactive)
        precision_diff_active_vs_inactive = abs(precision_true_vs_active - precision_true_vs_inactive)
        recall_diff_active_vs_inactive = abs(recall_true_vs_active - recall_true_vs_inactive)
        f1_diff_active_vs_inactive = abs(f1_true_vs_active - f1_true_vs_inactive)
        method_comparison['active_vs_inactive'].append(shd_active_vs_inactive)
        print(f"Active_Constraints vs Inactive_Constraints: SHD Diff = {shd_active_vs_inactive}, Precision Diff = {precision_diff_active_vs_inactive:.4f}, Recall Diff = {recall_diff_active_vs_inactive:.4f}, F1 Diff = {f1_diff_active_vs_inactive:.4f}")
        
        # 有向边约束 vs 同时约束
        shd_active_vs_both = abs(shd_true_vs_active - shd_true_vs_both)
        precision_diff_active_vs_both = abs(precision_true_vs_active - precision_true_vs_both)
        recall_diff_active_vs_both = abs(recall_true_vs_active - recall_true_vs_both)
        f1_diff_active_vs_both = abs(f1_true_vs_active - f1_true_vs_both)
        method_comparison['active_vs_both'].append(shd_active_vs_both)
        print(f"Active_Constraints vs Both_Constraints: SHD Diff = {shd_active_vs_both}, Precision Diff = {precision_diff_active_vs_both:.4f}, Recall Diff = {recall_diff_active_vs_both:.4f}, F1 Diff = {f1_diff_active_vs_both:.4f}")
        
        # 无边约束 vs 同时约束
        shd_inactive_vs_both = abs(shd_true_vs_inactive - shd_true_vs_both)
        precision_diff_inactive_vs_both = abs(precision_true_vs_inactive - precision_true_vs_both)
        recall_diff_inactive_vs_both = abs(recall_true_vs_inactive - recall_true_vs_both)
        f1_diff_inactive_vs_both = abs(f1_true_vs_inactive - f1_true_vs_both)
        method_comparison['inactive_vs_both'].append(shd_inactive_vs_both)
        print(f"Inactive_Constraints vs Both_Constraints: SHD Diff = {shd_inactive_vs_both}, Precision Diff = {precision_diff_inactive_vs_both:.4f}, Recall Diff = {recall_diff_inactive_vs_both:.4f}, F1 Diff = {f1_diff_inactive_vs_both:.4f}")
        
        # 保存当前配置下的详细指标结果到CSV文件
        print("\n保存当前配置的详细指标结果到CSV文件...")
        
        # 创建详细指标结果字典
        detailed_metrics = {
            'Method': ['No_Prior', 'Active_Constraints', 'Inactive_Constraints', 'Both_Constraints'],
            'SHD vs True_Graph': [
                met_true_vs_no_prior.metrics['shd'],
                met_true_vs_active.metrics['shd'],
                met_true_vs_inactive.metrics['shd'],
                met_true_vs_both.metrics['shd']
            ],
            'Precision vs True_Graph': [
                met_true_vs_no_prior.metrics['precision'],
                met_true_vs_active.metrics['precision'],
                met_true_vs_inactive.metrics['precision'],
                met_true_vs_both.metrics['precision']
            ],
            'Recall vs True_Graph': [
                met_true_vs_no_prior.metrics['recall'],
                met_true_vs_active.metrics['recall'],
                met_true_vs_inactive.metrics['recall'],
                met_true_vs_both.metrics['recall']
            ],
            'F1 vs True_Graph': [
                met_true_vs_no_prior.metrics['F1'],
                met_true_vs_active.metrics['F1'],
                met_true_vs_inactive.metrics['F1'],
                met_true_vs_both.metrics['F1']
            ],
            'FDR vs True_Graph': [
                met_true_vs_no_prior.metrics['fdr'],
                met_true_vs_active.metrics['fdr'],
                met_true_vs_inactive.metrics['fdr'],
                met_true_vs_both.metrics['fdr']
            ],
            'TPR vs True_Graph': [
                met_true_vs_no_prior.metrics['tpr'],
                met_true_vs_active.metrics['tpr'],
                met_true_vs_inactive.metrics['tpr'],
                met_true_vs_both.metrics['tpr']
            ],
            'FPR vs True_Graph': [
                met_true_vs_no_prior.metrics['fpr'],
                met_true_vs_active.metrics['fpr'],
                met_true_vs_inactive.metrics['fpr'],
                met_true_vs_both.metrics['fpr']
            ]
        }
        
        # 添加方法间差异指标
        method_diff_metrics = {
            'Method Comparison': [
                'No_Prior vs Active_Constraints',
                'No_Prior vs Inactive_Constraints',
                'No_Prior vs Both_Constraints',
                'Active_Constraints vs Inactive_Constraints',
                'Active_Constraints vs Both_Constraints',
                'Inactive_Constraints vs Both_Constraints'
            ],
            'SHD Difference': [
                shd_no_prior_vs_active,
                shd_no_prior_vs_inactive,
                shd_no_prior_vs_both,
                shd_active_vs_inactive,
                shd_active_vs_both,
                shd_inactive_vs_both
            ],
            'Precision Difference': [
                precision_diff_no_prior_vs_active,
                precision_diff_no_prior_vs_inactive,
                precision_diff_no_prior_vs_both,
                precision_diff_active_vs_inactive,
                precision_diff_active_vs_both,
                precision_diff_inactive_vs_both
            ],
            'Recall Difference': [
                recall_diff_no_prior_vs_active,
                recall_diff_no_prior_vs_inactive,
                recall_diff_no_prior_vs_both,
                recall_diff_active_vs_inactive,
                recall_diff_active_vs_both,
                recall_diff_inactive_vs_both
            ],
            'F1 Difference': [
                f1_diff_no_prior_vs_active,
                f1_diff_no_prior_vs_inactive,
                f1_diff_no_prior_vs_both,
                f1_diff_active_vs_inactive,
                f1_diff_active_vs_both,
                f1_diff_inactive_vs_both
            ]
        }
        
        # 创建DataFrame并保存为CSV
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_csv_path = f"{file_dir}/detailed_metrics_n{n_nodes}_h{h}.csv"
        detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
        print(f"详细指标结果已保存到: {detailed_csv_path}")
        
        # 保存方法间差异指标
        method_diff_df = pd.DataFrame(method_diff_metrics)
        method_diff_csv_path = f"{file_dir}/method_diff_metrics_n{n_nodes}_h{h}.csv"
        method_diff_df.to_csv(method_diff_csv_path, index=False, encoding='utf-8-sig')
        print(f"方法间差异指标已保存到: {method_diff_csv_path}")

# 将结果保存到CSV文件
print("\n将结果保存到CSV文件...")

# 确保目录存在
os.makedirs('algorithm/CIR/exp/compare_2', exist_ok=True)

# 创建DataFrame用于保存结果
results_df = pd.DataFrame(method_comparison, index=x_labels)

# 保存为CSV文件
csv_path = 'algorithm/CIR/exp/compare_2/methods_comparison_results.csv'
results_df.to_csv(csv_path, encoding='utf-8-sig')  # 使用utf-8-sig编码以支持中文
print(f"结果已保存到CSV文件: {csv_path}")

# 绘制结果图表
comparison_names = {
    # 方法之间的比较
    'no_prior_vs_active': 'No_Prior vs Active_Constraints',
    'no_prior_vs_inactive': 'No_Prior vs Inactive_Constraints',
    'no_prior_vs_both': 'No_Prior vs Both_Constraints',
    'active_vs_inactive': 'Active_Constraints vs Inactive_Constraints',
    'active_vs_both': 'Active_Constraints vs Both_Constraints',
    'inactive_vs_both': 'Inactive_Constraints vs Both_Constraints',
    # 与真实图的比较
    'true_vs_no_prior': 'True_Graph vs No_Prior',
    'true_vs_active': 'True_Graph vs Active_Constraints',
    'true_vs_inactive': 'True_Graph vs Inactive_Constraints',
    'true_vs_both': 'True_Graph vs Both_Constraints'
}
colors = {
    'no_prior_vs_active': 'blue',
    'no_prior_vs_inactive': 'green',
    'no_prior_vs_both': 'red',
    'active_vs_inactive': 'purple',
    'active_vs_both': 'orange',
    'inactive_vs_both': 'brown',
    'true_vs_no_prior': 'darkblue',
    'true_vs_active': 'darkgreen',
    'true_vs_inactive': 'darkred',
    'true_vs_both': 'darkviolet'
}

# 设置图表大小和样式 - 增大图表以容纳更多图例
plt.figure(figsize=(18, 12))
plt.style.use('ggplot')

# 绘制每种比较的结果
for comparison, values in method_comparison.items():
    # 区分方法间比较和与真实图比较的线型
    if 'true_vs' in comparison:
        plt.plot(range(len(x_labels)), values, marker='s', linewidth=2.5, markersize=9, 
                 color=colors[comparison], label=comparison_names[comparison], linestyle='--')
    else:
        plt.plot(range(len(x_labels)), values, marker='o', linewidth=2, markersize=8, 
                 color=colors[comparison], label=comparison_names[comparison])

# 设置图表标题和标签
plt.title('Methods Comparison: SHD Differences', fontsize=20, fontweight='bold')
plt.xlabel('Configuration (Nodes,Edges)', fontsize=16)
plt.ylabel('SHD Difference', fontsize=16)

# 优化横坐标显示
plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha='right', fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, linestyle='--', alpha=0.7)
# 调整图例位置和样式，使用两列显示以节省空间
plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           frameon=True, facecolor='white', edgecolor='gray', 
           framealpha=0.9, ncol=2)

# 调整子图之间的间距
plt.tight_layout(pad=2.0)
print("保存图表")

# 确保目录存在
os.makedirs('algorithm/CIR/exp/compare_2', exist_ok=True)

# 保存图表 - 使用更高DPI确保中文清晰显示
plt.savefig('algorithm/CIR/exp/compare_2/methods_shd_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

chart_path = 'algorithm/CIR/exp/compare_2/methods_shd_comparison.png'
print("\n分析完成！结果已保存到以下文件：")
print(f"1. 图表: {chart_path}")
print(f"2. 总体比较CSV数据: {csv_path}")
print(f"3. 详细指标CSV数据: 保存在各配置目录下的detailed_metrics_*.csv文件")
print(f"4. 方法间差异指标CSV数据: 保存在各配置目录下的method_diff_metrics_*.csv文件")
print(f"您可以使用Excel等工具打开CSV文件进行进一步分析。")