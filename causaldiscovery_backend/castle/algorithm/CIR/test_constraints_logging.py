"""
测试约束日志输出

运行此脚本查看约束配置和运行时信息
"""
import numpy as np
import logging
from castle.algorithms import NotearsNonlinear
from castle.datasets import IIDSimulation, DAG

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('constraint_logs.txt', mode='w')  # 保存到文件
    ]
)

# 配置约束
config = {
    "active": {
        "use": True,
        "method": "max",
        "threshold": 0.6,
        "lamb": 0.01,
        "name": "active"
    },
    "inactive": {
        "use": False,
        "lamb": 0.01,
        "name": "inactive"
    },
    "orient": {
        "use": True,
        "l1_lambda": 0.01,
        "l2_lambda": 0.05,
        "alpha": 3.0,
        "use_cumulative": True,
        "name": "orient"
    }
}

# 生成模拟数据
n_nodes = 10
n_edges = 20
weighted_random_dag = DAG.erdos_renyi(
    n_nodes=n_nodes, 
    n_edges=n_edges,
    weight_range=(0.5, 2.0), 
    seed=42
)

dataset = IIDSimulation(
    W=weighted_random_dag, 
    n=500,
    method='nonlinear', 
    sem_type='gp'
)
true_dag, X = dataset.B, dataset.X

# 生成候选图（采样40%的真实边）
random_matrix = np.random.rand(*true_dag.shape)
sampled_indices = (random_matrix <= 0.4).astype(int)
candidate = true_dag * sampled_indices

# 准备 candidate_dict
candidate_dict = {
    "active": candidate,
    "orient": candidate
}

print("=" * 80)
print("Starting NOTEARS with Constraints")
print("=" * 80)
print(f"Data: n_samples={X.shape[0]}, n_nodes={X.shape[1]}")
print(f"True edges: {int(true_dag.sum())}")
print(f"Candidate edges: {int(candidate.sum())}")
print("=" * 80)

# 创建模型（这里会触发约束初始化日志）
model = NotearsNonlinear(
    config=config,
    candidate_dict=candidate_dict,
    max_iter=20,  # 减少迭代次数用于测试
    device_type='cpu'  # 或 'gpu'
)

# 运行学习（这里会触发运行时日志）
model.learn(X)

print("=" * 80)
print("Training completed! Check 'constraint_logs.txt' for detailed logs.")
print("=" * 80)

# 打印结果统计
estimated_dag = model.causal_matrix
print(f"Estimated edges: {int(estimated_dag.sum())}")
print(f"True positives: {int((estimated_dag * true_dag).sum())}")
print(f"False positives: {int((estimated_dag * (1 - true_dag)).sum())}")
print(f"False negatives: {int(((1 - estimated_dag) * true_dag).sum())}")



