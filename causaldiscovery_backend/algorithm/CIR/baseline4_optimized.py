import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.utils.constraints import ActiveConstraints, InactiveConstraints, OrientationConstraints
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG

# 配置
config = {
    "active": {
        "use": False,
        "method": "max",
        "threshold": 0.6,
        "lamb": 0.07,
        "name": "active",
        "model": ActiveConstraints
    },
    "inactive": {
        "use": False,
        "lamb": 0.01,
        "model": InactiveConstraints,
        "name": "inactive"
    },
    "orient": {
        "use": True,
        "l2_lambda": 0.01,
        "l1_lambda": 0.01,
        "model": OrientationConstraints,
        "name": "orient",
        "alpha": "max",
        "use_cumulative": True
    }
}

type_dag = 'ER'
method = 'nonlinear'
sem_type = 'gp'

# 全局锁，确保 GPU 访问安全
gpu_lock = threading.Lock()

def run_single_experiment(n_nodes, h, exp_type='no_prior'):
    """
    运行单个实验
    
    Parameters
    ----------
    n_nodes : int
        节点数量
    h : int
        边倍数
    exp_type : str
        实验类型 'no_prior' 或 'orient'
    
    Returns
    -------
    dict
        包含实验结果的字典
    """
    try:
        print(f"[开始] n={n_nodes}, h={h}, type={exp_type}")
        start_time = time.time()
        
        # 生成数据
        n_edges = h * n_nodes
        weighted_random_dag = DAG.erdos_renyi(
            n_nodes=n_nodes, 
            n_edges=n_edges,
            weight_range=(0.5, 2.0), 
            seed=n_edges
        )
        
        dataset = IIDSimulation(
            W=weighted_random_dag, 
            n=120,
            method=method, 
            sem_type=sem_type
        )
        true_dag, X = dataset.B, dataset.X
        
        # 生成候选图
        random_matrix = np.random.rand(*true_dag.shape)
        sampled_indices = (random_matrix <= 0.4).astype(int)
        candidate = true_dag * sampled_indices
        candidate_dict = {"active": candidate, "orient": candidate}
        
        # 创建输出目录
        file_dir = f"algorithm/CIR/exp/compare_1/{n_nodes}_{h}"
        os.makedirs(file_dir, exist_ok=True)
        
        # 使用锁保护 GPU 访问（关键：避免 GPU 内存冲突）
        with gpu_lock:
            if exp_type == 'no_prior':
                print(f"  [{n_nodes}_{h}] 无先验训练中...")
                model = NotearsNonlinear(device_type="gpu", max_iter=50)
                model.learn(X)
                save_name = file_dir + "/no_prior.jpg"
            else:  # orient
                print(f"  [{n_nodes}_{h}] 有约束训练中...")
                model = NotearsNonlinear(
                    config=config,
                    candidate_dict=candidate_dict,
                    device_type="gpu",
                    max_iter=50
                )
                model.learn(X)
                save_name = file_dir + "/orient.jpg"
        
        # 保存图表（不需要锁，在 CPU 上）
        GraphDAG(model.causal_matrix, true_dag, show=False, save_name=save_name)
        
        # 计算指标
        met = MetricsDAG(model.causal_matrix, true_dag)
        
        elapsed = time.time() - start_time
        print(f"[完成] n={n_nodes}, h={h}, type={exp_type}, 用时={elapsed:.1f}s")
        print(f"  结果: {met.metrics}")
        
        return {
            'n_nodes': n_nodes,
            'h': h,
            'exp_type': exp_type,
            'metrics': met.metrics,
            'time': elapsed,
            'success': True
        }
        
    except Exception as e:
        print(f"[错误] n={n_nodes}, h={h}, type={exp_type}: {str(e)}")
        return {
            'n_nodes': n_nodes,
            'h': h,
            'exp_type': exp_type,
            'success': False,
            'error': str(e)
        }


def run_experiments_parallel(node_list, h_list, max_workers=2):
    """
    并行运行实验（在单个 GPU 上通过线程池）
    
    Parameters
    ----------
    node_list : list
        节点数量列表
    h_list : list
        边倍数列表
    max_workers : int
        并行工作线程数（推荐 2-4，避免 GPU 过载）
    """
    print("=" * 80)
    print(f"开始并行实验: {len(node_list)} 个节点配置 × {len(h_list)} 个边倍数 × 2 个实验类型")
    print(f"并行度: {max_workers} 个工作线程")
    print("=" * 80)
    
    # 生成所有任务
    tasks = []
    for n_nodes in node_list:
        for h in h_list:
            tasks.append((n_nodes, h, 'no_prior'))
            tasks.append((n_nodes, h, 'orient'))
    
    print(f"总任务数: {len(tasks)}")
    
    # 存储结果
    results = {
        'shd': {'no_prior': [], 'orient': []},
        'recall': {'no_prior': [], 'orient': []},
        'precision': {'no_prior': [], 'orient': []},
        'fdr': {'no_prior': [], 'orient': []},
        'tpr': {'no_prior': [], 'orient': []},
        'fpr': {'no_prior': [], 'orient': []}
    }
    
    x_labels = []
    completed = 0
    total_time = time.time()
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_single_experiment, n, h, exp_type): (n, h, exp_type)
            for n, h, exp_type in tasks
        }
        
        # 收集结果
        task_results = {}
        for future in as_completed(future_to_task):
            n, h, exp_type = future_to_task[future]
            result = future.result()
            
            completed += 1
            print(f"\n进度: {completed}/{len(tasks)} 完成")
            
            if result['success']:
                key = f"{n}_{h}"
                if key not in task_results:
                    task_results[key] = {}
                task_results[key][exp_type] = result['metrics']
    
    # 按顺序整理结果
    for n_nodes in node_list:
        for h in h_list:
            key = f"{n_nodes}_{h}"
            x_labels.append(f"n={n_nodes},e={n_nodes*h}")
            
            if key in task_results:
                for exp_type in ['no_prior', 'orient']:
                    if exp_type in task_results[key]:
                        metrics = task_results[key][exp_type]
                        for metric_name in results.keys():
                            results[metric_name][exp_type].append(metrics[metric_name])
    
    elapsed_total = time.time() - total_time
    print("\n" + "=" * 80)
    print(f"所有实验完成! 总用时: {elapsed_total:.1f}s ({elapsed_total/60:.1f}分钟)")
    print("=" * 80)
    
    return results, x_labels


def plot_results(results, x_labels):
    """绘制结果图表"""
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
        'orient': '有向边约束'
    }
    color_palette = ['blue', 'green']
    
    plt.figure(figsize=(22, 20))
    plt.style.use('ggplot')
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        
        available_methods = []
        for method_key, series in results[metric].items():
            if len(series) == len(x_labels) and len(series) > 0:
                available_methods.append(method_key)
        
        if not available_methods:
            print(f"跳过绘制 {metric}：无可用数据")
            continue
        
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
    
    os.makedirs('algorithm/CIR/exp/compare_1', exist_ok=True)
    plt.savefig('algorithm/CIR/exp/compare_1/causal_discovery_metrics_comparison.png', 
               bbox_inches='tight', dpi=150)
    
    print("图表已保存到 causal_discovery_metrics_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 配置
    node_list = [30, 35, 40]
    h_list = [3, 4]
    
    # 并行度设置
    # max_workers=1: 顺序执行（最安全）
    # max_workers=2: 轻度并行（推荐，GPU 内存足够时）
    # max_workers=3-4: 激进并行（需要大显存，可能OOM）
    max_workers = 2
    
    # 运行并行实验
    results, x_labels = run_experiments_parallel(node_list, h_list, max_workers=max_workers)
    
    # 绘制结果
    plot_results(results, x_labels)
    
    print("\n分析完成!")










