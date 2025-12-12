

from causallearn.utils.cit import CIT
from functools import partial
from copy import deepcopy
import os
import sys
import numpy as np
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms.pc.pc import  find_skeleton

from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG

from algorithm.utils.independence_utils import get_skeletion_d_sep,get_collide_structure,get_collide_constraint,bootstrap_pc_edge_freq


n_nodes = 15
n_edges = 10
method = 'nonlinear'
sem_type = 'gp-add'
m = 4
out_dir = f"algorithm/CIR/exp/has_more_edges/ex4/{m}"
os.makedirs(out_dir, exist_ok=True)
alpha = 0.8  # 先验构造 / 约束里会用到的 alpha

# node_list = [15, 30]
node_list = [20]
h_list = [2]
# true_ps = [0.4,0.6]
true_ps = [0.4]
# seeds = [3407, 7331, 104729, 8675309]
seeds = [3407]
for n in [150]:

    for n_nodes in node_list:
        for h in h_list:
            for true_p in true_ps:
                # for false_p in [0.2, 0.0]:
                for false_p in [0.0]:
                    for seed in seeds:
                        n_edges = h * n_nodes
                        print(
                            f"\n测试配置: n={n}, 节点数量={n_nodes}, edge={n_nodes * h}, "
                            f"true_p={true_p}, false_p={false_p}, seed={seed}"
                        )

                      

                        # ---- 生成真值带权 DAG ----
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
                            sem_type=sem_type
                        )
                        true_dag, X = dataset.B, dataset.X
                        print("真实矩阵")
                        print(true_dag)
                        constraints,colliders,confidence = bootstrap_pc_edge_freq(X,alpha=0.05,ci_method="rcit",n_boot=10)
                        print("约束")
                        print(constraints)
                        print("置信度")
                        print(confidence)
                        print("碰撞mask")
                        print(colliders)
                        print("p值矩阵")
                        
                        
                        print("******" * 10)

