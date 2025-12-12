# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
from re import sub
from typing import Callable, List, Any, Union, Optional
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from castle.algorithms import NotearsNonlinear
import numpy as np
import torch
import pandas as pd




class MatrixCompatibilityScorer:
    """
    基于邻接矩阵的兼容性评分器，专门用于处理numpy数组或tensor张量表示的因果图
    只考虑单向边关系，忽略双向边
    """
    def __init__(self, num_subsets: int, subset_size: float = 0.5,algo = "notearsmlp",algo_params=None):
        """
        初始化矩阵兼容性评分器
        
        参数:
            num_subsets: 要生成的子集数量
            subset_size: 子集大小占总变量数的比例，默认为0.5
        """
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.marginal_matrices = []
        self.variables_list = []
        self.algo_name = algo
        self.algo_params = algo_params
       
    
    def _draw_subsets(self, n_variables: int) -> List[List[int]]:
        """
        随机生成变量子集
        
        参数:
            n_variables: 变量总数
            
        返回:
            变量索引子集列表
        """
        subset_size = math.floor(n_variables * self.subset_size)
        subset_list = []
        for _ in range(self.num_subsets):
            subset = np.random.choice(range(n_variables), size=subset_size, replace=False)
            subset_list.append(subset.tolist())
        return subset_list
    
    @staticmethod
    def _ensure_numpy(matrix: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        确保输入矩阵是numpy数组
        
        参数:
            matrix: 输入矩阵，可以是numpy数组或torch张量
            
        返回:
            numpy数组形式的矩阵
        """
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().cpu().numpy()
        elif isinstance(matrix, np.ndarray):
            return matrix
        else:
            raise TypeError(f"不支持的矩阵类型: {type(matrix)}")
  
 
    
    @staticmethod
    def _calculate_shd(matrix1: np.ndarray, matrix2: np.ndarray) -> int:
        """
        计算两个邻接矩阵之间的结构汉明距离(SHD)
        只考虑单向边，忽略双向边
        
        参数:
            matrix1: 第一个邻接矩阵
            matrix2: 第二个邻接矩阵
            
        返回:
            结构汉明距离
        """
        # # 首先移除两个矩阵中的双向边
        # matrix1 = MatrixCompatibilityScorer._remove_bidirectional_edges(matrix1)
        # matrix2 = MatrixCompatibilityScorer._remove_bidirectional_edges(matrix2)
        
        # 计算差异
        diff = np.abs(matrix1 - matrix2)
        
        # 计算SHD（结构汉明距离）
        return int(np.sum(diff > 0))

    def marginalize(self, remaining_indices: List[int], joint_matrix: np.ndarray) -> np.ndarray:
        """
        边缘化图，保留指定的节点索引，移除其他节点
        
        参数:
            remaining_indices: 要保留的节点索引列表
            joint_matrix: 联合图的邻接矩阵
            
        返回:
            边缘化后的邻接矩阵
        """
        # 创建图的副本
        g_marginalised = joint_matrix.copy()
        n = joint_matrix.shape[0]
        
        # 转换为集合以便快速查找
        subset = set(remaining_indices)
        
        # 记录混杂节点对
        confounded_nodes = set()
        
        # 找出需要移除的节点索引
        nodes_to_remove = [i for i in range(n) if i not in subset]
        
        # 逐个移除节点
        for node_idx in nodes_to_remove:
            g_marginalised, confounded_nodes = self._marginalise_node(g_marginalised, node_idx, confounded_nodes)
            
            # 由于移除节点后，矩阵大小减小，需要更新后续要移除的节点索引
            # 这里我们简化处理，每次移除一个节点后重新计算剩余节点的索引
            if len(nodes_to_remove) > 1:  # 如果还有节点需要移除
                # 更新剩余节点的索引
                for i in range(len(nodes_to_remove)):
                    if nodes_to_remove[i] > node_idx:
                        nodes_to_remove[i] -= 1
        
        # # 处理混杂节点对，添加双向边
        # for x, y in confounded_nodes:
        #     # 检查这些节点是否在剩余的节点中
        #     if x in subset and y in subset:
        #         # 获取在新矩阵中的索引
        #         x_idx = list(remaining_indices).index(x)
        #         y_idx = list(remaining_indices).index(y)
        #         # 添加双向边
        #         g_marginalised[x_idx, y_idx] = 1
        #         g_marginalised[y_idx, x_idx] = 1
                
        return g_marginalised



    @staticmethod
    def _marginalise_node(graph: np.ndarray, node_idx: int, confounded_nodes: set) -> tuple[np.ndarray, set]:
        """
        边缘化一个节点，将其从图中移除，并适当调整边的关系
        
        参数:
            graph: 邻接矩阵表示的图
            node_idx: 要移除的节点索引
            confounded_nodes: 已知的混杂节点对集合
            
        返回:
            更新后的邻接矩阵和混杂节点对集合
        """
        n = graph.shape[0]
        # 创建图的副本
        new_graph = graph.copy()
        
        # 找出所有前驱节点（指向node_idx的节点）
        predecessors = [i for i in range(n) if i != node_idx and graph[i, node_idx] > 0]
        
        # 找出所有后继节点（node_idx指向的节点）
        successors = [i for i in range(n) if i != node_idx and graph[node_idx, i] > 0]
        
        # 为所有前驱节点添加到后继节点的边
        for pre in predecessors:
            for succ in successors:
                if pre != succ:  # 避免自环
                    new_graph[pre, succ] = 1  # 添加边
        
        # 处理混杂关系
        for suc_one in successors:
            for suc_two in successors:
                if suc_two != suc_one:
                    confounded_nodes.add((suc_one, suc_two))
                    confounded_nodes.add((suc_two, suc_one))
            
            # 处理间接混杂关系
            for x, _ in [(x, y) for (x, y) in confounded_nodes if y == node_idx]:
                confounded_nodes.add((x, suc_one))
                confounded_nodes.add((suc_one, x))
        
        # 创建一个新的邻接矩阵，不包含要移除的节点
        indices = [i for i in range(n) if i != node_idx]
        reduced_graph = new_graph[np.ix_(indices, indices)]
        
        # 更新混杂节点的索引（如果需要）
        # 这里假设混杂节点集合中存储的是节点的原始索引
        
        return reduced_graph, confounded_nodes
    
    def graphical_compatibility(self, joint_matrix: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算基于邻接矩阵的图形兼容性分数
        
        参数:
            joint_matrix: 联合图的邻接矩阵 (numpy数组或tensor)
            
        返回:
            平均结构汉明距离 (SHD)
        """
        # 确保输入是numpy数组
        joint_matrix = self._ensure_numpy(joint_matrix)
        
        if not hasattr(self, 'marginal_graphs') or not self.marginal_graphs or not hasattr(self, 'variables_list') or not self.variables_list:
            logging.warning("没有可用的边缘图或变量列表，无法计算兼容性分数")
            return 0.0
        
        shds = []
        for i, (marginal_matrix, subset) in enumerate(zip(self.marginal_graphs, self.variables_list)):
            marginal_matrix = self._ensure_numpy(marginal_matrix)
            
            # 边缘化联合图，只保留子集中的节点
            marginalized_joint = self.marginalize(subset, joint_matrix)
            
            # 计算SHD
            shd = self._calculate_shd(marginalized_joint, marginal_matrix)
            shds.append(shd)
            print(shds)
        return float(np.mean(shds)) if shds else 0.0
    
  
    
    def compatibility_score(self, data, joint_matrix: Union[np.ndarray, torch.Tensor], call="learn") -> float:
        """
        计算兼容性分数
        
        参数:
            data: 数据集，可以是DataFrame、numpy数组或列表
            joint_matrix: 联合图的邻接矩阵
            call: 算法对象中用于学习因果图的方法名，默认为"learn"
            
        返回:
            兼容性分数
        """
        try:
            
            # 确保联合矩阵是numpy数组
            joint_matrix = self._ensure_numpy(joint_matrix)
            
            # 确保数据是numpy数组
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            elif isinstance(data, list):
                data = np.array(data)
            
            # 清空之前的边缘图和变量列表
            self.marginal_graphs = []
            self.variables_list = []

            # 生成变量子集
            subset_list = self._draw_subsets(data.shape[1])
            tmp1,tmp2 = None,None
            if self.algo_params is not None:
                if "active_constraints" in self.algo_params:
                    tmp1 = self.algo_params['active_constraints'].copy()
                if "inactive_constraints" in self.algo_params:
                    tmp2 = self.algo_params['inactive_constraints'].copy()
                            
            # 对每个子集学习边缘图
            for subset in subset_list:
                algo = None
                if self.algo_name == "notearsmlp":
                    if self.algo_params is None:
                        algo = NotearsNonlinear(subset = subset)
                    else:
                      
                        if "true_dag" in self.algo_params:
                            
                             self.algo_params['true_dag'] = self.marginalize(subset,self.algo_params['true_dag'])
                        if "active_constraints" in self.algo_params:
                             self.algo_params['active_constraints'] = self.marginalize(subset,tmp1)
                        if "inactive_constraints" in self.algo_params:
                             self.algo_params['inactive_constraints'] = tmp2[subset][:,subset]
                        algo = NotearsNonlinear(subset=subset,**self.algo_params)
                # 获取算法对象中的学习方法
                learn_method = getattr(algo, call)
                if callable(learn_method):
                    # 使用子集数据学习边缘图
                    algo.subset = subset
                    learn_method(data[:, subset])
                    # 存储学习到的边缘图
                    self.marginal_graphs.append(algo.causal_matrix)
                    # 存储对应的变量子集
                    self.variables_list.append(subset)
                else:
                    raise ValueError(f"算法对象中不存在名为 '{call}' 的方法")
            
            # 计算图形兼容性
            return self.graphical_compatibility(joint_matrix)
        except Exception as e:
            logging.error(f"计算兼容性分数时发生错误: {e}")
            print(f"计算兼容性分数时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0