"""
带单调性约束的因果数据生成

生成过程：
1. 生成 DAG 结构
2. 为部分边指定单调性（增或减）
3. 使用单调非线性函数生成数据（简单函数或多层MLP）
4. 提取单调边矩阵作为先验约束

支持两种生成模式：
- 'simple': 使用预定义的单调函数（快速）
- 'mlp': 使用多层神经网络（更复杂、更真实）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import networkx as nx

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. MLP mode disabled.")


class MonotonicMLP(nn.Module):
    """
    单调性神经网络
    
    通过约束权重符号来确保单调性
    """
    
    def __init__(self, 
                 hidden_dims: List[int] = [10, 5],
                 monotonic_type: str = 'increasing',
                 weight_scale: float = 1.0):
        """
        Parameters
        ----------
        hidden_dims : list
            隐藏层维度列表
        monotonic_type : str
            单调性类型
            - 'increasing': 单调递增
            - 'decreasing': 单调递减
            - 'non_monotonic': 非单调
        weight_scale : float
            输出缩放因子
        """
        super(MonotonicMLP, self).__init__()
        
        self.monotonic_type = monotonic_type
        self.weight_scale = weight_scale
        
        # 构建网络层
        layers = []
        in_dim = 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.layers = nn.ModuleList(layers)
        
        # 初始化权重以确保单调性
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for layer in self.layers:
            if self.monotonic_type == 'increasing':
                # 递增：所有权重为正
                nn.init.uniform_(layer.weight, 0.1, 1.0)
            elif self.monotonic_type == 'decreasing':
                # 递减：所有权重为负
                nn.init.uniform_(layer.weight, -1.0, -0.1)
            else:  # non_monotonic
                # 非单调：权重可正可负
                nn.init.normal_(layer.weight, 0, 0.5)
            
            # 偏置项随机初始化
            nn.init.uniform_(layer.bias, -0.1, 0.1)
    
    def forward(self, x):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor, shape (n, 1)
            输入
        
        Returns
        -------
        y : torch.Tensor, shape (n, 1)
            输出
        """
        for i, layer in enumerate(self.layers):
            if self.monotonic_type in ['increasing', 'decreasing']:
                # 单调模式：强制权重符号
                if self.monotonic_type == 'increasing':
                    # 递增：权重取绝对值
                    weight = torch.abs(layer.weight)
                else:  # decreasing
                    # 递减：权重取负绝对值
                    weight = -torch.abs(layer.weight)
                
                x = torch.nn.functional.linear(x, weight, layer.bias)
            else:
                # 非单调模式：正常权重
                x = layer(x)
            
            # 激活函数（最后一层除外）
            if i < len(self.layers) - 1:
                if self.monotonic_type in ['increasing', 'decreasing']:
                    # 单调模式：使用单调激活函数
                    x = torch.nn.functional.softplus(x)  # 单调递增
                else:
                    # 非单调模式：使用 tanh
                    x = torch.tanh(x)
        
        # 缩放输出
        return x * self.weight_scale


class MonotonicDAGGenerator:
    """
    生成带单调性约束的 DAG 和数据
    """
    
    def __init__(self, 
                 n_nodes: int,
                 n_edges: int,
                 monotonic_ratio: float = 0.6,
                 seed: int = None,
                 generation_mode: str = 'simple'):
        """
        Parameters
        ----------
        n_nodes : int
            节点数量
        n_edges : int
            边数量
        monotonic_ratio : float
            单调边的比例 (0-1)，其余为非单调
        seed : int
            随机种子
        generation_mode : str
            数据生成模式
            - 'simple': 使用预定义函数（快速）
            - 'mlp': 使用多层神经网络（更复杂、需要PyTorch）
        """
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.monotonic_ratio = monotonic_ratio
        self.generation_mode = generation_mode
        
        if seed is not None:
            np.random.seed(seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(seed)
        
        # 检查 MLP 模式是否可用
        if generation_mode == 'mlp' and not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Falling back to 'simple' mode.")
            self.generation_mode = 'simple'
        
        # 生成 DAG 结构
        self.W, self.G = self._generate_dag()
        
        # 标记单调边
        self.monotonic_matrix = self._mark_monotonic_edges()
        
        # 生成函数（简单函数或 MLP）
        if self.generation_mode == 'mlp':
            self.edge_mlps = self._create_mlps()
        else:
            self.edge_functions = self._assign_functions()
        
    def _generate_dag(self) -> Tuple[np.ndarray, nx.DiGraph]:
        """生成随机 DAG"""
        # 使用 Erdos-Renyi 方法
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes))
        
        # 随机添加边，确保 DAG 性质
        edges_added = 0
        max_attempts = self.n_edges * 10
        attempts = 0
        
        while edges_added < self.n_edges and attempts < max_attempts:
            i = np.random.randint(0, self.n_nodes)
            j = np.random.randint(0, self.n_nodes)
            
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j)
                # 检查是否产生环
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(i, j)
                else:
                    edges_added += 1
            
            attempts += 1
        
        # 转为邻接矩阵
        W = nx.to_numpy_array(G)
        
        # 随机赋予权重 (0.5, 2.0)
        W[W > 0] = np.random.uniform(0.5, 2.0, size=(W > 0).sum())
        
        return W, G
    
    def _mark_monotonic_edges(self) -> np.ndarray:
        """
        标记单调边
        
        Returns
        -------
        monotonic_matrix : np.ndarray
            单调性矩阵
            0: 非边
            1: 单调递增边
            -1: 单调递减边
            2: 非单调边（复杂非线性）
        """
        monotonic_matrix = np.zeros_like(self.W)
        
        # 找出所有边
        edges = np.argwhere(self.W != 0)
        n_edges = len(edges)
        
        # 确定单调边数量
        n_monotonic = int(n_edges * self.monotonic_ratio)
        
        # 随机选择单调边
        monotonic_indices = np.random.choice(n_edges, n_monotonic, replace=False)
        
        for idx in range(n_edges):
            i, j = edges[idx]
            if idx in monotonic_indices:
                # 单调边：随机选择递增(1)或递减(-1)
                monotonic_matrix[i, j] = np.random.choice([1, -1])
            else:
                # 非单调边
                monotonic_matrix[i, j] = 2
        
        return monotonic_matrix
    
    def _assign_functions(self) -> Dict:
        """
        为每条边分配具体的函数
        
        Returns
        -------
        edge_functions : dict
            键: (i, j) 边
            值: {'type': str, 'params': dict}
        """
        functions = {}
        
        edges = np.argwhere(self.W != 0)
        
        for i, j in edges:
            mono_type = self.monotonic_matrix[i, j]
            weight = self.W[i, j]
            
            if mono_type == 1:  # 单调递增
                # 可选函数：线性、sigmoid、log、sqrt
                func_type = np.random.choice([
                    'linear_pos',    # w * x
                    'sigmoid',       # w * sigmoid(x)
                    'log',           # w * log(1 + |x|) * sign(x)
                    'sqrt',          # w * sqrt(|x|) * sign(x)
                    'power'          # w * x^p (p < 1)
                ])
                
                if func_type == 'linear_pos':
                    functions[(i, j)] = {
                        'type': 'linear_pos',
                        'weight': weight,
                        'params': {}
                    }
                elif func_type == 'sigmoid':
                    functions[(i, j)] = {
                        'type': 'sigmoid',
                        'weight': weight,
                        'params': {'scale': np.random.uniform(0.5, 2.0)}
                    }
                elif func_type == 'log':
                    functions[(i, j)] = {
                        'type': 'log',
                        'weight': weight,
                        'params': {}
                    }
                elif func_type == 'sqrt':
                    functions[(i, j)] = {
                        'type': 'sqrt',
                        'weight': weight,
                        'params': {}
                    }
                else:  # power
                    functions[(i, j)] = {
                        'type': 'power',
                        'weight': weight,
                        'params': {'p': np.random.uniform(0.3, 0.8)}
                    }
            
            elif mono_type == -1:  # 单调递减
                # 可选函数：负线性、负指数
                func_type = np.random.choice([
                    'linear_neg',    # -w * x
                    'exp_neg'        # -w * (1 - exp(-|x|)) * sign(x)
                ])
                
                if func_type == 'linear_neg':
                    functions[(i, j)] = {
                        'type': 'linear_neg',
                        'weight': weight,
                        'params': {}
                    }
                else:  # exp_neg
                    functions[(i, j)] = {
                        'type': 'exp_neg',
                        'weight': weight,
                        'params': {}
                    }
            
            else:  # mono_type == 2, 非单调
                # 可选函数：二次、sin、tanh
                func_type = np.random.choice([
                    'quadratic',     # w * x^2
                    'sin',           # w * sin(x)
                    'tanh_flip'      # w * (x - 0.5*tanh(x))
                ])
                
                if func_type == 'quadratic':
                    functions[(i, j)] = {
                        'type': 'quadratic',
                        'weight': weight,
                        'params': {}
                    }
                elif func_type == 'sin':
                    functions[(i, j)] = {
                        'type': 'sin',
                        'weight': weight,
                        'params': {'freq': np.random.uniform(0.5, 1.5)}
                    }
                else:  # tanh_flip
                    functions[(i, j)] = {
                        'type': 'tanh_flip',
                        'weight': weight,
                        'params': {}
                    }
        
        return functions
    
    def _apply_function(self, x: float, func_info: dict) -> float:
        """应用指定的函数"""
        func_type = func_info['type']
        weight = func_info['weight']
        params = func_info['params']
        
        if func_type == 'linear_pos':
            return weight * x
        
        elif func_type == 'linear_neg':
            return -weight * x
        
        elif func_type == 'sigmoid':
            scale = params['scale']
            return weight * (1 / (1 + np.exp(-scale * x)) - 0.5) * 2
        
        elif func_type == 'log':
            return weight * np.sign(x) * np.log(1 + np.abs(x))
        
        elif func_type == 'sqrt':
            return weight * np.sign(x) * np.sqrt(np.abs(x))
        
        elif func_type == 'power':
            p = params['p']
            return weight * np.sign(x) * (np.abs(x) ** p)
        
        elif func_type == 'exp_neg':
            return -weight * np.sign(x) * (1 - np.exp(-np.abs(x)))
        
        elif func_type == 'quadratic':
            return weight * x * x
        
        elif func_type == 'sin':
            freq = params['freq']
            return weight * np.sin(freq * x)
        
        elif func_type == 'tanh_flip':
            return weight * (x - 0.5 * np.tanh(x))
        
        else:
            return weight * x
    
    def _create_mlps(self) -> Dict:
        """
        为每条边创建一个小的 MLP
        
        Returns
        -------
        edge_mlps : dict
            键: (i, j) 边
            值: MonotonicMLP 对象
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot create MLPs.")
        
        mlps = {}
        edges = np.argwhere(self.W != 0)
        
        for i, j in edges:
            mono_type = self.monotonic_matrix[i, j]
            weight = self.W[i, j]
            
            # 创建 MLP
            if mono_type == 1:  # 单调递增
                mlps[(i, j)] = MonotonicMLP(
                    hidden_dims=[10, 5],
                    monotonic_type='increasing',
                    weight_scale=weight
                )
            elif mono_type == -1:  # 单调递减
                mlps[(i, j)] = MonotonicMLP(
                    hidden_dims=[10, 5],
                    monotonic_type='decreasing',
                    weight_scale=weight
                )
            else:  # 非单调
                mlps[(i, j)] = MonotonicMLP(
                    hidden_dims=[10, 5],
                    monotonic_type='non_monotonic',
                    weight_scale=weight
                )
        
        return mlps
    
    def _apply_mlp(self, x: np.ndarray, mlp) -> np.ndarray:
        """
        应用 MLP
        
        Parameters
        ----------
        x : np.ndarray
            输入数据 (n_samples,)
        mlp : MonotonicMLP
            MLP 模型
        
        Returns
        -------
        y : np.ndarray
            输出数据 (n_samples,)
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            y_tensor = mlp(x_tensor)
        return y_tensor.numpy().reshape(-1)
    
    def generate_data(self, n_samples: int, noise_scale: float = 1.0) -> np.ndarray:
        """
        生成数据
        
        Parameters
        ----------
        n_samples : int
            样本数量
        noise_scale : float
            噪声标准差
        
        Returns
        -------
        X : np.ndarray, shape (n_samples, n_nodes)
            生成的数据
        """
        X = np.zeros((n_samples, self.n_nodes))
        
        # 获取拓扑排序
        topo_order = list(nx.topological_sort(self.G))
        
        # 按拓扑顺序生成数据
        for node in topo_order:
            # 找到父节点
            parents = list(self.G.predecessors(node))
            
            if len(parents) == 0:
                # 源节点：从标准正态分布采样
                X[:, node] = np.random.randn(n_samples) * 2
            else:
                # 根据父节点和函数/MLP生成
                contribution = np.zeros(n_samples)
                
                for parent in parents:
                    if self.generation_mode == 'mlp':
                        # 使用 MLP
                        mlp = self.edge_mlps[(parent, node)]
                        contribution += self._apply_mlp(X[:, parent], mlp)
                    else:
                        # 使用简单函数
                        func_info = self.edge_functions[(parent, node)]
                        # 对每个样本应用函数
                        for i in range(n_samples):
                            contribution[i] += self._apply_function(X[i, parent], func_info)
                
                # 添加噪声
                noise = np.random.randn(n_samples) * noise_scale
                X[:, node] = contribution + noise
        
        return X
    
    def get_monotonic_edges_matrix(self, only_monotonic: bool = True) -> np.ndarray:
        """
        获取单调边矩阵（用作先验约束）
        
        Parameters
        ----------
        only_monotonic : bool
            True: 只返回单调边 (值为1或-1的边)
            False: 返回所有边的单调性信息
        
        Returns
        -------
        constraint_matrix : np.ndarray
            约束矩阵 (0: 无边, 1: 有边)
        """
        if only_monotonic:
            # 只保留单调边
            return (np.abs(self.monotonic_matrix) == 1).astype(float)
        else:
            # 返回所有边
            return (self.monotonic_matrix != 0).astype(float)
    
    def sample_monotonic_edges(self, sample_ratio: float = 0.5) -> np.ndarray:
        """
        从单调边中采样部分作为先验
        
        Parameters
        ----------
        sample_ratio : float
            采样比例 (0-1)
        
        Returns
        -------
        sampled_matrix : np.ndarray
            采样后的约束矩阵
        """
        monotonic_mask = (np.abs(self.monotonic_matrix) == 1)
        monotonic_edges = np.argwhere(monotonic_mask)
        
        n_monotonic = len(monotonic_edges)
        n_sample = int(n_monotonic * sample_ratio)
        
        # 随机采样
        sampled_indices = np.random.choice(n_monotonic, n_sample, replace=False)
        sampled_edges = monotonic_edges[sampled_indices]
        
        # 构建约束矩阵
        sampled_matrix = np.zeros_like(self.W)
        for i, j in sampled_edges:
            sampled_matrix[i, j] = 1
        
        return sampled_matrix
    
    def visualize(self, save_path: str = None):
        """可视化 DAG 和单调性"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 原始 DAG
        ax1 = axes[0]
        ax1.imshow(self.W != 0, cmap='Blues', interpolation='nearest')
        ax1.set_title('DAG Structure', fontsize=14, fontweight='bold')
        ax1.set_xlabel('To Node')
        ax1.set_ylabel('From Node')
        
        # 2. 单调性矩阵
        ax2 = axes[1]
        mono_visual = self.monotonic_matrix.copy()
        cmap = plt.cm.RdBu_r
        im = ax2.imshow(mono_visual, cmap=cmap, vmin=-2, vmax=2, interpolation='nearest')
        ax2.set_title('Monotonicity Matrix\n(1=increase, -1=decrease, 2=non-mono)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('To Node')
        ax2.set_ylabel('From Node')
        plt.colorbar(im, ax=ax2)
        
        # 3. 单调边约束（可用作先验）
        ax3 = axes[2]
        constraint = self.get_monotonic_edges_matrix(only_monotonic=True)
        ax3.imshow(constraint, cmap='Greens', interpolation='nearest')
        ax3.set_title('Monotonic Edges Constraint\n(Available as Prior)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('To Node')
        ax3.set_ylabel('From Node')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """打印摘要信息"""
        print("=" * 60)
        print("单调性 DAG 生成摘要")
        print("=" * 60)
        print(f"节点数: {self.n_nodes}")
        print(f"边数: {self.n_edges}")
        
        n_mono_inc = (self.monotonic_matrix == 1).sum()
        n_mono_dec = (self.monotonic_matrix == -1).sum()
        n_non_mono = (self.monotonic_matrix == 2).sum()
        
        print(f"\n单调性分布:")
        print(f"  递增边: {n_mono_inc} ({n_mono_inc/self.n_edges*100:.1f}%)")
        print(f"  递减边: {n_mono_dec} ({n_mono_dec/self.n_edges*100:.1f}%)")
        print(f"  非单调边: {n_non_mono} ({n_non_mono/self.n_edges*100:.1f}%)")
        print(f"  总单调边: {n_mono_inc + n_mono_dec} ({(n_mono_inc+n_mono_dec)/self.n_edges*100:.1f}%)")
        
        print(f"\n函数类型统计:")
        func_counts = {}
        for func_info in self.edge_functions.values():
            func_type = func_info['type']
            func_counts[func_type] = func_counts.get(func_type, 0) + 1
        
        for func_type, count in sorted(func_counts.items()):
            print(f"  {func_type}: {count}")
        
        print("=" * 60)


# 示例用法
if __name__ == "__main__":
    import os
    
    # 创建输出目录
    os.makedirs('algorithm/CIR/exp/monotonic', exist_ok=True)
    
    # 生成单调性 DAG
    print("生成带单调性约束的 DAG...")
    gen = MonotonicDAGGenerator(
        n_nodes=20,
        n_edges=40,
        monotonic_ratio=0.6,  # 60% 的边是单调的
        seed=42
    )
    
    # 打印摘要
    gen.print_summary()
    
    # 生成数据
    print("\n生成数据...")
    X = gen.generate_data(n_samples=1000, noise_scale=1.0)
    print(f"数据形状: {X.shape}")
    print(f"数据统计: mean={X.mean():.2f}, std={X.std():.2f}")
    
    # 获取约束矩阵
    print("\n提取单调边约束...")
    constraint_full = gen.get_monotonic_edges_matrix(only_monotonic=True)
    print(f"完整单调边数量: {constraint_full.sum():.0f}")
    
    # 采样部分单调边作为先验
    constraint_sampled = gen.sample_monotonic_edges(sample_ratio=0.5)
    print(f"采样单调边数量 (50%): {constraint_sampled.sum():.0f}")
    
    # 可视化
    print("\n生成可视化...")
    gen.visualize(save_path='algorithm/CIR/exp/monotonic/dag_monotonic.png')
    
    # 保存数据和约束
    np.save('algorithm/CIR/exp/monotonic/X_data.npy', X)
    np.save('algorithm/CIR/exp/monotonic/true_dag.npy', gen.W)
    np.save('algorithm/CIR/exp/monotonic/monotonic_matrix.npy', gen.monotonic_matrix)
    np.save('algorithm/CIR/exp/monotonic/constraint_full.npy', constraint_full)
    np.save('algorithm/CIR/exp/monotonic/constraint_sampled.npy', constraint_sampled)
    
    print("\n数据已保存到 algorithm/CIR/exp/monotonic/")
    print("完成!")

