import numpy as np
import networkx as nx
from networkx.algorithms.dag import transitive_reduction
from networkx.exception import NetworkXNoCycle
import torch
from typing import Tuple, Optional, List, Tuple as TupleType

def get_transitive_reduction(G: nx.DiGraph) -> nx.DiGraph:
    # 得到传递规约
    # if not nx.is_directed_acyclic_graph(G):
    #     raise ValueError("transitive_reduction 需要 DAG")
    # TR = nx.DiGraph()
    # TR.add_nodes_from(G.nodes())
    # TC = nx.transitive_closure(G)
    # for u in G.nodes():
    #     succ_u = list(G.successors(u))
    #     for v in succ_u:
    #         has_alt = any((w != v) and TC.has_edge(w, v) for w in succ_u)
    #         if not has_alt:
    #             TR.add_edge(u, v)
    # return TR
    try:
        result = transitive_reduction(G)
        return result
    except Exception as e:
        print(e)
        nodes = list(G.nodes())
        A_ = nx.to_numpy_array(G, nodelist=nodes, dtype=int, weight=None)


        print("邻接矩阵：\n", A_)
        # print(G.adjacency)
        return None
    



def get_maximal_paths(G: nx.DiGraph):
    maximal_paths = []
    sources = [v for v in G.nodes() if G.in_degree(v) == 0]
    def dfs(u,path = [],visited = set()):
        path.append(u)
        visited.add(u)
        succ = list(G.successors(u))
        if not succ:
            maximal_paths.append(path.copy())
        else:
            for v in succ:
                if v not in visited:
                    dfs(v, path, visited)
        path.pop()
        visited.remove(u)
        
    for source in sources:
        dfs(source)
    return maximal_paths

def adjacency_to_digraph(adj_matrix: np.ndarray|torch.Tensor) -> nx.DiGraph:
    """将 0/1 邻接矩阵转换为 networkx.DiGraph。"""
    num_nodes = adj_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = np.transpose(np.nonzero(adj_matrix))
    for i, j in edges:
        G.add_edge(int(i), int(j))
    return G


def ensure_acyclic_digraph(G: nx.DiGraph) -> TupleType[nx.DiGraph, List[TupleType[int, int]]]:
    """返回移除所有有向环后的副本以及被移除的边列表。"""
    dag = G.copy()
    removed_edges: List[TupleType[int, int]] = []

    while True:
        try:
            cycle_edges = nx.find_cycle(dag, orientation="original")
        except NetworkXNoCycle:
            break

        # cycle_edges: list[(u, v, orientation)]
        u, v, _ = cycle_edges[-1]
        if dag.has_edge(u, v):
            dag.remove_edge(u, v)
            removed_edges.append((u, v))

    return dag, removed_edges


def list_to_adjacency(list_matrix: list) -> np.ndarray:
    pass




def get_active_constraints(G: nx.DiGraph) ->np.ndarray:
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G 需要是 DAG")
    active_constraints = nx.to_numpy_array(G, nodelist=list(G.nodes()), dtype=int)
    return active_constraints

def get_inactive_constraints(G: nx.DiGraph):
    active_constraints = get_active_constraints(G)
    return 1 - active_constraints


# Build candidate matrices (subset of true edges as active, subset of true non-edges as inactive)
def build_candidate_matrices(true_dag: np.ndarray,
                             active_ratio: float = 0.4,
                             inactive_ratio: float = 0.6,
                             rng: Optional[np.random.Generator] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    # Ensure binary matrix
    true_bin = (true_dag != 0).astype(int)

    # Sample subset of true edges for active constraints
    active_mask = (rng.random(true_bin.shape) <= active_ratio).astype(int)
    active_matrix = true_bin * active_mask

    # Sample subset of true non-edges for inactive constraints
    non_edge_mask = (true_bin == 0).astype(int)
    inactive_mask = (rng.random(true_bin.shape) <= inactive_ratio).astype(int)
    inactive_matrix = np.zeros_like(true_bin)
    pick_inactive = (non_edge_mask == 1) & (inactive_mask == 1)
    inactive_matrix[pick_inactive] = 1

    return active_matrix, inactive_matrix

       









if __name__ == "__main__":
    # 简单测试
    # 图：0 -> 1 -> 2，且 0 -> 2（冗余边）; 2 -> 3
    A = np.array([
    [0, 1, 1, 1, 0],  # 1 → 2, 1 → 4
    [0, 0, 1, 0, 0],  # 2 → 3
    [0, 0, 0, 0, 0],  # 3 终点
    [0, 0, 0, 0, 1],  # 4 → 5
    [0, 0, 0, 0, 0] 
    ], dtype=int)

    print("候选图 A:\n", A)
    A_g = adjacency_to_digraph(A)
    A_tr = get_transitive_reduction(A_g)

   
    print("\n传递约简 A_tr:\n", A_tr.edges)

    paths = get_maximal_paths(A_tr)
    print("\n最大路径集合:", paths)
    print("\n活跃约束:\n", 
    get_active_constraints(A_tr ))
    print("\n不活跃约束:\n", 
    get_inactive_constraints(A_g))
  