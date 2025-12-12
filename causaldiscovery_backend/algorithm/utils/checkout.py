import torch
import numpy as np
from collections import deque
def _to_numpy(W):
    """
    将输入 W 统一转成 numpy.ndarray。
    - 如果本身是 np.ndarray：直接返回
    - 如果是 torch.Tensor：detach + cpu + numpy
    - 否则：报错
    """
    if isinstance(W, np.ndarray):
        return W
    if torch is not None and isinstance(W, torch.Tensor):
        return W.detach().cpu().numpy()
    raise TypeError(
        f"W must be a numpy.ndarray or torch.Tensor, got {type(W)}"
    )


def is_acyclic(W):
    W = _to_numpy(W)
    d = W.shape[0]
    indeg = W.sum(axis = 0).astype(int)
    queue = deque([i    for i in range(d) if indeg[i] == 0 ] )
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1

        for i in range(d):
            if W[u,i] != 0:
                indeg[i] -= 1
                if indeg[i] == 0:
                    queue.append(i)
    return  visited == d
       







def get_paths(u,v,W):
    W = _to_numpy(W)
    visited = set([u])
    res = []
    d = W.shape[0]
    
    def dfs(path,pre):
        if pre == v:
            res.append(path[:])
            return 
        for i in range(d):
            if i not in visited:
                if W[pre][i] != 0:
                    visited.add(i)
                    path.append(i)
                    dfs(path,i)
                    path.pop()
                    visited.remove(i)
    dfs([u],u)
    return res


def has_more_paths(u,v,W):
    res = get_paths(u,v,W)

    return len(res)



