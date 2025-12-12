from copy import deepcopy
from itertools import combinations

import joblib
import numpy as np
from castle.common.independence_tests import CITest


def _loop(G, d):
    """
    Check if |adj(x, G)\{y}| < d for every pair of adjacency vertices in G

    Parameters
    ----------
    G: numpy.ndarray
        The undirected graph  G
    d: int
        depth of conditional vertices

    Returns
    -------
    out: bool
        if False, denote |adj(i, G)\{j}| < d for every pair of adjacency
        vertices in G, then finished loop.
    """

    assert G.shape[0] == G.shape[1]

    pairs = [(x, y) for x, y in combinations(set(range(G.shape[0])), 2)]
    less_d = 0
    for i, j in pairs:
        adj_i = set(np.argwhere(G[i] != 0).reshape(-1, ))
        z = adj_i - {j}  # adj(C, i)\{j}
        if len(z) < d:
            less_d += 1
        else:
            break
    if less_d == len(pairs):
        return False
    else:
        return True

def find_skeleton(data, alpha, ci_test, variant='original',
                  priori_knowledge=None, base_skeleton=None,
                  p_cores=1, s=None, batch=None):
    """Find skeleton graph from G using PC algorithm

    It learns a skeleton graph which contains only undirected edges
    from data.

    Parameters
    ----------
    data : array, (n_samples, n_features)
        Dataset with a set of variables V
    alpha : float, default 0.05
        significant level
    ci_test : str, callable
        ci_test method, if str, must be one of [`fisherz`, `g2`, `chi2`].
        if callable, must return a tuple that  the last element is `p_value` ,
        like (_, _, p_value) or (chi2, dof, p_value).
        See more: `castle.common.independence_tests.CITest`
    variant : str, default 'original'
        variant of PC algorithm, contains [`original`, `stable`, `parallel`].
        If variant == 'parallel', need to provide the flowing 3 parameters.
    base_skeleton : array, (n_features, n_features)
        prior matrix, must be undirected graph.
        The two conditionals `base_skeleton[i, j] == base_skeleton[j, i]`
        and `and base_skeleton[i, i] == 0` must be satisified which i != j.
    p_cores : int
        Number of CPU cores to be used
    s : bool, default False
        memory-efficient indicator
    batch : int
        number of edges per batch

    if s is None or False, or without batch, batch=|J|.
    |J| denote number of all pairs of adjacency vertices (X, Y) in G.

    Returns
    -------
    skeleton : array
        The undirected graph
    seq_set : dict
        Separation sets
        Such as key is (x, y), then value is a set of other variables
        not contains x and y.

    Examples
    --------
    >>> from castle.algorithms.pc.pc import find_skeleton
    >>> from castle.datasets import load_dataset

    >>> true_dag, X = load_dataset(name='iid_test')
    >>> skeleton, sep_set = find_skeleton(data, 0.05, 'fisherz')
    >>> print(skeleton)
    [[0. 0. 1. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 1. 1. 1. 1. 0. 1. 0.]
     [1. 0. 0. 0. 1. 0. 0. 1. 0. 0.]
     [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
     [0. 1. 1. 1. 0. 0. 0. 0. 0. 1.]
     [1. 1. 0. 0. 0. 0. 0. 1. 1. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 0. 1. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 1. 0. 0. 0. 1.]
     [0. 0. 0. 1. 1. 1. 0. 1. 1. 0.]]
    """

    def test(x, y):

        K_x_y = 1
        sub_z = None
        # On X's neighbours
        adj_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
        z_x = adj_x - {y}  # adj(X, G)\{Y}
        if len(z_x) >= d:
            # |adj(X, G)\{Y}| >= d
            for sub_z in combinations(z_x, d):
                sub_z = list(sub_z)
                _, _, p_value = ci_test(data, x, y, sub_z)
                if p_value >= alpha:
                    K_x_y = 0
                    # sep_set[(x, y)] = sub_z
                    break
            if K_x_y == 0:
                return K_x_y, sub_z

        return K_x_y, sub_z

    def parallel_cell(x, y):

        # On X's neighbours
        K_x_y, sub_z = test(x, y)
        if K_x_y == 1:
            # On Y's neighbours
            K_x_y, sub_z = test(y, x)

        return (x, y), K_x_y, sub_z

    if ci_test == 'fisherz':
        ci_test = CITest.fisherz_test
    elif ci_test == 'g2':
        ci_test = CITest.g2_test
    elif ci_test == 'chi2':
        ci_test = CITest.chi2_test
    elif callable(ci_test):
        ci_test = ci_test
    else:
        raise ValueError(f'The type of param `ci_test` expect callable,'
                         f'but got {type(ci_test)}.')

    n_feature = data.shape[1]
    if base_skeleton is None:
        skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
    else:
        row, col = np.diag_indices_from(base_skeleton)
        base_skeleton[row, col] = 0
        skeleton = base_skeleton
    nodes = set(range(n_feature))

    # update skeleton based on priori knowledge
    for i, j in combinations(nodes, 2):
        if priori_knowledge is not None and (
                priori_knowledge.is_forbidden(i, j)
                and priori_knowledge.is_forbidden(j, i)):
            skeleton[i, j] = skeleton[j, i] = 0

    sep_set = {}
    d = -1
    while _loop(skeleton, d):  # until for each adj(C,i)\{j} < l
        d += 1
        if variant == 'stable':
            C = deepcopy(skeleton)
        else:
            C = skeleton
        if variant != 'parallel':
            for i, j in combinations(nodes, 2):
                if skeleton[i, j] == 0:
                    continue
                adj_i = set(np.argwhere(C[i] == 1).reshape(-1, ))
                z = adj_i - {j}  # adj(C, i)\{j}
                if len(z) >= d:
                    # |adj(C, i)\{j}| >= l
                    for sub_z in combinations(z, d):
                        sub_z = list(sub_z)
                        _, _, p_value = ci_test(data, i, j, sub_z)
                        if p_value >= alpha:
                            skeleton[i, j] = skeleton[j, i] = 0
                            sep_set[(i, j)] = sub_z
                            break
        else:
            J = [(x, y) for x, y in combinations(nodes, 2)
                 if skeleton[x, y] == 1]
            if not s or not batch:
                batch = len(J)
            if batch < 1:
                batch = 1
            if not p_cores or p_cores == 0:
                raise ValueError(f'If variant is parallel, type of p_cores '
                                 f'must be int, but got {type(p_cores)}.')
            for i in range(int(np.ceil(len(J) / batch))):
                each_batch = J[batch * i: batch * (i + 1)]
                parallel_result = joblib.Parallel(n_jobs=p_cores,
                                                  max_nbytes=None)(
                    joblib.delayed(parallel_cell)(x, y) for x, y in
                    each_batch
                )
                # Synchronisation Step
                for (x, y), K_x_y, sub_z in parallel_result:
                    if K_x_y == 0:
                        skeleton[x, y] = skeleton[y, x] = 0
                        sep_set[(x, y)] = sub_z

    return skeleton, sep_set




def pro