# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tools to learn a ``StructureModel`` which describes the conditional dependencies between variables in a dataset.
"""
import os
import sys
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import inspect
import traceback

from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms.gradient.notears.torch.nonlinear import (
    NotearsNonlinear,
)
from algorithm.utils.constraints import ActiveConstraints, InactiveConstraints

import logging
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from castle.algorithms import PC, CORL, DAG_GNN, GAE, GES, RL, GraNDAG, MCSL, anm, ICALiNGAM, ANMNonlinear, DirectLiNGAM
from causaldiscovery_backend.castle.common.priori_knowledge import PrioriKnowledge

from sklearn.utils import check_array


from causalnex.structure.pytorch.dist_type import DistTypeContinuous, dist_type_aliases
from causalnex.structure.structuremodel import StructureModel

__all__ = ["from_numpy", "from_pandas"]

from torch.cuda import device


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments



def from_numpy(
    X: np.ndarray,
use_model,
    dist_type_schema: Dict[int, str] = None,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    w_threshold: float = None,
    max_iter: int = 100,
    candidate_active: np.ndarray = None,
    candidate_inactive: np.ndarray = None,
    constraint_config: Dict = None,
    tabu_edges: List[Tuple[int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    use_gpu: bool = True,

    **kwargs,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure with lasso regularisation
    describing conditional dependencies between variables in data presented as a numpy array.

    Based on DAGs with NO TEARS.
    @inproceedings{zheng2018dags,
        author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
        booktitle = {Advances in Neural Information Processing Systems},
        title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
        year = {2018},
        codebase = {https://github.com/xunzheng/notears}
    }

    Args:
        X: 2d input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.
        dist_type_schema: The dist type schema corresponding to the passed in data X.
        It maps the positional column in X to the string alias of a dist type.
        A list of alias names can be found in ``dist_type/__init__.py``.
        If None, assumes that all data in X is continuous.

        lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
        NOTE when using nonlinearities, the l1 loss only applies to the dag_layer.

        use_bias: Whether to fit a bias parameter in the NOTEARS algorithm.

        ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
        When using nonlinear layers use of this parameter is recommended.

        hidden_layer_units: An iterable where its length determine the number of layers used,
        and the numbers determine the number of nodes used for the layer in order.

        w_threshold: fixed threshold for absolute edge weights.

        max_iter: max number of dual ascent steps during optimisation.

        tabu_edges: list of edges(from, to) not to be included in the graph.

        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.

        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

        use_gpu: use gpu if it is set to True and CUDA is available.

        **kwargs: additional arguments for NOTEARS MLP model

    Returns:
        StructureModel: a graph of conditional dependencies between data variables.

    Raises:
        ValueError: If schema does not correspond to columns.
    """
    # n examples, d properties
    if not X.size:
        raise ValueError("Input data X is empty, cannot learn any structure")
    logging.info("Learning structure using 'NOTEARS' optimisation.")

    # Check array for NaN or inf values
    check_array(X)

    if dist_type_schema is not None:

        # make sure that there is one provided key per column
        if set(range(X.shape[1])).symmetric_difference(set(dist_type_schema.keys())):
            raise ValueError(
                f"Difference indices and expected indices. Got {dist_type_schema} schema"
            )

    # if dist_type_schema is None, assume all columns are continuous, else init the alias mapped object
    dist_types = (
        [DistTypeContinuous(idx=idx) for idx in np.arange(X.shape[1])]
        if dist_type_schema is None
        else [
            dist_type_aliases[alias](idx=idx) for idx, alias in dist_type_schema.items()
        ]
    )

    # shape of X before preprocessing
    _, d_orig = X.shape
    # perform dist type pre-processing (i.e. column expansion)
    for dist_type in dist_types:
        # NOTE: preprocess_X must be called first to perform possible column expansions
        X = dist_type.preprocess_X(X)
        tabu_edges = dist_type.preprocess_tabu_edges(tabu_edges)
        tabu_parent_nodes = dist_type.preprocess_tabu_nodes(tabu_parent_nodes)
        tabu_child_nodes = dist_type.preprocess_tabu_nodes(tabu_child_nodes)
    # shape of X after preprocessing
    _, d = X.shape

    # if None or empty, convert into a list with single item
    if hidden_layer_units is None:
        hidden_layer_units = [0]
    elif isinstance(hidden_layer_units, list) and not hidden_layer_units:
        hidden_layer_units = [0]

    # if no hidden layer units, still take 1 iteration step with bounds


    # Flip i and j because Pytorch flattens the vector in another direction


    # Map priors: explicit candidates override tabu-derived masks
    candidate_active = None if candidate_active is None else np.asarray(candidate_active)
    candidate_inactive = None if candidate_inactive is None else np.asarray(candidate_inactive)

    if candidate_inactive is None and (tabu_edges or tabu_parent_nodes or tabu_child_nodes):
        inactive_mask = np.zeros((d, d), dtype=np.float32)
        if tabu_edges is not None:
            for fro, to in tabu_edges:
                inactive_mask[fro, to] = 1.0
        if tabu_parent_nodes is not None:
            for parent in tabu_parent_nodes:
                inactive_mask[parent, :] = 1.0
        if tabu_child_nodes is not None:
            for child in tabu_child_nodes:
                inactive_mask[:, child] = 1.0
        candidate_inactive = inactive_mask

    for cand, name in ((candidate_active, "candidate_active"), (candidate_inactive, "candidate_inactive")):
        if cand is not None and cand.shape != (d, d):
            raise ValueError(f"{name} must have shape ({d}, {d}) after preprocessing, got {cand.shape}")


    sm = None
    try:
        if use_model == "notears-mlp":
            sm = notears_mlp(X=X, d=d, dist_types=dist_types,
                             hidden_layer_units=hidden_layer_units, lasso_beta=lasso_beta,
                             ridge_beta=ridge_beta, use_bias=use_bias, use_gpu=use_gpu,
                             kwargs=kwargs, max_iter=max_iter, w_threshold= w_threshold,
                             d_orig=d_orig,
                             candidate_active=candidate_active,
                             candidate_inactive=candidate_inactive,
                             constraint_config=constraint_config)
        elif use_model == "pc":

            sm = pc(X,d=d,dist_types=dist_types,tabu_parent_nodes=tabu_parent_nodes,tabu_child_nodes=tabu_child_nodes,tabu_edges=tabu_edges,d_orig=d_orig)
        elif use_model == "mcsl":
            sm = mcsl(X,d=d,dist_types=dist_types,tabu_parent_nodes=tabu_parent_nodes,tabu_child_nodes=tabu_child_nodes,tabu_edges=tabu_edges,d_orig=d_orig)
        elif use_model == "dag_gnn":
            sm = dag_gnn(X=X, d=d, dist_types=dist_types,
                             hidden_layer_units=hidden_layer_units, lasso_beta=lasso_beta,
                             ridge_beta=ridge_beta, use_bias=use_bias, use_gpu=use_gpu,
                             kwargs=kwargs, max_iter=max_iter, w_threshold= w_threshold,
                             d_orig=d_orig, tabu_parent_nodes=tabu_parent_nodes, tabu_child_nodes=tabu_child_nodes,
                             tabu_edges=tabu_edges,device_type="cpu")
        elif use_model == "ges":
            sm = ges(X,d=d,dist_types=dist_types,tabu_parent_nodes=tabu_parent_nodes,tabu_child_nodes=tabu_child_nodes,tabu_edges=tabu_edges,d_orig=d_orig)




        elif use_model == "directlingam":
            sm = directLiNGAM(X, d=d, dist_types=dist_types, tabu_parent_nodes=tabu_parent_nodes, tabu_child_nodes=tabu_child_nodes,
                     tabu_edges=tabu_edges, d_orig=d_orig)



        elif use_model == "grandag":
            sm = graNDAG(X=X, d=d, dist_types=dist_types,
                             hidden_layer_units=hidden_layer_units, lasso_beta=lasso_beta,
                             ridge_beta=ridge_beta, use_bias=use_bias, use_gpu=use_gpu,
                             kwargs=kwargs, max_iter=max_iter, w_threshold= w_threshold,
                             d_orig=d_orig, tabu_parent_nodes=tabu_parent_nodes, tabu_child_nodes=tabu_child_nodes,
                             tabu_edges=tabu_edges,device_type="cpu")

        else:
            raise ValueError(
               "没有该模型"
            )

        return sm
    except Exception as e:
        traceback.print_exc()
        return sm
    finally:
        return sm




def directLiNGAM(X,d,d_orig,dist_types,tabu_parent_nodes =None,tabu_child_nodes=None,tabu_edges = None):
    prior_knowledge = np.ones(shape = (d,d))
    prior_knowledge = -1 * prior_knowledge
    if tabu_parent_nodes is not None:
        for parent in tabu_parent_nodes:
            for i in range(d):
                prior_knowledge[i,parent] = 0

    if tabu_child_nodes is not None:
        for child in tabu_child_nodes:
            for i in range(d):
                prior_knowledge[child,i] = 0
    if tabu_edges is not None:
        for fro,to in tabu_edges:
            prior_knowledge[to,fro] = 0



    # pc = PC( priori_knowledge = priorKnowledge)
    # pc.learn(X)
    #
    #
    # sm = StructureModel(pc.causal_matrix)

    direct = DirectLiNGAM(prior_knowledge=prior_knowledge)
    #direct = DirectLiNGAM()
    direct.learn(X)
    sm = StructureModel(direct.causal_matrix)
    mean_effect = np.zeros(shape=(d,d))
    for i in range(d):
        for j in range(d):
            mean_effect[i,j] = direct.estimate_total_effect(X,i,j)


    mean_effect = torch.zeros((d, d))
    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=float(mean_effect[u, v]),
        )

    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    for node in sm.nodes(data=True):
        node[1]["bias"] = 0
    adj = deepcopy(direct.causal_matrix)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

    return sm


def pc(X,d,d_orig,dist_types,tabu_parent_nodes =None,tabu_child_nodes=None,tabu_edges = None):
    forbidden_list = []
    if tabu_parent_nodes is not None:
        for parent in tabu_parent_nodes:
            for i in range(d):
                forbidden_list.append((parent,i))

    if tabu_child_nodes is not None:
        for child in tabu_child_nodes:
            for i in range(d):
                forbidden_list.append((i,child))
    if tabu_edges is not None:
        forbidden_list.extend(tabu_edges)

    priorKnowledge = PrioriKnowledge(d)
    priorKnowledge.add_forbidden_edges(forbidden_list)

    pc = PC( priori_knowledge = priorKnowledge)
    pc.learn(X)


    sm = StructureModel(pc.causal_matrix)

    mean_effect = torch.zeros((d,d))
    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=float(mean_effect[u, v]),
        )

    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    for node in sm.nodes(data=True):
       node[1]["bias"] = 0
    adj = deepcopy(pc.causal_matrix)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

    return sm

def ges(X,d,d_orig,dist_types,tabu_parent_nodes =None,tabu_child_nodes=None,tabu_edges = None):
    algo = GES()
    algo.learn(X)
    sm = StructureModel(algo.causal_matrix)

    mean_effect = torch.zeros((d, d))
    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=float(mean_effect[u, v]),
        )

    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    for node in sm.nodes(data=True):
        node[1]["bias"] = 0
    adj = deepcopy(algo.causal_matrix)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])
    return sm

def graNDAG(X, d, dist_types, hidden_layer_units, lasso_beta, ridge_beta, use_bias, use_gpu, kwargs, max_iter,
            w_threshold, d_orig, tabu_parent_nodes, tabu_child_nodes, tabu_edges,device_type="cpu"):
    true_dag = [[1 for i in range(d)] for j in range(d)]
    if tabu_edges is not None:
        for i, j in tabu_edges:
            true_dag[i][j] = 0

    # if tabu_parent_nodes is not None:
    #     for child in tabu_child_nodes:
    #         for i in range(d):
    #             forbidden_list.append((i,child))

    if tabu_parent_nodes is not None:
        for parent in tabu_parent_nodes:
            for i in range(d):
                true_dag[parent][i] = 0
    if tabu_child_nodes is not None:
        for child in tabu_child_nodes:
            for i in range(d):
                true_dag[i][child] = 0

    granDAG = GraNDAG(input_dim=X.shape[1])
    granDAG.learn(X)
    sm = StructureModel(granDAG.causal_matrix)
    mean_effect = granDAG.get_mean_effect(X)

    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=mean_effect[u, v],
        )

    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    for node in sm.nodes(data=True):
        node[1]["bias"] = 0
    adj = deepcopy(granDAG.causal_matrix)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])
    return sm

def mcsl(X,d,dist_types,d_orig,tabu_edges = None,
         tabu_parent_nodes = None,tabu_child_nodes = None,num_hidden_layers = 4,
         device_type='cpu',device_ids='0'):



    true_dag = [[ 1 for i in range(d)] for j in range(d)]
    if tabu_edges is not None:
        for i,j in tabu_edges:
            true_dag[i][j] = 0

    # if tabu_parent_nodes is not None:
    #     for child in tabu_child_nodes:
    #         for i in range(d):
    #             forbidden_list.append((i,child))

    if tabu_parent_nodes is not None:
        for parent in tabu_parent_nodes:
            for i in range(d):
                true_dag[parent][i] = 0
    if tabu_child_nodes is not None:
        for child in tabu_child_nodes:
            for i in range(d):
                true_dag[i][child] = 0

    mc = MCSL(model_type='nn',
              iter_step=100,
              rho_thresh=1e20,
              init_rho=1e-5,
              rho_multiply=10,
              graph_thresh=0.5,
              l1_graph_penalty=2e-3,
              device_type=device_type,device_ids=device_ids,num_hidden_layers=num_hidden_layers)
    try:
        mc.learn(X, pns_mask=true_dag)
    except:
        traceback.print_exc()
    sm = StructureModel(mc.causal_matrix)

    mean_effect = mc.get_mean_effect()

    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=mean_effect[u, v],
        )

    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    for node in sm.nodes(data=True):
       node[1]["bias"] = 0
    adj = deepcopy(mc.causal_matrix)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

    return sm


# def anm():
#     anm = ANMNonlinear()
#     anm.learn()



def dag_gnn(X, d, dist_types, hidden_layer_units, lasso_beta, ridge_beta, use_bias, use_gpu, kwargs, max_iter,
                    w_threshold, d_orig, tabu_edges=None, tabu_parent_nodes=None, tabu_child_nodes=None,device_type="cpu"):
        hidden_layer_bnds = hidden_layer_units[0] if hidden_layer_units[0] else 1
        bnds = [
            (0, 0)
            if i == j
            else (0, 0)
            if tabu_edges is not None and (i, j) in tabu_edges
            else (0, 0)
            if tabu_parent_nodes is not None and i in tabu_parent_nodes
            else (0, 0)
            if tabu_child_nodes is not None and j in tabu_child_nodes
            else (None, None)
            for j in range(d)
            for _ in range(hidden_layer_bnds)
            for i in range(d)
        ]
        model = DAG_GNN(
        )
        model.learn(X)
        sm = StructureModel(model.causal_matrix)

        if w_threshold:
            sm.remove_edges_below_threshold(w_threshold)

        # extract the mean effect and add as edge attribute
        mean_effect = model.get_mean_effect(X)

        for u, v, edge_dict in sm.edges.data(True):
            sm.add_edge(
                u,
                v,
                origin="learned",
                weight=edge_dict["weight"],
                mean_effect=mean_effect[u, v],
            )

        # set bias as node attribute


        # attach each dist_type object to corresponding node(s)
        for dist_type in dist_types:
            sm = dist_type.add_to_node(sm)

        # preserve the structure_learner as a graph attribute
        sm.graph["structure_learner"] = model
        for node in sm.nodes(data=True):
            node[1]["bias"] = 0
        # collapse the adj down and store as graph attr
        adj = deepcopy(model.causal_matrix)
        for dist_type in dist_types:
            adj = dist_type.collapse_adj(adj)
        sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

        return sm


def notears_mlp(
    X,
    d,
    dist_types,
    hidden_layer_units,
    lasso_beta,
    ridge_beta,
    use_bias,
    use_gpu,
    kwargs,
    max_iter,
    w_threshold,
    d_orig,
    candidate_active=None,
    candidate_inactive=None,
    constraint_config=None,
):
    candidate_dict = {}
    if candidate_active is not None:
        candidate_dict["active"] = candidate_active
    if candidate_inactive is not None:
        candidate_dict["inactive"] = candidate_inactive

    if constraint_config is None:
        constraint_config = {
            "active": {
                "use": candidate_active is not None,
                "method": "max",
                "threshold": 0.6,
                "lamb": 0.05,
                "model": ActiveConstraints,
                "name": "active",
                "use_transitive_closure":True,
                "alpha" : 0.8,
                "beta":2,
                "threshold1":0.2
            },
            "inactive": {
            "use": True,
            "lamb":0.05,
            "method": "potential",
            "tau": 0.05,
            "beta": 20.0,
            "sigma": 1e-3,
            "adapt_tau": True,
            "adapt_beta": True,
            "adapt_lamb": False,
            "model": InactiveConstraints,
            "name": "inactive",

            },
        }
    else:
        constraint_config = deepcopy(constraint_config)
        for name, conf in constraint_config.items():
            conf.setdefault("name", name)
            if "model" not in conf:
                conf["model"] = ActiveConstraints if name == "active" else InactiveConstraints
            conf.setdefault("use", False)

    # delegate training to existing NotearsNonlinear implementation
    model = NotearsNonlinear(
        hidden_layers=tuple(hidden_layer_units) if hidden_layer_units else (1,),
        lambda1=lasso_beta,
        lambda2=ridge_beta,
        w_threshold=w_threshold if w_threshold is not None else 0.0,
        max_iter=max_iter,
        bias=use_bias,
        use_pytorch_optimizer=kwargs.get("use_pytorch_optimizer", False),
        config=constraint_config,
        candidate_dict=candidate_dict,
        device_type="gpu" if use_gpu else "cpu",
    )
    model.learn(X)

    weights = getattr(model, "weight_causal_matrix", None)
    if weights is None:
        weights = model.causal_matrix
    weights_np = np.asarray(weights)

    sm = StructureModel(weights_np, origin="learned")

    # attach each dist_type object to corresponding node(s)
    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    # preserve the structure_learner as a graph attribute
    sm.graph["structure_learner"] = model

    # collapse the adj down and store as graph attr
    adj = deepcopy(weights_np)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

    return sm


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def from_pandas(
    X: pd.DataFrame,
    dist_type_schema: Dict[Union[str, int], str] = None,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    max_iter: int = 100,
    w_threshold: float = None,
    candidate_active: np.ndarray = None,
    candidate_inactive: np.ndarray = None,
    constraint_config: Dict = None,
    tabu_edges: List[Tuple[str, str]] = None,
    tabu_parent_nodes: List[str] = None,
    tabu_child_nodes: List[str] = None,
    use_gpu: bool = True,
    soft_tabu_edges: List[Tuple[str, str]] = None,
    soft_tabu_parent_nodes: List[str] = None,
    soft_tabu_child_nodes: List[str] = None,
    use_model = "notears-mlp",
    **kwargs,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure describing conditional dependencies between variables
    in data presented as a pandas dataframe.

    The optimisation is to minimise a score function :math:`F(W)` over the graph's
    weighted adjacency matrix, :math:`W`, subject to the a constraint function :math:`h(W)`,
    where :math:`h(W) == 0` characterises an acyclic graph.
    :math:`h(W) > 0` is a continuous, differentiable function that encapsulated how acyclic the graph is
    (less == more acyclic).
    Full details of this approach to structure learning are provided in the publication:

    Based on DAGs with NO TEARS.
    @inproceedings{zheng2018dags,
        author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
        booktitle = {Advances in Neural Information Processing Systems},
        title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
        year = {2018},
        codebase = {https://github.com/xunzheng/notears}
    }

    Args:
        X: 2d input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.

        dist_type_schema: The dist type schema corresponding to the passed in data X.
        It maps the pandas column name in X to the string alias of a dist type.
        A list of alias names can be found in ``dist_type/__init__.py``.
        If None, assumes that all data in X is continuous.

        lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
        NOTE when using nonlinearities, the l1 loss only applies to the dag_layer.

        use_bias: Whether to fit a bias parameter in the NOTEARS algorithm.

        ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
        When using nonlinear layers use of this parameter is recommended.

        hidden_layer_units: An iterable where its length determine the number of layers used,
        and the numbers determine the number of nodes used for the layer in order.

        w_threshold: fixed threshold for absolute edge weights.

        max_iter: max number of dual ascent steps during optimisation.

        tabu_edges: list of edges(from, to) not to be included in the graph.

        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.

        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

        use_gpu: use gpu if it is set to True and CUDA is available

        **kwargs: additional arguments for NOTEARS MLP model

    Returns:
         StructureModel: graph of conditional dependencies between data variables.

    Raises:
        ValueError: If X does not contain data.
    """

    data = deepcopy(X)

    # if dist_type_schema is not None, convert dist_type_schema from cols to idx
    dist_type_schema = (
        dist_type_schema
        if dist_type_schema is None
        else {X.columns.get_loc(col): alias for col, alias in dist_type_schema.items()}
    )
    non_numeric_cols = data.select_dtypes(exclude="number").columns

    if len(non_numeric_cols) > 0:
        raise ValueError(
            "All columns must have numeric data. "
            f"Consider mapping the following columns to int {non_numeric_cols}"
        )

    col_idx = {c: i for i, c in enumerate(data.columns)}
    idx_col = {i: c for c, i in col_idx.items()}

    if tabu_edges:
        tabu_edges = [(col_idx[u], col_idx[v]) for u, v in tabu_edges]
    if tabu_parent_nodes:
        tabu_parent_nodes = [col_idx[n] for n in tabu_parent_nodes]
    if tabu_child_nodes:
        tabu_child_nodes = [col_idx[n] for n in tabu_child_nodes]
    if soft_tabu_edges:
        tabu_edges = [(col_idx[u], col_idx[v]) for u, v in tabu_edges]
    if soft_tabu_parent_nodes:
        tabu_parent_nodes = [col_idx[n] for n in tabu_parent_nodes]
    if soft_tabu_child_nodes:
        tabu_child_nodes = [col_idx[n] for n in tabu_child_nodes]


    g = from_numpy(
        X=data.values,
        dist_type_schema=dist_type_schema,
        lasso_beta=lasso_beta,
        ridge_beta=ridge_beta,
        use_bias=use_bias,
        hidden_layer_units=hidden_layer_units,
        w_threshold=w_threshold,
        max_iter=max_iter,
        candidate_active=candidate_active,
        candidate_inactive=candidate_inactive,
        constraint_config=constraint_config,
        tabu_edges=tabu_edges,
        tabu_parent_nodes=tabu_parent_nodes,
        tabu_child_nodes=tabu_child_nodes,
        use_gpu=use_gpu,
        use_model=use_model,
        **kwargs,
    )


    # set comprehension to ensure only unique dist types are extraced
    # NOTE: this prevents double-renaming caused by the same dist type used on expanded columns
    unique_dist_types = {node[1]["dist_type"] for node in g.nodes(data=True)}
    # use the dist types to update the idx_col mapping
    idx_col_expanded = deepcopy(idx_col)
    for dist_type in unique_dist_types:
        idx_col_expanded = dist_type.update_idx_col(idx_col_expanded)

    sm = StructureModel()
    # add expanded set of nodes
    sm.add_nodes_from(list(idx_col_expanded.values()))

    # recover the edge weights from g
    for u, v, edge_dict in g.edges.data(True):
        sm.add_edge(
            idx_col_expanded[u],
            idx_col_expanded[v],
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=edge_dict["mean_effect"],
        )

    # retrieve all graphs attrs
    for key, val in g.graph.items():
        sm.graph[key] = val

    # recover the node biases from g
    for node in g.nodes(data=True):
        node_name = idx_col_expanded[node[0]]
        sm.nodes[node_name]["bias"] = node[1]["bias"]

    # recover and preseve the node dist_types
    for node_data in g.nodes(data=True):
        node_name = idx_col_expanded[node_data[0]]
        sm.nodes[node_name]["dist_type"] = node_data[1]["dist_type"]

    # recover the collapsed model from g
    sm_collapsed = StructureModel()
    sm_collapsed.add_nodes_from(list(idx_col.values()))
    for u, v, edge_dict in g.graph["graph_collapsed"].edges.data(True):
        sm_collapsed.add_edge(
            idx_col[u],
            idx_col[v],
            origin="learned",
            weight=edge_dict["weight"],
        )
    sm.graph["graph_collapsed"] = sm_collapsed

    return sm
