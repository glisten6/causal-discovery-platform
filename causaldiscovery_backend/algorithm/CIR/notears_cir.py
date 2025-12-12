import numpy as np
from castle import GraphDAG, MetricsDAG
from castle.datasets import IIDSimulation, DAG

from CIR.nonlinear_qpm import NotearsNonlinear
from CIR.utils.util import find_skeleton


if __name__ == "__main__":
    type = 'ER'  # or `SF`
    h = 2  # ER2 when h=5 --> ER5
    n_nodes = 30
    n_edges = h * n_nodes
    method = 'nonlinear'
    sem_type = 'gp'

    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                          weight_range=(0.5, 2.0), seed=1)

    dataset = IIDSimulation(W=weighted_random_dag, n=1000,
                            method=method, sem_type=sem_type)
    true_dag, X = dataset.B, dataset.X
    skeleton,_ = find_skeleton(X,alpha=0.05, ci_test='fisherz')
    cir_regulation = np.where(skeleton == 0,1,0)

    al = NotearsNonlinear(cir = True,cir_regulation=cir_regulation)
    al.learn(X)
    GraphDAG(al.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(al.causal_matrix, true_dag)
    print("with cir")
    print(met.metrics)




    al = NotearsNonlinear()
    al.learn(X)
    GraphDAG(al.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(al.causal_matrix, true_dag)
    print("without cir")
    print(met.metrics)
