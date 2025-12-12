import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets.simulator import DAG


def build_notears_er_matrix(
    d: int,
    expected_edges: int,
    weight_range: tuple[float, float] | None = (0.5, 2.0),
    seed: int | None = None,
) -> torch.Tensor:
    """Use NOTears Erdos-Renyi generator to build a (possibly weighted) DAG adjacency matrix."""
    if d <= 0:
        raise ValueError("d must be a positive integer")

    if expected_edges < 0:
        raise ValueError("expected_edges must be non-negative")

    W_np = DAG.erdos_renyi(n_nodes=d, n_edges=expected_edges, weight_range=weight_range, seed=seed)
    return torch.tensor(W_np, dtype=torch.float32)


def compute_w_power_d(W: torch.Tensor) -> torch.Tensor:
    """Return W^d to inspect nilpotency of DAG adjacency matrix under NOTears ER construction."""
    if W.dim() != 2 or W.size(0) != W.size(1):
        raise ValueError("W must be a square matrix")

    d = W.size(0)
    return torch.linalg.matrix_power(W, d)


if __name__ == "__main__":
    d = 30
    expected_edges = 90  # analogous to NOTears ER configuration
    W = build_notears_er_matrix(d=d, expected_edges=expected_edges, seed=43)
    W_power_d = compute_w_power_d(W)

    print("Constructed acyclic matrix W:\n", W)
    print(f"\nW^{d} =\n", W_power_d)

    zero_matrix = torch.zeros_like(W)
    if torch.allclose(W_power_d, zero_matrix, atol=1e-6):
        print("\nW^d is the zero matrix, consistent with DAG acyclicity.")
    else:
        print("\nWarning: W^d has non-zero entries. Check generator parameters or numerical tolerance.")