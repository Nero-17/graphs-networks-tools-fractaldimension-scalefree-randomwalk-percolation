# examples/basic_usage.py
!pip install --no-cache-dir git+https://github.com/Nero-17/fractal-dimension-of-graphs.git

import networkx as nx
from graph_fractal_dimension import (
    cpu_compute_box_dimension,
    gpu_compute_box_dimension,
)


def main():
    # 1. Build a test graph (Barabási–Albert model)
    n_nodes = 200
    m_edges = 3
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=0)

    print("=== CPU version (no diameter threshold) ===")
    R2_cpu, dim_cpu = cpu_compute_box_dimension(
        G,
        plot="on",              # show log–log plot
        diameter_threshold=None # no diameter cut-off
    )
    print(f"CPU  R^2: {R2_cpu:.4f}")
    print(f"CPU  dim_B: {dim_cpu:.4f}")

    print("\n=== CPU version with diameter_threshold=9 ===")
    R2_cpu_cut, dim_cpu_cut = cpu_compute_box_dimension(
        G,
        plot="off",             # do not plot this time
        diameter_threshold=9,   # if diameter(G) <= 9, returns (0.0, 0.0)
    )
    print(f"CPU (threshold)  R^2: {R2_cpu_cut:.4f}")
    print(f"CPU (threshold)  dim_B: {dim_cpu_cut:.4f}")

    # 2. Optional GPU example (only if available and torch+CUDA installed)
    if gpu_compute_box_dimension is not None:
        print("\n=== GPU version with diameter_threshold=9 ===")
        R2_gpu, dim_gpu = gpu_compute_box_dimension(
            G,
            plot="off",
            diameter_threshold=9,
        )
        print(f"GPU  R^2: {R2_gpu:.4f}")
        print(f"GPU  dim_B: {dim_gpu:.4f}")
    else:
        print("\nGPU version not available (PyTorch/CUDA not installed).")


if __name__ == "__main__":
    main()
