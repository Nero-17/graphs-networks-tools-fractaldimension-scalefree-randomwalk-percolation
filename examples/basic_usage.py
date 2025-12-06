# examples/basic_usage.py

import networkx as nx
from graph_fractal_dimension import cpu_compute_box_dimension


def main():
    # Build a test graph
    n_nodes = 200
    m_edges = 3
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=0)

    # Compute box dimension
    R2, dimB = cpu_compute_box_dimension(
        G,
        plot="on",
        count_diameter_less_nine="on",
    )

    print(f"R^2 of log–log fit: {R2:.4f}")
    print(f"Estimated box dimension dim_B: {dimB:.4f}")


if __name__ == "__main__":
    main()
