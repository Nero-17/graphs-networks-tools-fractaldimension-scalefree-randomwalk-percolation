"""
GPU-based prototype for computing the box-counting dimension of graphs.

Main entry point:
    gpu_compute_box_dimension(G, plot='off', count_diameter_less_nine='on', device=None)

This function mirrors the API of cpu_compute_box_dimension, but uses
PyTorch on CUDA to accelerate the ball computations.
"""

from math import floor

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# PyTorch is optional: we handle the case where it is not installed.
try:
    import torch
except ImportError:  # torch not installed
    torch = None


def _check_torch():
    """
    Ensure that PyTorch is installed and a CUDA GPU is available.
    """
    if torch is None:
        raise ImportError(
            "gpu_compute_box_dimension requires PyTorch. "
            "Please install it, e.g. `pip install torch`."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. "
            "In Colab, select a GPU runtime (Runtime -> Change runtime type -> GPU)."
        )


def _build_sparse_adj(G, device):
    """
    Build a symmetric sparse adjacency matrix on the given device.

    Returns
    -------
    adj : torch.sparse_coo_tensor, shape [n, n]
        Undirected adjacency matrix (0/1).
    nodes : list
        List of nodes in a fixed order.
    node_to_idx : dict
        Mapping from node label to row/column index.
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    row = []
    col = []

    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        # Undirected: add both (i, j) and (j, i)
        row.extend([i, j])
        col.extend([j, i])

    if len(row) == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        indices = torch.tensor([row, col], dtype=torch.long, device=device)
        values = torch.ones(len(row), dtype=torch.float32, device=device)

    adj = torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()
    return adj, nodes, node_to_idx


def _gpu_ball_from_node(adj, start_idx, r):
    """
    Compute the ball of graph-radius <= r around start_idx using GPU operations.

    Parameters
    ----------
    adj : sparse [n, n] tensor on CUDA
        Adjacency matrix.
    start_idx : int
        Index of the starting node.
    r : int
        Radius in graph distance.

    Returns
    -------
    visited : bool tensor of shape [n]
        visited[i] is True iff distance(start_idx, i) <= r.
    """
    device = adj.device
    n = adj.size(0)

    visited = torch.zeros(n, dtype=torch.bool, device=device)
    frontier = torch.zeros(n, dtype=torch.bool, device=device)

    visited[start_idx] = True
    frontier[start_idx] = True

    for _ in range(r):
        if not frontier.any():
            break

        # Multiply sparse adjacency by the frontier vector
        front_f = frontier.to(dtype=torch.float32).view(-1, 1)  # [n, 1]
        neighbors = torch.sparse.mm(adj, front_f).view(-1) > 0

        new_frontier = neighbors & ~visited
        if not new_frontier.any():
            break

        visited |= new_frontier
        frontier = new_frontier

    return visited


def gpu_compute_box_dimension(
    G,
    plot="off",
    count_diameter_less_nine="on",
    device=None,
):
    """
    Compute the box (fractal) dimension of a graph using a GPU prototype.

    Parameters
    ----------
    G : networkx.Graph or object with .to_networkx()
        Input graph.
    plot : {'on', 'off'}, optional
        If 'on', show a log–log plot of N_box(l) vs l and the fitted line.
    count_diameter_less_nine : {'on', 'off'}, optional
        If 'off' and diameter(G) <= 9, skip computation and return None.
    device : str or None, optional
        PyTorch device, e.g. 'cuda' or 'cuda:0'.
        If None, defaults to 'cuda'.

    Returns
    -------
    (R2, box_dimension) : tuple of floats
        R2 is the coefficient of determination of the log–log regression.
        box_dimension is the estimated box-counting dimension.
        Returns (0.0, 0.0) if regression cannot be performed.
    """
    from sklearn.linear_model import LinearRegression

    _check_torch()

    if device is None:
        device = "cuda"

    # normalise switch just in case someone passes 'OFF', 'Off', etc.
    flag = str(count_diameter_less_nine).lower()

    # ---- 1. Preprocessing: undirected, connected ----
    if not isinstance(G, nx.Graph):
        G = G.to_networkx()
    else:
        G = G.copy()

    if nx.is_directed(G):
        G = G.to_undirected()

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # ---- 2. Diameter (still CPU, NetworkX) ----
    try:
        diameter = nx.diameter(G)
    except Exception as e:
        print("Error computing diameter:", e)
        return 0.0, 0.0

    # if user explicitly wants to skip small-diameter graphs
    if diameter <= 9 and flag == "off":
        return 0.0, 0.0

    max_l = max(1, floor(diameter / 2))

    # ---- 3. Build sparse adjacency on GPU ----
    adj, nodes, node_to_idx = _build_sparse_adj(G, device=torch.device(device))
    n = len(nodes)

    l_values = []
    N_box_values = []

    # ---- 4. Greedy box covering for each l ----
    for l in range(1, max_l + 1):
        r = l // 2  # match CPU logic exactly; allow r = 0

        uncovered = torch.ones(n, dtype=torch.bool, device=adj.device)
        box_count = 0

        # Cache balls for this specific (l, r)
        ball_cache = {}

        def get_ball(idx):
            if idx in ball_cache:
                return ball_cache[idx]
            ball = _gpu_ball_from_node(adj, idx, r)
            ball_cache[idx] = ball
            return ball

        if l % 2 == 0:
            # ---- Even l: single-centre balls ----
            while uncovered.any():
                best_size = -1
                best_cov = None

                for idx in range(n):
                    if not uncovered[idx]:
                        continue
                    ball = get_ball(idx)
                    cov = ball & uncovered
                    size = int(cov.sum().item())
                    if size > best_size:
                        best_size = size
                        best_cov = cov

                if best_cov is None:
                    break

                uncovered &= ~best_cov
                box_count += 1

        else:
            # ---- Odd l: try two-centre boxes first ----
            while uncovered.any():
                best_size = -1
                best_cov = None

                # Try pairs (u, v) with v neighbour of u
                for u_idx in range(n):
                    if not uncovered[u_idx]:
                        continue
                    u_label = nodes[u_idx]
                    for v_label in G.neighbors(u_label):
                        v_idx = node_to_idx[v_label]
                        if not uncovered[v_idx]:
                            continue
                        ball_u = get_ball(u_idx)
                        ball_v = get_ball(v_idx)
                        cov = (ball_u | ball_v) & uncovered
                        size = int(cov.sum().item())
                        if size > best_size:
                            best_size = size
                            best_cov = cov

                if best_cov is not None and best_size > 0:
                    uncovered &= ~best_cov
                    box_count += 1
                    continue

                # Fallback: single-centre balls (same as even l)
                if not uncovered.any():
                    break

                best_size = -1
                best_cov = None
                for idx in range(n):
                    if not uncovered[idx]:
                        continue
                    ball = get_ball(idx)
                    cov = ball & uncovered
                    size = int(cov.sum().item())
                    if size > best_size:
                        best_size = size
                        best_cov = cov

                if best_cov is None:
                    break

                uncovered &= ~best_cov
                box_count += 1

        l_values.append(l)
        N_box_values.append(box_count)

    # ---- 5. Regression ----
    if len(l_values) < 2:
        return 0.0, 0.0

    log_l = np.log(np.array(l_values)).reshape(-1, 1)
    log_N_box = np.log(np.array(N_box_values))

    reg = LinearRegression().fit(log_l, log_N_box)
    m = reg.coef_[0]
    b = reg.intercept_
    R2 = reg.score(log_l, log_N_box)
    box_dimension = -m

    # ---- 6. Optional plotting ----
    if plot == "on":
        x = np.linspace(min(l_values), max(l_values), 100)

        plt.figure(figsize=(6, 4))
        plt.scatter(l_values, N_box_values, label="data")
        plt.plot(x, np.exp(b) * x**m, label=f"fit: N={np.exp(b):.2f}·l^{m:.2f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("l")
        plt.ylabel("N_box(l)")
        plt.title(f"GPU R²={R2:.3f}, dim_B={box_dimension:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return float(R2), float(box_dimension)
