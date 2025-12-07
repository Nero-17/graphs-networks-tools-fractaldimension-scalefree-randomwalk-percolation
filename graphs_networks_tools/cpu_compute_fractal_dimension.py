# graph_fractal_dimension/box_dimension.py

"""
Compute the (box-counting) fractal dimension of graphs.

Main public function:
    cpu_compute_box_dimension(G, plot='off', diameter_threshold=None)
"""

from math import floor

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def cpu_compute_fractal_dimension(
    G,
    plot: str = "off",
    diameter_threshold=None,
):
    """
    Compute the box (fractal) dimension of a graph G.

    Parameters
    ----------
    G : networkx.Graph or object with .to_networkx()
        The input graph.
    plot : {'on', 'off'}, optional
        If 'on', show a log–log plot of N_box(l) vs l and the fitted line.
    diameter_threshold : int or None, optional
        If not None and the graph diameter is <= diameter_threshold,
        the function returns (0.0, 0.0) without performing the regression.
        If None, no diameter cut-off is applied (apart from the usual
        requirement of having at least two scales for regression).

    Returns
    -------
    R2 : float
        Coefficient of determination of the log–log linear regression.
    box_dimension : float
        Estimated box-counting dimension. Returns (0.0, 0.0) if regression
        cannot be performed (e.g., fewer than 2 data points, or
        diameter <= diameter_threshold when a threshold is given).
    """
    # ---- 1. Preprocessing: undirected, connected ----
    if not isinstance(G, nx.Graph):
        G = G.to_networkx()
    else:
        G = G.copy()

    if nx.is_directed(G):
        G = G.to_undirected()

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # ---- 2. Diameter ----
    try:
        diameter = nx.diameter(G)
    except Exception as e:
        print("Error computing diameter:", e)
        return 0.0, 0.0

    # Optional diameter threshold: if set and diameter is too small, bail out.
    if diameter_threshold is not None and diameter <= diameter_threshold:
        return 0.0, 0.0

    max_l = max(1, floor(diameter / 2))
    l_values = []
    N_box_values = []

    # ---- 3. Box covering ----
    for l in range(1, max_l + 1):
        uncovered = set(G.nodes())
        box_count = 0
        r = l // 2

        if l % 2 == 0:
            # Even l: single-centre balls of radius r
            while uncovered:
                best_center = None
                best_cov = set()
                for node in uncovered:
                    cov = set(
                        nx.single_source_shortest_path_length(
                            G, node, cutoff=r
                        ).keys()
                    ) & uncovered
                    if len(cov) > len(best_cov):
                        best_cov = cov
                        best_center = node
                uncovered -= best_cov
                box_count += 1
        else:
            # Odd l: try two-centre boxes (union of two radius-r balls)
            while uncovered:
                best_pair = None
                best_union = set()
                for u in uncovered:
                    for v in G.neighbors(u):
                        if v not in uncovered:
                            continue
                        cov_u = set(
                            nx.single_source_shortest_path_length(
                                G, u, cutoff=r
                            ).keys()
                        )
                        cov_v = set(
                            nx.single_source_shortest_path_length(
                                G, v, cutoff=r
                            ).keys()
                        )
                        union_cov = (cov_u | cov_v) & uncovered
                        if len(union_cov) > len(best_union):
                            best_union = union_cov
                            best_pair = (u, v)
                if best_union:
                    uncovered -= best_union
                else:
                    # Fallback: single-centre greedy ball
                    best_center = None
                    best_cov = set()
                    for node in uncovered:
                        cov = set(
                            nx.single_source_shortest_path_length(
                                G, node, cutoff=r
                            ).keys()
                        ) & uncovered
                        if len(cov) > len(best_cov):
                            best_cov = cov
                            best_center = node
                    uncovered -= best_cov
                box_count += 1

        l_values.append(l)
        N_box_values.append(box_count)

    # ---- 4. Regression ----
    if len(l_values) < 2:
        return 0.0, 0.0

    log_l = np.log(np.array(l_values)).reshape(-1, 1)
    log_N_box = np.log(np.array(N_box_values))

    reg = LinearRegression().fit(log_l, log_N_box)
    m = reg.coef_[0]
    b = reg.intercept_
    R2 = reg.score(log_l, log_N_box)
    box_dimension = -m

    # ---- 5. Optional plotting ----
    if plot == "on":
        import numpy as _np  # local alias to avoid confusion

        plt.figure(figsize=(6, 4))
        plt.scatter(l_values, N_box_values, label="data")
        x = _np.linspace(min(l_values), max(l_values), 100)
        plt.plot(x, _np.exp(b) * x**m, label=f"fit: N={_np.exp(b):.2f}·l^{m:.2f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("l")
        plt.ylabel("N_box(l)")
        plt.title(f"R²={R2:.3f}, dim_B={box_dimension:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return float(R2), float(box_dimension)
