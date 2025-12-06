import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def compute_degree_dimension(
    G,
    k_min: int = 1,
    plot: str = "off",
):
    """
    Estimate the degree dimension (degree exponent) of a graph G
    by fitting a power law to the degree distribution on a log–log scale.

    Parameters
    ----------
    G : networkx.Graph or object with .to_networkx()
        Input graph. If directed or disconnected, it is converted to
        an undirected graph and restricted to the largest connected component.
    k_min : int, optional
        Minimum degree to include in the fit (default: 1). Degrees < k_min
        are discarded before fitting.
    plot : {'on', 'off'}, optional
        If 'on', show a log–log plot of the degree distribution and the fit.

    Returns
    -------
    R2 : float
        Coefficient of determination of the log–log linear regression.
    degree_dimension : float
        Estimated degree dimension (positive). Returns (0.0, 0.0, ...)
        if regression cannot be performed (e.g., fewer than 2 points).
    unique_degrees : np.ndarray
        Degrees used in the fit (after applying k_min).
    counts : np.ndarray
        Counts of vertices with the corresponding degrees.
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

    # ---- 2. Degree distribution ----
    degrees = np.array([d for _, d in G.degree()])
    unique_degrees, counts = np.unique(degrees, return_counts=True)

    # Exclude very small degrees if desired (and always exclude 0)
    mask = (unique_degrees >= max(k_min, 1))
    unique_degrees = unique_degrees[mask]
    counts = counts[mask]

    if len(unique_degrees) < 2 or counts.sum() == 0:
        return 0.0, 0.0, unique_degrees, counts

    # ---- 3. Log–log regression ----
    log_degrees = np.log(unique_degrees).reshape(-1, 1)
    log_counts = np.log(counts)

    reg = LinearRegression().fit(log_degrees, log_counts)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    R2 = reg.score(log_degrees, log_counts)
    degree_dimension = -slope  # P(k) ~ k^{-degree_dimension}

    # ---- 4. Optional plotting ----
    if plot == "on":
        x = np.linspace(unique_degrees.min(), unique_degrees.max(), 100)
        y_fit = np.exp(intercept) * x**slope

        plt.figure(figsize=(6, 4))
        plt.scatter(unique_degrees, counts, label="data")
        plt.plot(x, y_fit, label=f"fit: ~ k^{slope:.2f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("degree k")
        plt.ylabel("count N(k)")
        plt.title(f"R²={R2:.3f}, degree dimension={degree_dimension:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return float(R2), float(degree_dimension), unique_degrees, counts
