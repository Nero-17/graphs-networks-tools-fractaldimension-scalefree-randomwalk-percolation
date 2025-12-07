def draw_bond_percolation(G, p=0.6, layout=None, ax=None):
    """
    在给定的 ax 上绘制一次 Bernoulli bond percolation 模拟结果。

    参数：
      - G: 原图（NetworkX 对象）。
      - p: 边保留概率，默认 0.6。
      - layout: 使用的节点位置字典。如果为 None，则自动计算（但建议外部计算后传入）。
      - ax: 绘图的 matplotlib 轴对象。如果为 None，则新建一个图。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    if layout is None:
        layout = nx.spring_layout(G, scale=2)

    occupied_edges = []
    removed_edges = []
    for edge in G.edges():
        if random.random() < p:
            occupied_edges.append(edge)
        else:
            removed_edges.append(edge)

    nx.draw_networkx_nodes(G, layout, node_color='lightblue', node_size=1, ax=ax)
    nx.draw_networkx_edges(G, layout, edgelist=occupied_edges, style='solid', width=2, edge_color='red', ax=ax)
    nx.draw_networkx_edges(G, layout, edgelist=removed_edges, style='dashed', width=1, edge_color='black', ax=ax)
    ax.set_title(f"Bond Percolation (p = {p})")
    ax.axis("off")
