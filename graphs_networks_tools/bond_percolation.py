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



def plot_percolation_curve(G, num_points=20, num_test=10, percolation_error=0.1,
                           mapping_function=None, eval_function=None, iterations=1,
                           diameter_ratio_plot="on", degree_dimension_plot="off",
                           percolation_connectivity="on", percolation_phase_diagram="off",
                           phase_iterations=20, phase_grid_resolution=100,
                           double_percolation_analysis="off"):
    """
    在 [0,1] 内取不同的 p 值，对图 G 进行 percolation 实验，计算：
      1. 最大随机簇直径与原图直径的比值（当 diameter_ratio_plot=="on" 时计算）；
      2. 节点 1 和节点 2 的连通概率（当 percolation_connectivity=="on" 时计算与绘图）；
      3. 如果提供了 mapping_function 与 eval_function，则计算 f(F^(iterations-1)(p, p))；
      4. 如果 degree_dimension_plot=="on"，则对每个 p 值进行 num_test 次实验，
         计算最大连通子图的 degree dimension，取平均后绘制“平均 degree dimension vs p”曲线；
      5. 如果 percolation_phase_diagram=="on"，则在 [0,1]×[0,1] 上生成相图，
         判断映射 F 的吸引域（收敛到 (0,0) 标为红色，收敛到 (1,1) 标为蓝色），
         并记录红蓝交界处的点（构成边界）。
      6. 如果 double_percolation_analysis=="on"，则利用相图中得到的边界点（若相图未开启则重新计算），
         将每个边界点 (p₁, p₂) 视为双概率条件（蓝色边 open 概率为 p₁，红色边 open 概率为 p₂），
         进行 num_test 次 percolation 实验，计算最大 cluster 直径/原图直径的比值，最后整理成有序数组并绘图。

    同时输出最小满足连通概率 > percolation_error 的 p 值（threshold）。

    参数说明（除前面已有参数外，新增加的参数说明如下）：
      - percolation_phase_diagram: "on" 表示绘制映射 F 的相图；"off" 则不计算。
      - phase_iterations: 相图中对每个 (x, y) 迭代 mapping_function 的次数。
      - phase_grid_resolution: 相图网格的分辨率（例如 100 表示 100×100 网格）。
      - double_percolation_analysis: "on" 表示利用相图边界点（双概率）计算 percolation cluster 直径比值；"off" 则不计算。
    """
    p_values = np.linspace(0, 1, num_points)

    # 用于存储不同 p 下的结果
    connectivity_probs = []  # 连通概率（仅在 percolation_connectivity=="on" 时记录）
    if diameter_ratio_plot == "on":
        ratio_values = []
    if degree_dimension_plot == "on":
        avg_degree_dimensions = []

    # 对于每个 p 值进行实验
    for p_val in p_values:
        # 计算簇直径比值
        if diameter_ratio_plot == "on":
            H = G.copy()
            for edge in list(H.edges()):
                if random.random() > p_val:
                    H.remove_edge(*edge)
            H_undirected = H.to_undirected() if nx.is_directed(H) else H
            if H_undirected.number_of_edges() == 0:
                max_diam = 0
            else:
                if nx.is_connected(H_undirected):
                    max_diam = nx.diameter(H_undirected)
                else:
                    max_diam = max(nx.diameter(H_undirected.subgraph(c)) for c in nx.connected_components(H_undirected))
            # 原图直径（第一次计算即可）
            if 'original_diam' not in locals():
                G_undirected = G.to_undirected() if nx.is_directed(G) else G
                if nx.is_connected(G_undirected):
                    original_diam = nx.diameter(G_undirected)
                else:
                    original_diam = max(nx.diameter(G_undirected.subgraph(c)) for c in nx.connected_components(G_undirected))
            ratio = max_diam / original_diam if original_diam > 0 else 0
            ratio_values.append(ratio)

        # 对每个 p 值进行 num_test 次实验
        success_count = 0
        degree_dims = []  # 记录每次实验中最大连通子图的 degree dimension
        for _ in range(num_test):
            H_test = G.copy()
            for edge in list(H_test.edges()):
                if random.random() > p_val:
                    H_test.remove_edge(*edge)
            H_test = H_test.to_undirected() if nx.is_directed(H_test) else H_test
            if percolation_connectivity == "on":
                if nx.has_path(H_test, 1, 2):
                    success_count += 1
            if degree_dimension_plot == "on":
                components = list(nx.connected_components(H_test))
                if len(components) > 0:
                    largest_comp_nodes = max(components, key=len)
                    largest_comp = H_test.subgraph(largest_comp_nodes)
                    degree_dim, _, _ = compute_degree_dimension(largest_comp)
                    degree_dims.append(degree_dim)
        if percolation_connectivity == "on":
            connectivity_prob = success_count / num_test
            connectivity_probs.append(connectivity_prob)
        if degree_dimension_plot == "on":
            avg_degree_dimensions.append(np.mean(degree_dims) if degree_dims else 0)

    threshold_p = None
    if percolation_connectivity == "on":
        for p_val, cp in zip(p_values, connectivity_probs):
            if cp > percolation_error:
                threshold_p = p_val
                break

    # 绘制直径比值图（如果启用）
    if diameter_ratio_plot == "on":
        fig1, ax1 = plt.subplots(figsize=(12.1, 6))
        ax1.plot(p_values, ratio_values, marker='o')
        ax1.set_xlabel("p (edge retention probability)")
        ax1.set_ylabel("Max cluster diameter / Original graph diameter")
        ax1.set_title("Percolation Cluster Diameter Ratio vs p")
        ax1.grid(True)
        ax1.set_xticks(np.arange(0, 1.05, 0.05))
        plt.show()

    # 绘制连通概率及 mapping 函数（f(F^(iterations-1)(p,p))）图
    if percolation_connectivity == "on" or (mapping_function is not None and eval_function is not None):
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        if percolation_connectivity == "on":
            ax2.plot(p_values, connectivity_probs, marker='o', color='green', label="Connectivity probability")
        if mapping_function is not None and eval_function is not None:
            mapped_values = []
            for p_val in p_values:
                x, y = p_val, p_val
                for _ in range(iterations - 1):
                    x, y = mapping_function(x, y)
                mapped_value = eval_function(x, y)
                mapped_values.append(mapped_value)
            ax2.plot(p_values, mapped_values, marker='s', color='orange', label=f"f(F^({iterations-1})(p,p))")
        ax2.set_xlabel("p (edge retention probability)")
        ax2.set_ylabel("Value")
        ax2.set_title("Percolation Connectivity and Mapping Function vs p")
        ax2.grid(True)
        ax2.set_xticks(np.arange(0, 1.05, 0.05))
        ax2.set_yticks(np.arange(0, 1.05, 0.05))
        ax2.legend()
        plt.show()

    # 单独绘制平均 Degree Dimension vs p 的图（如果启用）
    if degree_dimension_plot == "on":
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(p_values, avg_degree_dimensions, marker='^', color='red', label="Avg. Degree Dimension")
        ax3.set_xlabel("p (edge retention probability)")
        ax3.set_ylabel("Avg. Degree Dimension")
        ax3.set_title("Average Degree Dimension vs p")
        ax3.grid(True)
        ax3.set_xticks(np.arange(0, 1.05, 0.05))
        ax3.legend()
        plt.show()

    # 输出阈值信息（仅当开启连通性计算时）
    if percolation_connectivity == "on":
        if threshold_p is not None:
            print(f"Threshold p (connectivity probability > {percolation_error}): {threshold_p:.3f}")
        else:
            print(f"No threshold p found with connectivity probability > {percolation_error}.")

    # 单独绘制原图 G 的 degree distribution 的 log–log 图（作为参考，当 degree_dimension_plot=="on" 时）
    if degree_dimension_plot == "on":
        degree_dim_full, unique_degrees, counts = compute_degree_dimension(G)
        plt.figure(figsize=(8,6))
        plt.scatter(np.log(unique_degrees), np.log(counts), label="Data points")
        slope, intercept = np.polyfit(np.log(unique_degrees), np.log(counts), 1)
        x_fit = np.linspace(min(np.log(unique_degrees)), max(np.log(unique_degrees)), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color='red', label=f"Fit line: slope={slope:.2f}")
        plt.xlabel("log(Degree)")
        plt.ylabel("log(Count)")
        plt.title(f"G's Degree Distribution Log-Log Plot (Degree Dimension = {degree_dim_full:.2f})")
        plt.legend()
        plt.show()

    # 绘制相图：在 [0,1]×[0,1] 上判断映射 F 的吸引域（当 percolation_phase_diagram=="on" 时）
    if percolation_phase_diagram == "on" and mapping_function is not None:
        resolution = phase_grid_resolution
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.linspace(0, 1, resolution)
        classification = np.empty((resolution, resolution), dtype=int)
        for i, x0 in enumerate(x_vals):
            for j, y0 in enumerate(y_vals):
                x, y = x0, y0
                for _ in range(phase_iterations):
                    x, y = mapping_function(x, y)
                d0 = np.sqrt(x**2 + y**2)
                d1 = np.sqrt((x-1)**2 + (y-1)**2)
                classification[j, i] = 0 if d0 < d1 else 1
        # 寻找边界点：若某点邻域内有分类不同的点，则认为其在红蓝交界处
        boundaries = []
        for i in range(resolution):
            for j in range(resolution):
                current = classification[j, i]
                neighbors = []
                if i > 0: neighbors.append(classification[j, i-1])
                if i < resolution-1: neighbors.append(classification[j, i+1])
                if j > 0: neighbors.append(classification[j-1, i])
                if j < resolution-1: neighbors.append(classification[j+1, i])
                if any(nb != current for nb in neighbors):
                    boundaries.append((x_vals[i], y_vals[j]))
        boundaries = np.array(boundaries)

        plt.figure(figsize=(8,6))
        plt.imshow(classification, extent=[0,1,0,1], origin='lower', cmap=plt.get_cmap('coolwarm'), alpha=0.6)
        plt.colorbar(label='Attractor: 0 -> (0,0) [red], 1 -> (1,1) [blue]')
        if boundaries.size > 0:
            plt.scatter(boundaries[:,0], boundaries[:,1], c='black', s=1, label='Boundary')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phase Diagram for Mapping F')
        plt.legend()
        plt.show()

    # 新功能：双概率 percolation 分析——利用相图中边界点（或若未绘制相图则重新计算）来模拟双概率 percolation，
    # 蓝色边 open 概率 = p₁, 红色边 open 概率 = p₂，计算最大 cluster 直径 / 原图直径。
        # 新功能：双概率 percolation 分析——利用相图边界点（双概率）计算 percolation cluster 的直径 ratio 以及 edge 数量 ratio
    if double_percolation_analysis == "on" and mapping_function is not None:
        # 获取边界点数组：如果之前相图已计算，则使用 boundaries，否则重新计算
        if percolation_phase_diagram == "on" and 'boundaries' in locals():
            boundaries_array = boundaries
        else:
            resolution = phase_grid_resolution
            x_vals = np.linspace(0, 1, resolution)
            y_vals = np.linspace(0, 1, resolution)
            classification = np.empty((resolution, resolution), dtype=int)
            for i, x0 in enumerate(x_vals):
                for j, y0 in enumerate(y_vals):
                    x, y = x0, y0
                    for _ in range(phase_iterations):
                        x, y = mapping_function(x, y)
                    d0 = np.sqrt(x**2 + y**2)
                    d1 = np.sqrt((x-1)**2 + (y-1)**2)
                    classification[j, i] = 0 if d0 < d1 else 1
            boundaries_temp = []
            for i in range(resolution):
                for j in range(resolution):
                    current = classification[j, i]
                    neighbors = []
                    if i > 0: neighbors.append(classification[j, i-1])
                    if i < resolution-1: neighbors.append(classification[j, i+1])
                    if j > 0: neighbors.append(classification[j-1, i])
                    if j < resolution-1: neighbors.append(classification[j+1, i])
                    if any(nb != current for nb in neighbors):
                        boundaries_temp.append((x_vals[i], y_vals[j]))
            boundaries_array = np.array(boundaries_temp)
        # 将边界点按 p₁（即 x 值）排序，形成有序数组
        boundaries_sorted = boundaries_array[np.argsort(boundaries_array[:,0])]

        # 计算原图的直径与边数
        G_undirected = G.to_undirected() if nx.is_directed(G) else G
        if nx.is_connected(G_undirected):
            original_diam = nx.diameter(G_undirected)
        else:
            original_diam = max(nx.diameter(G_undirected.subgraph(c)) for c in nx.connected_components(G_undirected))
        original_edge_count = G.number_of_edges()

        double_points = []
        double_diameter_ratios = []
        double_edge_ratios = []
        for (p1, p2) in boundaries_sorted:
            diameter_ratio_experiments = []
            edge_ratio_experiments = []
            for _ in range(num_test):
                H_temp = G.copy()
                for edge in list(H_temp.edges()):
                    edge_color = H_temp.edges[edge].get("color", None)
                    if edge_color == "blue":
                        if random.random() > p1:
                            H_temp.remove_edge(*edge)
                    elif edge_color == "red":
                        if random.random() > p2:
                            H_temp.remove_edge(*edge)
                    else:
                        # 若边颜色不是蓝也不是红，可不做处理或采用默认
                        pass
                H_temp = H_temp.to_undirected() if nx.is_directed(H_temp) else H_temp
                if H_temp.number_of_edges() == 0:
                    diameter_ratio_experiments.append(0)
                    edge_ratio_experiments.append(0)
                else:
                    # 计算直径 ratio
                    if nx.is_connected(H_temp):
                        max_diam_temp = nx.diameter(H_temp)
                    else:
                        max_diam_temp = max(nx.diameter(H_temp.subgraph(comp)) for comp in nx.connected_components(H_temp))
                    diameter_ratio_experiments.append(max_diam_temp / original_diam if original_diam > 0 else 0)
                    # 计算最大簇的边数量 ratio
                    components = list(nx.connected_components(H_temp))
                    largest_comp_nodes = max(components, key=len)
                    largest_comp = H_temp.subgraph(largest_comp_nodes)
                    edge_count = largest_comp.number_of_edges()
                    edge_ratio_experiments.append(edge_count / original_edge_count if original_edge_count > 0 else 0)
            avg_diameter_ratio = np.mean(diameter_ratio_experiments)
            avg_edge_ratio = np.mean(edge_ratio_experiments)
            double_points.append((p1, p2))
            double_diameter_ratios.append(avg_diameter_ratio)
            double_edge_ratios.append(avg_edge_ratio)
        double_points = np.array(double_points)

        # 绘制直径 ratio 的散点图，使用黑白色图 (gray_r: 低值白，高值黑)
        plt.figure(figsize=(8,6))
        sc1 = plt.scatter(double_points[:,0], double_points[:,1], c=double_diameter_ratios, cmap='gray_r', s=50)
        plt.xlabel("p₁ (Probability for blue edges)")
        plt.ylabel("p₂ (Probability for red edges)")
        plt.title("Double-Percolation Cluster Diameter Ratio on Boundary Points")
        plt.colorbar(sc1, label="Cluster Diameter Ratio")
        plt.show()

        # 绘制 edge 数量 ratio 的散点图，使用黑白色图 (gray_r: 低值白，高值黑)
        plt.figure(figsize=(8,6))
        sc2 = plt.scatter(double_points[:,0], double_points[:,1], c=double_edge_ratios, cmap='gray_r', s=50)
        plt.xlabel("p₁ (Probability for blue edges)")
        plt.ylabel("p₂ (Probability for red edges)")
        plt.title("Double-Percolation Cluster Edge Count Ratio on Boundary Points")
        plt.colorbar(sc2, label="Cluster Edge Count Ratio")
        plt.show()










def plot_IGS_and_percolation(G, layout, p=0.6, direction="on", colour="on"):
    """
    在同一画布上并排展示 IGS 图与 bond percolation 模拟图，
    并共用同一个由外部计算好的 layout。

    参数：
      - G: 迭代图系统生成的图（NetworkX 对象）。
      - layout: 外部计算好的节点位置字典。
      - p: bond percolation 的边保留概率，默认 0.6。
      - direction: "on" 表示绘制有向图（显示箭头）；"off" 表示无向图。
      - colour: "on" 表示按照边属性 color 绘制；"off" 表示所有边绘制为黑色。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制 IGS 图
    if colour == "on":
        edge_colors = [attr.get("color", "black") for u, v, attr in G.edges(data=True)]
    else:
        edge_colors = "black"
    arrow_flag = (direction == "on")
    nx.draw(G, layout, ax=axes[0], with_labels=False, edge_color=edge_colors,
            node_size=1, arrows=arrow_flag)
    axes[0].set_title("Iterated Graph Systems")
    axes[0].axis("off")

    # 绘制 bond percolation 图
    draw_bond_percolation(G, p=p, layout=layout, ax=axes[1])

    plt.tight_layout()
    plt.show()
