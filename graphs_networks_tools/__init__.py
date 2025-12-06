from graphs_networks_tools import (
    cpu_compute_fractal_dimension,
    gpu_compute_fractal_dimension,
    compute_scale_free_exponent,
)

# ...

R2_cpu, dim_cpu = cpu_compute_box_dimension(
    G,
    plot="off",
    diameter_threshold=9,
)

R2_gpu, dim_gpu = gpu_compute_box_dimension(
    G,
    plot="off",
    diameter_threshold=9,
)
