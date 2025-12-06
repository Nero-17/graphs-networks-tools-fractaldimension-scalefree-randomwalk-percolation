# graphs_networks_tools/__init__.py

from .box_dimension import cpu_compute_box_dimension

try:
    from .box_dimension_gpu import gpu_compute_box_dimension
except Exception:
    gpu_compute_box_dimension = None

from .scale_free_exponent import compute_degree_dimension

__all__ = [
    "cpu_compute_fractal_dimension",
    "gpu_compute_fractal_dimension",
    "compute_scale_free_exponent",
]
