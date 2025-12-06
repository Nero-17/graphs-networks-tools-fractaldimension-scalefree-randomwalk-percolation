# graphs_networks_tools/__init__.py

from .graphs_networks_tools import cpu_compute_fractal_dimension

try:
    from .graphs_networks_tools import gpu_compute_fractal_dimension
except Exception:
    gpu_compute_box_dimension = None

from .graphs_networks_tools import compute_scale_free_exponent

__all__ = [
    "cpu_compute_fractal_dimension",
    "gpu_compute_fractal_dimension",
    "compute_scale_free_exponent",
]
