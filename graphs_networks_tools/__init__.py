# graphs_networks_tools/__init__.py

"""
Graphs Networks Tools:
- Box-counting (fractal) dimension of graphs (CPU + GPU)
- Degree dimension (degree exponent / scale-free exponent)
"""

from .box_dimension import cpu_compute_box_dimension

# GPU version is optional: import if available, otherwise set to None
try:
    from .box_dimension_gpu import gpu_compute_box_dimension
except Exception:
    gpu_compute_box_dimension = None

from .scale_free_exponent import compute_degree_dimension

__all__ = [
    "cpu_compute_box_dimension",
    "gpu_compute_box_dimension",
    "compute_degree_dimension",
]
