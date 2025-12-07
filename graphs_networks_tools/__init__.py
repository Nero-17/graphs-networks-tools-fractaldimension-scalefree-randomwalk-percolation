# graphs_networks_tools/__init__.py

"""
Graphs Networks Tools:
- Fractal dimension of graphs (CPU + GPU)
- Scale-free exponent / degree exponent (degree dimension)
- Bond percolation visualisation
"""

from .cpu_compute_fractal_dimension import cpu_compute_fractal_dimension

# GPU version is optional: if import fails, keep a None placeholder
try:
    from .gpu_compute_fractal_dimension import gpu_compute_fractal_dimension
except Exception:
    gpu_compute_fractal_dimension = None

from .compute_scale_free_exponent import compute_scale_free_exponent
from .draw_bond_percolation import draw_bond_percolation

__all__ = [
    "cpu_compute_fractal_dimension",
    "gpu_compute_fractal_dimension",
    "compute_scale_free_exponent",
    "draw_bond_percolation",
]
