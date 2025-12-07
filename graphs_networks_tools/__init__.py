"""
Graphs Networks Tools

Main features (public API):

- Fractal dimension of graphs (box-counting / box dimension)
    * cpu_compute_fractal_dimension
    * gpu_compute_fractal_dimension           (None if GPU backend not available)

- Scale-free exponent / degree dimension
    * compute_degree_dimension            (preferred name)
    * compute_degree_dimension               (backwards-compatible alias)

- Percolation and IGS visualisation (optional, if the module is present)
    * draw_bond_percolation
    * plot_percolation_curve
    * plot_IGS_and_percolation
"""

# ------------------------
# Fractal / box dimension
# ------------------------

from .cpu_compute_fractal_dimension import cpu_compute_fractal_dimension

# Backwards-compatible alias for the old README naming
cpu_compute_box_dimension = cpu_compute_fractal_dimension

try:
    from .gpu_compute_fractal_dimension import gpu_compute_fractal_dimension
except Exception:
    # If the GPU module or its dependencies are not available,
    # expose a None placeholder rather than failing at import time.
    gpu_compute_fractal_dimension = None

# Backwards-compatible alias
gpu_compute_box_dimension = gpu_compute_fractal_dimension


# ---------------------------------------------
# Scale-free exponent / degree dimension tools
# ---------------------------------------------

# Newer naming convention: compute_scale_free_exponent
# Older naming convention: compute_degree_dimension
try:
    # Preferred: a module that already defines both names
    from . compute_degree_dimension  import (
         compute_degree_dimension,
        compute_degree_dimension,
    )
except ImportError:
    # Fallback: older module name or API, where only compute_degree_dimension exists
    try:
        from . compute_degree_dimension  import compute_degree_dimension
    except ImportError:
        # If neither module is present, expose placeholders
        def compute_degree_dimension(*args, **kwargs):
            raise ImportError(
                "No scale-free exponent module found "
                "(expected 'compute_scale_free_exponent.py' or 'scale_free_exponent.py')."
            )

    # In this fallback case, we treat degree dimension as the scale-free exponent
    compute_degree_dimension = compute_degree_dimension


# --------------------------------------
# Percolation / IGS visualisation tools
# --------------------------------------

# These are optional: if you put your functions
# draw_bond_percolation, plot_percolation_curve, plot_IGS_and_percolation
# into a file like `percolation_tools.py`, this block will expose them.
try:
    from .percolation_tools import (
        draw_bond_percolation,
        plot_percolation_curve,
        plot_IGS_and_percolation,
    )
except ImportError:
    # If the user has not added percolation_tools.py yet,
    # we simply do not expose these names (and do not fail import).
    draw_bond_percolation = None
    plot_percolation_curve = None
    plot_IGS_and_percolation = None


__all__ = [
    # Fractal / box dimension
    "cpu_compute_fractal_dimension",
    "gpu_compute_fractal_dimension",
    "cpu_compute_box_dimension",
    "gpu_compute_box_dimension",

    # Scale-free / degree dimension
    "compute_degree_dimension",

    # Percolation / IGS (optional)
    "draw_bond_percolation",
    "plot_percolation_curve",
    "plot_IGS_and_percolation",
]
