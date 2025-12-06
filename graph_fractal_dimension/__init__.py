from .box_dimension import cpu_compute_box_dimension

try:
    from .box_dimension_gpu import gpu_compute_box_dimension
except Exception:
    # GPU dependencies (torch, CUDA) might be missing.
    gpu_compute_box_dimension = None

__all__ = ["cpu_compute_box_dimension", "gpu_compute_box_dimension"]
