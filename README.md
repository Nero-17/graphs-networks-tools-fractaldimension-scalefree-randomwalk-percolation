# Graphs Networks Tools: Fractal Dimension/ Scale-Free/ Random Walk/ Percolation

This repository provides tools to estimate the **box-counting (fractal) dimension** of graphs, using a greedy box-covering algorithm in the graph metric.

Two implementations are available:

- `cpu_compute_box_dimension` – a pure NetworkX + NumPy/Scikit-learn implementation.
- `gpu_compute_box_dimension` – a prototype GPU implementation using PyTorch sparse operations.

Both functions have the **same mathematical logic**; the GPU version only tries to accelerate the “balls in the graph” computations.

---

## Installation

At the moment the package is distributed directly from GitHub.

```bash
pip install git+https://github.com/Nero-17/fractal-dimension-of-graphs.git
