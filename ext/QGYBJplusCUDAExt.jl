"""
CUDA support for QGYBJplus: `GPU()` runs models on the current CUDA device.

Device arrays are `CuArray`s, kernels run on `CUDABackend`, and the horizontal
transforms use cuFFT through its AbstractFFTs plans. Select a device with
`CUDA.device!(n)` before building the grid when several are available.
"""
module QGYBJplusCUDAExt

using QGYBJplus
using CUDA

# Kernels launch on `get_backend(::CuArray)`, which is a default `CUDABackend()`.
QGYBJplus.default_gpu_backend() = CUDA.CUDABackend()

end
