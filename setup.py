import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))


setup(
    name="mini_dpvo",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "cuda_corr",
            sources=[
                "mini_dpvo/altcorr/correlation.cpp",
                "mini_dpvo/altcorr/correlation_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
        CUDAExtension(
            "cuda_ba",
            sources=["mini_dpvo/fastba/ba.cpp", "mini_dpvo/fastba/ba_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
        CUDAExtension(
            "lietorch_backends",
            include_dirs=[
                osp.join(ROOT, "mini_dpvo/lietorch/include"),
                osp.join(ROOT, "thirdparty/eigen-3.4.0"),
            ],
            sources=[
                "mini_dpvo/lietorch/src/lietorch.cpp",
                "mini_dpvo/lietorch/src/lietorch_gpu.cu",
                "mini_dpvo/lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
