# import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

# # Read the EIGEN_INCLUDE_DIR environment variable
# eigen_include_dir = os.environ.get("EIGEN_INCLUDE_DIR")
# if eigen_include_dir is None:
#     raise EnvironmentError(
#         "EIGEN_INCLUDE_DIR environment variable is not set. "
#         "Please set it in your .pixi.sh to the path of Eigen's include directory."
#     )


setup(
    name="mini_dpvo",
    packages=find_packages(include=["mini_dpvo", "mini_dpvo.*"]),
    package_data={
        # Include all header files in the 'include' directories
        "mini_dpvo.lietorch": ["include/*.h"],
    },
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
                # eigen_include_dir,  # Use the environment variable here
                osp.join(ROOT, ".pixi/envs/default/include/eigen3"),
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
