import glob
from setuptools import setup, Extension
import pybind11
import numpy as np

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

os.environ["CFLAGS"] = "-O0"
os.environ["CXXFLAGS"] = "-O0"

src_files = glob.glob('src/*.cpp')

# CPU 최적화 컴파일 플래그
cpp_extra_compile_args = [
    '-O3',
    '-march=native',
    '-ffast-math',
    # '-fopenmp', 
    '-std=c++17'
]
cpp_extra_link_args = [
    # '-fopenmp',
]

ext_modules = [
    Extension(
        name='custom_backend',
        sources=src_files,
        include_dirs=[
            'include',
            pybind11.get_include(),
            np.get_include()
        ],
        extra_compile_args=cpp_extra_compile_args,
        extra_link_args=cpp_extra_link_args,
        language='c++'
    )
]

setup(
    name='custom_backend',
    ext_modules=ext_modules,
)