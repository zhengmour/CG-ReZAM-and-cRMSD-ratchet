# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os

# Functions to find Eigen paths
def find_eigen_path():
    possible_paths = [
        # Manual installation path
        "path to eigen/3.4.0/include/eigen3",
        # Conda environment path
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include'),
        'path to anaconda3/envs/lmpEnv/include',
        '/usr/include/eigen3',
        '/usr/local/include/eigen3', 
        '/usr/include',
        '/usr/local/include',
    ]
    
    for path in possible_paths:
        eigen_header = os.path.join(path, 'Eigen', 'Core')
        if os.path.exists(eigen_header):
            return path
    
    # If it can't be found, return the include path of the conda environment
    return os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include')

# Define extension modules
ext_modules = [
    Pybind11Extension(
        "C_PIRMSD",
        [
            "C_PIRMSD.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            find_eigen_path(),
            "/gpfs/home/gauss/zhengda/opt/mpich/4.3.0/include",
        ],
        libraries=[
            "mpi"
        ],
        library_dirs=[
            "/gpfs/home/gauss/zhengda/opt/mpich/4.3.0/lib"
        ],
        language='c++'
    ),
]

setup(
    name="C_PIRMSD",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

# Using python setup.py build_ext --inplace