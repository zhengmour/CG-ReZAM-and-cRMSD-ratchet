"""Build with the same Python/MPI as LAMMPS; set EIGEN_INCLUDE_DIR and MPI_HOME."""
import os
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

prefix = Path(os.environ.get('CONDA_PREFIX', os.sys.prefix))
candidates = [os.environ.get('EIGEN_INCLUDE_DIR', ''), prefix / 'include/eigen3',
              prefix / 'include', '/usr/include/eigen3', '/usr/local/include/eigen3']
eigen = next((str(p) for p in candidates if p and (Path(p) / 'Eigen/Core').exists()), None)
if eigen is None:
    raise RuntimeError('Set EIGEN_INCLUDE_DIR to the directory containing Eigen/Core')
mpi = Path(os.environ.get('MPI_HOME', str(prefix)))
setup(name='C_PIRMSD', ext_modules=[Pybind11Extension(
    'C_PIRMSD', ['C_PIRMSD.cpp'], include_dirs=[eigen, str(mpi / 'include')],
    libraries=['mpi'], library_dirs=[str(mpi / 'lib')],
    runtime_library_dirs=['$ORIGIN'], cxx_std=17,
    extra_compile_args=['-O3'],
)], cmdclass={'build_ext': build_ext}, zip_safe=False, python_requires='>=3.8')
