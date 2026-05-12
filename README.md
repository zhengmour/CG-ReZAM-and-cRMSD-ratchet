# CG-ReZAM-and-cRMSD-ratchet-scheme

Coarse-grained reactive molecular dynamics and enhanced-sampling workflows for MFI zeolite crystallization, including parameter fitting, CGMD production, cRMSD-based ratchet sampling, and topology analysis.

## Overview

This repository contains the main computational workflow used for:

- coarse-grained force-field parameter fitting against QM and structural targets
- GA-based parameter search for CGMD
- CGMD production setup and batch execution
- cRMSD / PIRMSD enhanced sampling for crystallization trajectories
- post-processing and topology analysis of simulation outputs

The codebase is organized as a workflow repository rather than a standalone Python package. Most modules are intended to be run as scripts inside an HPC environment.

## Repository Layout

- `N01_GA_Process/`: GA-based parameter optimization for CGMD targets
- `N02_SPO_Process/`: SPO / parameter optimization workflow against QM and MD-derived losses
- `N03_CGMD_Process/`: CGMD input files and batch scripts
- `N04_cRMSD_Process/`: PIRMSD / cRMSD enhanced-sampling workflow, LAMMPS inputs, and C++ extension
- `N05_analysis_data/`: trajectory reading, topology construction, reaction counting, plotting, and summary export

## Environment Requirements

This repository mixes Python, C++, shell, and HPC batch workflows. A full run does not rely on a single lightweight environment.

### Core languages and tools

- `Python 3.8+`
- `C++ compiler` with C++11-or-newer support
- `bash`
- `LAMMPS`
- `MPI` runtime and development headers
- `Slurm` for job submission in the provided workflow scripts

### Python dependencies

The following libraries are imported by the current modules in this repository:

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `MDAnalysis`
- `networkx`
- `scikit-learn`
- `mpi4py`
- `pybind11`
- `setuptools`
- `dtaidistance`
- `scikit-opt`
- `shape-similarity`

### Native / external scientific dependencies

- `LAMMPS Python module`
  Required by `N04_cRMSD_Process/PIRMSD.py`.
- `Eigen`
  Required to build the `C_PIRMSD` extension in `N04_cRMSD_Process/setup.py`.
- `MPI library`
  Required both for `mpi4py` and for compiling / running the PIRMSD extension.
- `PLUMED`
  Referenced by `N04_cRMSD_Process/sbatch.sh` for enhanced-sampling runs.
- `GROMACS`
  Referenced by the GA workflow and some CGMD comparison steps.

## Recommended Installation

For analysis-only usage, a Python environment is sufficient. For the full workflow, you will also need a working HPC stack with LAMMPS, MPI, and Slurm.

Example Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib tqdm MDAnalysis networkx scikit-learn mpi4py pybind11 setuptools dtaidistance scikit-opt shape-similarity
```

If you need the cRMSD / PIRMSD workflow, make sure the active environment can also import:

- `lammps`
- `mpi4py`

and that your compiler can find:

- `Eigen`
- `MPI` headers and libraries

## Building the PIRMSD Extension

`N04_cRMSD_Process/` includes a compiled extension, `C_PIRMSD`, used by the enhanced-sampling workflow.

Build it in place:

```bash
cd N04_cRMSD_Process
python3 setup.py build_ext --inplace
```

Notes:

- `setup.py` currently contains machine-specific MPI include and library paths.
- You will likely need to edit `N04_cRMSD_Process/setup.py` before building on a different cluster or workstation.
- `Eigen` must be installed and discoverable by the compiler.

## Workflow Summary

### 1. GA parameter search

Main entry:

```bash
cd N01_GA_Process
python3 GA_main.py 20 50 64 6 ./workspace ./scripts ./optimize.log
```

This stage uses:

- `scikit-opt`
- `shape-similarity`
- `MDAnalysis`
- external simulation commands submitted from the workflow

### 2. SPO / parameter optimization

Main entry:

```bash
cd N02_SPO_Process
python3 optimize_main.py 64 1
```

This stage orchestrates:

- parameter generation
- QM-to-CG mapping
- force / energy comparisons
- Slurm job submission
- MD and QM loss aggregation

Related helper scripts live under `N02_SPO_Process/scripts/`.

### 3. CGMD production

`N03_CGMD_Process/` contains prepared topology, coordinate, and `.mdp` files together with a batch script for production or equilibration runs.

Main files include:

- `sbatch.sh`
- `system.itp`
- `*.mdp`
- `*.itp`
- `*.pdb`

### 4. cRMSD / PIRMSD enhanced sampling

Main components:

- `N04_cRMSD_Process/PIRMSD.py`
- `N04_cRMSD_Process/C_PIRMSD.cpp`
- `N04_cRMSD_Process/sbatch.sh`
- `N04_cRMSD_Process/lmp.*.in`

This workflow depends on:

- `LAMMPS`
- `mpi4py`
- `lammps` Python module
- compiled `C_PIRMSD`
- Slurm
- optionally `PLUMED`, depending on the batch configuration

### 5. Trajectory and topology analysis

Main entry:

```bash
cd N05_analysis_data
python3 silicate_analysis.py <file_type> <n_processes> <traj_path>
```

Supported readers include trajectories or structures based on:

- `.gro`
- `.xtc`
- LAMMPS `.data`
- LAMMPS `.dump` / `.lammpstrj`
- PIRMSD trajectory directories

Typical outputs:

- `Analysis__Qns.csv`
- `Analysis__Linear.csv`
- `Analysis__Branch.csv`
- `Analysis__Cyclic.csv`
- `Analysis__Rings.csv`
- `Analysis__Cluster.csv`
- `Analysis__reactions.csv`
- plot images generated by the plotting utilities

## HPC Assumptions

Several scripts assume an HPC cluster environment with:

- `sbatch`
- `squeue`
- `mpirun`
- cluster-specific module or environment setup

The provided batch scripts also contain placeholder paths such as:

- `<Path to Plumed>`
- `<Path to LAMMPS>`

These must be replaced with real installation paths for your machine.

## Reproducibility Notes

- This repository is workflow-oriented and contains hard-coded HPC assumptions in some scripts.
- Native extension build settings are not yet portable across clusters.
- Some scripts assume specific file naming conventions for trajectories and restart folders.
- Before running production jobs, review all `sbatch.sh`, `setup.py`, and input templates for local path, MPI, and scheduler compatibility.

## Citation

If you find this work useful, please cite:

Zheng, D.; Wang, J.; Cui, H.; Zhang, D.; Li, G. Combining Coarse-Grained Reactive Molecular Dynamics with an Enhanced-Sampling Method for MFI Zeolite Crystallization. *J. Chem. Theory Comput.* 2026, 22(3): 1409-1425. https://doi.org/10.1021/acs.jctc.5c01745
