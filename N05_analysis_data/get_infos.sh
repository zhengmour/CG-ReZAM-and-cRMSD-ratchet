###################################################################
#       get_infos.sh xtc -> CGMD                                  #
#       get_infos.sh rmsd_lammpstrj -> Enhance sampling           #
###################################################################

#!/bin/bash

#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1

export OMPI_MCA_btl_openib_warn_no_device_params_found=0

export PLUMED2_HOME=<Path to Plumed>
export PLUMED_ROOT=$PLUMED2_HOME
export PATH=$PLUMED2_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PLUMED2_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PLUMED2_HOME/lib:$LIBRARY_PATH

LAMMPS=<Path to LAMMPS>
export PATH=$LAMMPS/src:$PATH
export LD_LIBRARY_PATH=$LAMMPS/src:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LAMMPS/src:$LIBRARY_PATH
export LD_PRELOAD=$PLUMED2_HOME/lib/libplumed.so:$PLUMED2_HOME/lib/libplumedKernel.so

conda activate lmpEnv

python silicate_analysis.py $1 32
