#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1

DEFAULT_INIT_TIME=0.0
DEFAULT_INIT_ITER=0

# Set up the base environment
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

# READY complete environment settings
export PLUMED2_HOME=<Path to Plumed>
export PLUMED_ROOT=$PLUMED2_HOME
export PATH=$PLUMED2_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PLUMED2_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PLUMED2_HOME/lib:$LIBRARY_PATH

# LAMMPS environment settings
LAMMPS=<Path to LAMMPS>
export PATH=$LAMMPS/src:$PATH
export LD_LIBRARY_PATH=$LAMMPS/src:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LAMMPS/src:$LIBRARY_PATH

# Preloaded PLUMED libraries
export LD_PRELOAD=$PLUMED2_HOME/lib/libplumed.so:$PLUMED2_HOME/lib/libplumedKernel.so

cd $SLURM_SUBMIT_DIR

# Handle init_time
max_time=$(find . -maxdepth 1 -type f -name "lmp_*.lammpstrj" | sed -n 's/.*lmp_\([0-9]*\.[0-9]*\)\.lammpstrj/\1/p' | sort -nr | head -n1)

# Handle init_iter
max_iter_PIRMSD=$(find . -maxdepth 1 -type d -name "PIRMSD_*" | sed -n 's|.*/PIRMSD_\([0-9]*\)|\1|p' | sort -nr | head -n1)
max_iter_COLVAR=$(find . -maxdepth 1 -type f -name "COLVAR.*" | sed -n 's|.*/COLVAR.\([0-9]*\)|\1|p' | sort -nr | head -n1)
max_iter=$(( max_iter_PIRMSD > max_iter_COLVAR ? max_iter_PIRMSD : max_iter_COLVAR ))

niters=$(python3 -c "import json; print(json.load(open('params.json'))['Simulation_Parameters']['niterations'])")
echo "${niters} cycle need to process"

# Check if the value is found, otherwise use the default
if [ -z "$max_time" ]; then
  echo "No lmp_*.lammpstrj found. Using default init_time."
  max_time=$DEFAULT_INIT_TIME
  # Generate a random input structure
  mpirun -np 64 lmp_mpi -in lmp.create.in -var seed $(date +%s)
fi

if [ -z "$max_iter" ]; then
  echo "No PIRMSD_* directories found. Using default init_iter."
  max_iter=$DEFAULT_INIT_ITER
fi

# Modify the params.json file
if [ -f "params.json" ]; then
  echo "Updating params.json with init_time = $max_time, init_iter = $max_iter"
  sed -i "s/\"init_time\"[[:space:]]*:[[:space:]]*[0-9\.]*/\"init_time\" : $max_time/" params.json
  sed -i "s/\"init_iter\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_iter\" : $max_iter/" params.json
else
  echo "params.json not found!"
  exit 1
fi

max_time=$(echo ${max_time} + 0.05*0.5 | bc -l)
max_time=$(printf "%.4f" "$max_time")
sed -i "s/lmp_.*.lammpstrj/lmp_${max_time}.lammpstrj/g" lmp.begin.in
mpirun -np  $SLURM_NTASKS lmp_mpi -in lmp.begin.in -log log.begin.lammps
test $? -gt 0 && echo "Error in Begin process" && exit
sed -i "s/\"init_time\"[[:space:]]*:[[:space:]]*[0-9\.]*/\"init_time\" : $max_time/" params.json

iter=$max_iter
echo "[${iter}/${niters}] Begin process"
while [ ${iter} -le ${niters} ] 
do
  sed -i "s/\"init_iter\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_iter\" : $iter/" params.json
  max_time=$(find . -maxdepth 1 -type f -name "lmp_*.lammpstrj" | sed -n 's/.*lmp_\([0-9]*\.[0-9]*\)\.lammpstrj/\1/p' | sort -nr | head -n1)
  sed -i "s/\"init_time\"[[:space:]]*:[[:space:]]*[0-9\.]*/\"init_time\" : $max_time/" params.json

  # create input lmp.in and params.json
  python EnhanceSampling.py   

  # Enhance porcess
  if [ -f "pirmsd_${iter}.json" ]; then
    echo "[${iter}/${niters}] PIRMSD process"
    mpirun -np $SLURM_NTASKS  python PIRMSD.py pirmsd_${iter}.json
    test $? -gt 0 && echo "Error in PIRMSD process" && exit
    # Fixed SI and Relaxation
    mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.relax.in -log log.relax.lammps
    test $? -gt 0 && echo "Error in Relax process" && exit
    rm pirmsd_${iter}.json lmp_${iter}.relax.in
  fi

  # MD process
  echo "[${iter}/${niters}] CGMD process"
  mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.md.in -log log.md.lammps
  test $? -gt 0 && echo "Error in CGMD process" && exit
  rm lmp_${iter}.md.in

  dir="PIRMSD_${iter}"
  logfile="${dir}/history.log"
  threshold=0.5

  lastline=$(tail -n 1 "$logfile")
  echo "[$(date +'%F %T')] read: $logfile -> \"$lastline\""
  rmsd=$(echo "$lastline" | awk -F'RMSD is ' '{print $2}' | awk '{print $1}')
  is_lt=$(awk -v r="$rmsd" -v t="$threshold" 'BEGIN{print (r < t) ? 1 : 0}')

  if [[ "$is_lt" -eq 1 ]]; then
    echo "RMSD <$threshold, END."
    break
  fi

  iter=$((iter + 1))
done
