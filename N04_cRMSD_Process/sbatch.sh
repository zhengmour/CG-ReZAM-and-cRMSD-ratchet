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

current_path=`pwd`
cat lmp.in

# Update lmp_*.lammpstrj timestamps to the correct values
bash ./correction_lmp_time.sh
bash ./rename_files.sh

# Handle init_time
max_time=$(find . -maxdepth 1 -type f -name "lmp_*.lammpstrj" | sed -n 's/.*lmp_\([0-9]*\.[0-9]*\)\.lammpstrj/\1/p' | sort -nr | head -n1)

# Handle init_iter
max_iter_PIRMSD=$(find . -maxdepth 1 -type d -name "PIRMSD_*" | sed -n 's|.*/PIRMSD_\([0-9]*\)|\1|p' | sort -nr | head -n1)
max_iter_COLVAR=$(find . -maxdepth 1 -type f -name "COLVAR.*" | sed -n 's|.*/COLVAR.\([0-9]*\)|\1|p' | sort -nr | head -n1)
max_iter=$(( max_iter_PIRMSD > max_iter_COLVAR ? max_iter_PIRMSD : max_iter_COLVAR ))

# Check whether a value was found; otherwise use the default
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

echo "[${max_iter}/1000] Begin process"
max_time=$(echo ${max_time} + 0.05*0.5 | bc -l)
max_time=$(printf "%.4f" "$max_time")
sed -i "s/lmp_.*.lammpstrj/lmp_${max_time}.lammpstrj/g" lmp.begin.in
mpirun -np  $SLURM_NTASKS lmp_mpi -in lmp.begin.in -log log.begin.lammps
test $? -gt 0 && echo "Error in Begin process" && exit
sed -i "s/\"init_time\"[[:space:]]*:[[:space:]]*[0-9\.]*/\"init_time\" : $max_time/" params.json

echo "startting kMC/MD process"
iter=$max_iter
while [ $iter -le 600 ]
do
  sed -i "s/\"init_iter\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_iter\" : $iter/" params.json
  max_time=$(find . -maxdepth 1 -type f -name "lmp_*.lammpstrj" | sed -n 's/.*lmp_\([0-9]*\.[0-9]*\)\.lammpstrj/\1/p' | sort -nr | head -n1)
  sed -i "s/\"init_time\"[[:space:]]*:[[:space:]]*[0-9\.]*/\"init_time\" : $max_time/" params.json

  echo "[${iter}/1000] kMC process"
  # kMC processvim p
  mpirun -np $SLURM_NTASKS python variableMD.py 
  test $? -gt 0 && echo "Error in kMC process" && exit

  if [ -f "pirmsd_${iter}.json" ]; then
    echo "[${iter}/1000] PIRMSD process"
    mpirun -np $SLURM_NTASKS  python PIRMSD.py pirmsd_${iter}.json
    test $? -gt 0 && echo "Error in PIRMSD process" && exit
    # Fixed SI and relaxation
    mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.relax.in -log log.relax.lammps
    test $? -gt 0 && echo "Error in Relax process" && exit
    rm pirmsd_${iter}.json lmp_${iter}.relax.in
  fi

  echo "[${iter}/1000] MD process"
  # MD process
  mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.md.in -log log.md.lammps
  test $? -gt 0 && echo "Error in MD process" && exit
  rm lmp_${iter}.md.in

  # Enhance porcess
  if [ -f "lmp_${iter}.annealing.in" ]; then
    echo "[${iter}/1000] Annealing process"
    mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.annealing.in -log log.annealing.lammps
    test $? -gt 0 && echo "Error in Annealing process" && exit
    rm lmp_${iter}.annealing.in
  fi

  if [ -f "lmp_${iter}.plumed.in" ]; then
    echo "[${iter}/1000] Plumed process"
    mpirun -np $SLURM_NTASKS  lmp_mpi -in lmp_${iter}.plumed.in -log log.plumed.lammps
    test $? -gt 0 && echo "Error in Plumed process" && exit
    mv COLVAR COLVAR.${iter}
    rm plumed.log lmp_${iter}.plumed.in
  fi

  dir="PIRMSD_${iter}"
  logfile="${dir}/history.log"
  threshold=0.0

  # Read the last line
  lastline=$(tail -n 1 "$logfile")
  echo "[$(date +'%F %T')] Read: $logfile -> \"$lastline\""

  # Parse RMSD: find the first number after "RMSD is "
  rmsd=$(echo "$lastline" | awk -F'RMSD is ' '{print $2}' | awk '{print $1}')

  # Use awk for floating-point comparison
  is_lt=$(awk -v r="$rmsd" -v t="$threshold" 'BEGIN{print (r < t) ? 1 : 0}')

  if [[ "$is_lt" -eq 1 ]]; then
    echo "RMSD <$threshold, stopping the loop."
    break
  fi

  iter=$((iter + 1))
done
