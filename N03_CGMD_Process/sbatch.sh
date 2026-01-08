#!/bin/bash  
    
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1 

packmol < packmol_water.inp

gmx_mpi editconf -f init.pdb -o init.gro -box 17.0 17.0 17.0
echo -e "a Si\n! r SIV\nq\n" | gmx_mpi make_ndx -f init.gro -o si_water.ndx
sed -i "s/!SIV/notSIV/g" si_water.ndx

gmx_mpi grompp -f em_cg.mdp -c init.gro -r init.gro -n si_water.ndx -p topol_water.top -o em_cg.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm em_cg -ntomp 64
test $? -gt 0 && exit

gmx_mpi grompp -f em2_cg.mdp -c em_cg.gro -r em_cg.gro -n si_water.ndx -p topol_water.top -o em2_cg.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm em2_cg -ntomp 64
test $? -gt 0 && exit

gmx_mpi grompp -f eq_cg.mdp -c em2_cg.gro -r em2_cg.gro -n si_water.ndx -p topol_water.top -o eq_cg.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm eq_cg -ntomp 64 
test $? -gt 0 && exit

gmx_mpi grompp -f eq2_cg.mdp -c eq_cg.gro -r eq_cg.gro -n si_water.ndx -p topol_water.top -o eq2_cg.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm eq2_cg -ntomp 64 
test $? -gt 0 && exit

gmx_mpi grompp -f NPT.mdp -c eq2_cg.gro -r eq2_cg.gro -n si_water.ndx -p topol_water.top -o NPT.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm NPT -ntomp 64 
test $? -gt 0 && exit

gmx_mpi grompp -f NVT.mdp -c NPT.gro -r NPT.gro -n si_water.ndx -p topol_water.top -o NVT.tpr -maxwarn 10
gmx_mpi mdrun -v -deffnm NVT -ntomp 64 
test $? -gt 0 && exit

