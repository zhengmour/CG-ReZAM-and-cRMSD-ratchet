#!/bin/bash  

#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1 

if [ $# -ne 9 ]
then
echo "xxxxxxxxxxxxxxxxxxxx  INPUT ERROR  xxxxxxxxxxxxxxxxxxxxxxx"
echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
echo "                                              "
echo " Usage : run <sigma> <epsilon> <sigma> <epsilon> <dVS> <repulsion> <lambda> <dir> <Ntomp>"
echo "..........................................................."
exit
fi

SIGMA_Q=$1    # Q-Q
EPSILON_Q=$2
SIGMA_D=$3    # D-D
EPSILON_D=$4
D_VS=$5
REP=$6
LAMBDA=$7
DIR=$8
Ntomp=$9

RMIN_QSI=$(awk "BEGIN {print $SIGMA_Q*2^(1/6)}") 
RMIN=$(awk "BEGIN {print $SIGMA_D*2^(1/6)}")
D_Q_LENGTH=$(awk "BEGIN {print ($RMIN_QSI-$RMIN)/(2*0.9724) }")
D_VS_Q=$(awk "BEGIN {print $D_Q_LENGTH+$D_VS}")

C12_Q=$(awk "BEGIN {print 4*$EPSILON_Q*($SIGMA_Q^12)}")
C6_Q=$(awk "BEGIN {print 4*$EPSILON_Q*($SIGMA_Q^6)}")

C12_D=$(awk "BEGIN {print 4*$EPSILON_D*($SIGMA_D^12)}")
C6_D=$(awk "BEGIN {print 4*$EPSILON_D*($SIGMA_D^6)}")
D_D=$(awk -v OFMT='%.3f' "BEGIN {
    d1 = $D_Q_LENGTH / cos(35.25 * 3.14159 / 180)
    d2 = $D_Q_LENGTH * cos(70.5 * 3.14159 / 180)
    d3 = d2 / cos(35.25 * 3.14159 / 180)
    d_d_length = d1 + d3; print d_d_length}" | cut -c -5)

RUN=$DIR/"sigQ_"$SIGMA_Q"_epsQ_"$EPSILON_Q"_sigD_"$SIGMA_D"_epsD_"$EPSILON_D"_dVS_"$D_VS"_rep_"$REP"_lambda_"$LAMBDA
FILES=$DIR/files
cp $FILES/* $RUN/

# Replace stell8_mono.pdb
python `pwd`/create_particle.py $SIGMA_Q $SIGMA_D $D_VS $RUN

# Modify gene.itp
sed -i "4s/.*/#define C12_Q_Q     $C12_Q /"     $RUN/gene.itp
sed -i "5s/.*/#define C6_Q_Q      $C6_Q /"      $RUN/gene.itp
sed -i "28s/.*/#define C12_D_D     $C12_D /"     $RUN/gene.itp
sed -i "29s/.*/#define C6_D_D      $C6_D /"      $RUN/gene.itp
sed -i "21s/.*/#define D_D_LENGTH  $D_D /"       $RUN/gene.itp
sed -i "20s/.*/#define D_Q_LENGTH  $D_Q_LENGTH/" $RUN/gene.itp
sed -i "39s/.*/#define VW_LENGTH   $D_VS_Q/"     $RUN/gene.itp
sed -i "40s/.*/#define VX_LENGTH   $D_VS_Q/"     $RUN/gene.itp
sed -i "41s/.*/#define VY_LENGTH   $D_VS_Q/"     $RUN/gene.itp
sed -i "42s/.*/#define VZ_LENGTH   $D_VS_Q/"     $RUN/gene.itp
sed -i "49s/.*/#define VS_D_REP    $REP/"        $RUN/gene.itp

awk -v lam="$LAMBDA" '
NR == 34 || NR == 36 { 
    $4 = $4 * lam; 
    $5 = $5 * lam 
} 
{ print }
' $RUN/system.itp > $RUN/tmp_file && mv $RUN/tmp_file $RUN/system.itp

MIN=$RUN"/em_cg.mdp"
MIN2=$RUN"/em2_cg.mdp"
EQL=$RUN"/eq2_cg.mdp"
NPT=$RUN"/NPT.mdp"
NVT=$RUN"/NVT.mdp"

current_path=`pwd`
cd $RUN
~/opt/packmol/packmol < packmol.inp
cd $current_path
gmx_mpi -nobackup -nocopyright editconf -f $RUN/init.pdb -o $RUN/init.gro -box 10.0 10.0 10.0
echo -e "a Si\n! r SIV\nq\n" | gmx_mpi make_ndx -f $RUN/init.gro -o $RUN/si_water.ndx
sed -i "s/!SIV/notSIV/g" $RUN/si_water.ndx

gmx_mpi -nobackup -nocopyright grompp -f $MIN -c $RUN/init.gro -n $RUN/si_water.ndx -p $RUN/topol_water.top -o $RUN/em_cg.tpr -maxwarn 10 || exit 1
gmx_mpi -nobackup -nocopyright mdrun -deffnm $RUN/em_cg -ntomp $Ntomp || exit 1

gmx_mpi -nobackup -nocopyright grompp -f $MIN2 -c $RUN/em_cg.gro -n $RUN/si_water.ndx -p $RUN/topol_water.top -o $RUN/em2_cg.tpr -maxwarn 10 || exit 1
gmx_mpi -nobackup -nocopyright mdrun -deffnm $RUN/em2_cg -ntomp $Ntomp || exit 1

gmx_mpi -nobackup -nocopyright grompp -f $EQL -c $RUN/em2_cg.gro -n $RUN/si_water.ndx -p $RUN/topol_water.top -o $RUN/eq_cg.tpr -maxwarn 10 || exit 1
gmx_mpi -nobackup -nocopyright mdrun -deffnm $RUN/eq_cg -ntomp $Ntomp || exit 1

gmx_mpi -nobackup -nocopyright grompp -f $NPT -c $RUN/eq_cg.gro -t $RUN/eq_cg.cpt -n $RUN/si_water.ndx -p $RUN/topol_water.top -o $RUN/NPT.tpr -maxwarn 10 || exit 1
gmx_mpi -nobackup -nocopyright mdrun -deffnm $RUN/NPT -ntomp $Ntomp || exit 1

gmx_mpi -nobackup -nocopyright grompp -f $NVT -c $RUN/NPT.gro -t $RUN/NPT.cpt -n $RUN/si_water.ndx -p $RUN/topol_water.top -o $RUN/NVT.tpr -maxwarn 10 || exit 1
gmx_mpi -nobackup -nocopyright mdrun -deffnm $RUN/NVT -ntomp $Ntomp || exit 1

# Merge trajectory
if [ -e $RUN/NVT.xtc ]; then
    printf "%f\n%f\n%f\n" 0.000 1000.0 2000.0 | gmx_mpi trjcat -f $RUN/eq_cg.xtc $RUN/NPT.xtc $RUN/NVT.xtc -o $RUN/md_cg.xtc -settime
    echo 2 | gmx_mpi trjconv -f $RUN/md_cg.xtc -s $RUN/NVT.tpr -o $RUN/md_cg_si.xtc
    echo 2 | gmx_mpi trjconv -f $RUN/init.gro -s $RUN/NVT.tpr -o $RUN/md_cg_si.gro
fi
rm $RUN/NVT.xtc $RUN/NVT.trr $RUN/NPT.xtc $RUN/NPT.trr $RUN/md_cg.xtc
