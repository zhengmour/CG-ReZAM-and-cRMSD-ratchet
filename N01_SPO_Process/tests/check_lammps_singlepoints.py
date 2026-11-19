"""Actual LAMMPS LJ dimer test of the new QM-evaluation/readout pipeline."""
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
from solve_CG import build_manifest
from evaluate_qm_batch import evaluate_sample
from spo_loss import qm_losses

p=argparse.ArgumentParser();p.add_argument('--lammps',required=True);args=p.parse_args()
with tempfile.TemporaryDirectory(prefix='spo real LAMMPS with spaces ') as tmp:
 root=Path(tmp);reference=root/'reference';template=root/'lmp_qm.in'
 template.write_text('''units real
atom_style full
atom_modify map yes
read_data ${datafile}.data
mass * 28
pair_style lj/cut 12
pair_coeff * * 0.5 2.0
thermo_style custom step pe epair
thermo_modify format float %.16g
dump forces all custom 1 ${name}.lammpstrj id type x y z fx fy fz
dump_modify forces sort id format float %.16g
run 0 post no
''')
 for i,distance in enumerate([3.,4.,5.]):
  folder=reference/'Si-Na'/str(i);folder.mkdir(parents=True)
  energy=4*.5*((2/distance)**12-(2/distance)**6)
  force=24*.5/distance*((2/distance)**6-2*(2/distance)**12)
  (folder/(str(i)+'.data')).write_text(f'''LJ test

2 atoms
1 atom types

-20 30 xlo xhi
-20 30 ylo yhi
-20 30 zlo zhi

Atoms # full

1 1 1 0 0 0 0
2 2 1 0 {distance} 0 0
''')
  (folder/(str(i)+'_energy.xyz')).write_text(str(energy+17.))
  (folder/(str(i)+'_force.xyz')).write_text(f'2\nForces\n1 {force} 0 0\n1 {-force} 0 0\n')
 samples=build_manifest(reference,root/'dataset.json')
 try:
  predicted={s['id']:evaluate_sample(s,template,args.lammps,root/'results',(3,)) for s in samples}
 except Exception:
  for logfile in (root/'results').rglob('*.log'):
   print(str(logfile),logfile.read_text()[-5000:],file=sys.stderr)
  raise
 losses=qm_losses(predicted,{s['id']:s for s in samples})
 np.testing.assert_allclose(losses,0,atol=1e-12)
 print(json.dumps(dict(samples=len(samples),energy_rmse=float(losses[0]),force_rmse=float(losses[1]),zero_loss=float(losses[2]),paths_with_spaces=True)))
