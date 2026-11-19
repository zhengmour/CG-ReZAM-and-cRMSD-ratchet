"""Create a deterministic dataset manifest; no working-directory-dependent scans."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workflow import load_workflow
from energy_and_force_diff import read_reference_force_table
import numpy as np


def build_manifest(root,output):
    root=Path(root).resolve();samples=[]
    for energy_file in sorted(root.rglob('*_energy.xyz')):
        # Raw AA exports have a separate directory and are never fitting records.
        if 'aa_exports' in energy_file.parts:continue
        stem=energy_file.name[:-len('_energy.xyz')]
        data=energy_file.with_name(stem+'.data');force=energy_file.with_name(stem+'_force.xyz')
        if not data.is_file() or not force.is_file():raise ValueError('Incomplete mapped sample: '+str(energy_file))
        sample_id=energy_file.parent.relative_to(root).as_posix()
        if sample_id=='.':raise ValueError('Mapped samples must be grouped as family/sample')
        energy=float(energy_file.read_text().strip());table=read_reference_force_table(force)
        if not np.isfinite(energy):raise ValueError('Nonfinite reference energy')
        samples.append(dict(id=sample_id,family=sample_id.split('/')[0],energy=energy,
            data=str(data),forces=str(force),atom_count=len(table),sha256={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in [('data',data),('forces',force)]}))
    if not samples:raise ValueError('No mapped *_energy.xyz samples under '+str(root))
    if len({s['id'] for s in samples})!=len(samples):raise ValueError('Multiple mapped samples in one directory')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(version=1,units=dict(energy='kcal/mol',force='kcal/mol/angstrom'),samples=samples),indent=2)+'\n')
    return samples

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--cg-root',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();cfg=load_workflow(args.config)
    samples=build_manifest(args.cg_root or cfg['paths']['cg_reference'],args.output or cfg['paths']['dataset_manifest'])
    print('Indexed',len(samples),'samples')
