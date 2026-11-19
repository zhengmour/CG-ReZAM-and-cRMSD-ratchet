"""Run CG single points on mapped QM structures; each sample has its own directory."""
import argparse,hashlib,json,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from energy_and_force_diff import read_energy,force_squared_error


def evaluate_sample(sample,input_file,executable,output,excluded):
    for key in ('data','forces'):
        if hashlib.sha256(Path(sample[key]).read_bytes()).hexdigest()!=sample['sha256'][key]:
            raise ValueError('Reference sample changed after indexing: '+sample['id'])
    folder=Path(output).resolve()/'samples'/hashlib.sha256(sample['id'].encode()).hexdigest()[:20]
    folder.mkdir(parents=True,exist_ok=True)
    for mol in Path(input_file).parent.glob('*.mol'):shutil.copy2(mol,folder/mol.name)
    name=folder/'singlepoint'
    # LAMMPS suppresses variable substitution inside quotes. Keep its variables
    # space-free by referencing a local link, while subprocess handles full paths.
    link=folder/'reference.data'
    if link.is_symlink():link.unlink()
    link.symlink_to(Path(sample['data']).resolve())
    command=[executable,'-in',str(Path(input_file).resolve()),'-var','datafile','reference',
             '-var','name','singlepoint','-log',str(folder/'lammps.log'),'-screen','none']
    with (folder/'stdout.log').open('w') as stream:
        subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,check=True)
    sq,n=force_squared_error(str(name)+'.lammpstrj',sample['forces'],excluded)
    result=dict(id=sample['id'],energy=read_energy(folder/'lammps.log'),force_squared_error=sq,force_sites=n)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','input','output'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--executable',required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--batch-index',type=int,default=0);p.add_argument('--batch-count',type=int,default=1)
    p.add_argument('--exclude-types',default='3');a=p.parse_args()
    samples=json.loads(a.manifest.read_text())['samples'][a.batch_index::a.batch_count]
    excluded=tuple(int(v) for v in a.exclude_types.split(',') if v)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(lambda s:evaluate_sample(s,a.input,a.executable,a.output,excluded),samples))
    out=a.output/('batch_%04d.json'%a.batch_index);temp=out.with_suffix('.tmp')
    temp.write_text(json.dumps(results,indent=2)+'\n');temp.replace(out)
if __name__=='__main__':main()
