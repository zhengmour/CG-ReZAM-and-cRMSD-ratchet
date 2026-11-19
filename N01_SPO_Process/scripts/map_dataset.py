"""Batch AA-to-CG mapping with config-rooted paths and an explicit failure list."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workflow import load_workflow
from map_QM_to_CG import convert_sample
import re


def worker(task):
    try:return str(task[0]),str(convert_sample(*task)),None
    except Exception as exc:return str(task[0]),None,str(exc)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--list',type=Path)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--failed',type=Path);a=p.parse_args();cfg=load_workflow(a.config)
    root=Path(cfg['paths']['qm_source']);out=Path(cfg['paths']['cg_reference'])
    directories=[root/line.strip() for line in a.list.read_text().splitlines() if line.strip()] if a.list else sorted({p.parent for p in root.rglob('*.log')})
    if not directories:raise ValueError('No Gaussian sample directories')
    bonds={int(m[0]):float(m[1]) for m in re.findall(r'^bond_coeff\s+(\d+)\s+\S+\s+([0-9.]+)',(Path(cfg['paths']['templates'])/'lmp_qm.in').read_text(),re.M)}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(worker,[(d,root,out,bonds[1],bonds[2]-bonds[1]) for d in directories]))
    failures=[dict(sample=d,error=e) for d,_,e in results if e]
    failed=a.failed or out/'mapping_failures.json';failed.parent.mkdir(parents=True,exist_ok=True);failed.write_text(json.dumps(failures,indent=2)+'\n')
    print('Mapped',len(results)-len(failures),'of',len(results),'samples; failures:',failed)
    return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
