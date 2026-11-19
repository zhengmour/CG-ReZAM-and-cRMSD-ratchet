"""Read named thermo columns and ID-indexed force dumps; fail on missing data."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_energy(logfile,column='PotEng'):
    header=None;value=None
    for line in Path(logfile).read_text().splitlines():
        words=line.split()
        if words and words[0]=='Step':
            header=words if column in words else None
            continue
        if header:
            if len(words)!=len(header):header=None;continue
            try:numbers=[float(w) for w in words]
            except ValueError:header=None;continue
            value=numbers[header.index(column)]
    if value is None or not np.isfinite(value):raise ValueError('No finite '+column+' thermo value in '+str(logfile))
    return value


def read_force_table(dump_file):
    lines=Path(dump_file).read_text().splitlines();last=None;i=0
    while i<len(lines):
        if lines[i]!='ITEM: TIMESTEP':raise ValueError('Unexpected dump content')
        n=int(lines[i+3]);fields=lines[i+8].split()[2:]
        if not all(k in fields for k in ('id','type','fx','fy','fz')):raise ValueError('Missing force dump columns')
        rows=lines[i+9:i+9+n]
        if len(rows)!=n:raise ValueError('Truncated force dump')
        table={}
        for row in rows:
            words=row.split();atom=int(words[fields.index('id')])
            if atom in table:raise ValueError('Duplicate atom ID in force dump')
            table[atom]=(int(words[fields.index('type')]),np.array([float(words[fields.index(k)]) for k in ('fx','fy','fz')]))
        last=table;i+=9+n
    if not last:raise ValueError('Empty force dump')
    return last


def read_reference_force_table(path):
    lines=Path(path).read_text().splitlines();n=int(lines[0])
    if len(lines[2:])!=n:raise ValueError('Reference force atom count mismatch')
    return {i:(int(row.split()[0]),np.array(list(map(float,row.split()[1:4])))) for i,row in enumerate(lines[2:],1)}


def force_squared_error(dump_file,reference_file,excluded_types=(3,)):
    current=read_force_table(dump_file);reference=read_reference_force_table(reference_file)
    if set(current)!=set(reference):raise ValueError('CG/reference force atom IDs differ')
    squared=0.;count=0
    for atom,(kind,ref) in reference.items():
        typ,force=current[atom]
        if typ!=kind:raise ValueError('CG/reference atom type mismatch at '+str(atom))
        if not np.isfinite(force).all() or not np.isfinite(ref).all():raise ValueError('Nonfinite force')
        if kind in excluded_types:continue
        squared+=float(np.sum((force-ref)**2));count+=1
    if not count:raise ValueError('No force sites after exclusions')
    return squared,count


def read_forces(path):
    table=read_force_table(path)
    return np.array([table[k][1] for k in sorted(table) if table[k][0]!=3])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True);parser.add_argument('--dump',type=Path,required=True)
    parser.add_argument('--reference-forces',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sq,n=force_squared_error(args.dump,args.reference_forces)
    args.output.write_text(json.dumps(dict(energy=read_energy(args.log),force_squared_error=sq,force_sites=n),indent=2)+'\n')
if __name__=='__main__':main()
