"""Replace only the intended LJ style, preserving Coulomb/repulsion terms."""
import json,re,sys
from pathlib import Path
import numpy as np

ELEMENT_TYPES={'Si':1,'O':2,'V':3,'C':4,'N':5,'Na':6}
LJ_STYLE='lj/gromacs/coul/gromacs'


def replace_lammps_params(input_file,output_file,params_file):
    params=json.loads(Path(params_file).read_text());pairs={};seen=set()
    for name,values in params.get('pairs',{}).items():
        names=name.split('-')
        if len(names)!=2 or any(n not in ELEMENT_TYPES for n in names):raise ValueError('Unknown pair '+name)
        if len(values)!=2 or not np.isfinite(values).all() or values[0]<0 or values[1]<=0:raise ValueError('Invalid epsilon/sigma '+name)
        key=tuple(sorted(ELEMENT_TYPES[n] for n in names))
        if key in pairs:raise ValueError('Duplicate/reversed pair '+name)
        pairs[key]=values
    lines=[]
    for line in Path(input_file).read_text().splitlines(keepends=True):
        body,sep,comment=line.partition('#');words=body.split()
        if len(words)>=6 and words[0]=='pair_coeff' and words[1].isdigit() and words[2].isdigit() and words[3]==LJ_STYLE:
            key=tuple(sorted(map(int,words[1:3])))
            if key in pairs:
                words[4:6]=list(map(str,pairs[key]));seen.add(key)
                line=' '.join(words)+(' #'+comment.rstrip('\n') if sep else '')+'\n'
        lines.append(line)
    if set(pairs)!=seen:raise ValueError('Requested LJ pair absent from template: '+str(set(pairs)-seen))
    Path(output_file).write_text(''.join(lines))

if __name__=='__main__':replace_lammps_params(*sys.argv[1:4])
