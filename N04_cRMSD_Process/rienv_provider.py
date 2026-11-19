"""Real PLUMED RIEnv snapshots for core selection only (no RIEnv bias).

Generate complete interior local references from the supplied crystal, remove
only proper-rotation/permutation duplicates, and evaluate the kernel projection. External precomputed scores can instead
be passed to PIRMSD.run for an embedding application.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def equivalent(a,b,tolerance=.005):
    if len(a)!=len(b):return False
    ra,rb=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    if np.max(np.abs(np.sort(ra)-np.sort(rb)))>tolerance:return False
    da=np.linalg.norm(a[:,None]-a[None,:],axis=2);db=np.linalg.norm(b[:,None]-b[None,:],axis=2)
    if np.max(np.abs(np.sort(da.ravel())-np.sort(db.ravel())))>2*tolerance:return False
    i=int(np.argmax(ra));j=int(np.argmax(np.linalg.norm(np.cross(a[i],a),axis=1)))
    def basis(x,y):
        x=x/np.linalg.norm(x);y=y-x*np.dot(x,y)
        if np.linalg.norm(y)<1e-8:return None
        y/=np.linalg.norm(y);return np.column_stack((x,y,np.cross(x,y)))
    first=basis(a[i],a[j])
    if first is None:return False
    for k in np.flatnonzero(abs(rb-ra[i])<tolerance):
        for l in np.flatnonzero(abs(rb-ra[j])<tolerance):
            if abs(db[k,l]-da[i,j])>2*tolerance:continue
            second=basis(b[k],b[l])
            if second is None:continue
            dist=np.linalg.norm((a@(second@first.T).T)[:,None]-b[None,:],axis=2)
            rows,cols=linear_sum_assignment(dist*dist)
            if dist[rows,cols].max()<2*tolerance:return True
    return False


class RIEnvProvider:
    def __init__(self,reference,settings=None,directory=None):
        self.settings=dict(executable='plumed',reference_radius=7.2,sigma=.8,lambda_value=200,zeta=3)
        self.settings.update(settings or {})
        self.directory=Path(directory or tempfile.mkdtemp(prefix='pirmsd-rienv-')).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        explicit=self.settings.get('references')
        if explicit:
            self.references=[Path(p).resolve() for p in explicit]
            if not all(p.is_file() for p in self.references):raise ValueError('Missing RIEnv reference PDB')
        else:
            coords=np.asarray(reference)[:,2:5];radius=self.settings['reference_radius']
            interior=np.flatnonzero(np.all((coords-coords.min(axis=0)>radius)&(coords.max(axis=0)-coords>radius),axis=1))
            if not len(interior):raise ValueError('No complete interior RIEnv environment; supply explicit references or a larger crystal')
            tree=cKDTree(coords);templates=[]
            for i in interior:
                ids=tree.query_ball_point(coords[i],radius)
                env=coords[[j for j in ids if j!=i]]-coords[i]
                if len(env)<3:continue
                if not any(equivalent(env,other) for other in templates):templates.append(env)
            if not templates:raise ValueError('No valid RIEnv reference environments')
            self.references=[]
            for k,env in enumerate(templates):
                path=self.directory/('env_%03d.pdb'%k)
                lines=['MODEL     1']
                for j,p in enumerate(env,1):
                    lines.append('ATOM  %5d   Si MOL     1    %8.3f%8.3f%8.3f  1.00  0.00          SI  '%(j,*p))
                lines+=['ENDMDL','END'];path.write_text('\n'.join(lines)+'\n');self.references.append(path)

    def compute(self,coordinates,box):
        coordinates=np.asarray(coordinates,dtype=float);n=len(coordinates)
        if not n:return np.zeros(0)
        executable=shutil.which(self.settings['executable'])
        if executable is None:raise RuntimeError('RIEnv requires the PLUMED executable with ENVIRONMENTSIMILARITYKERNEL')
        with tempfile.TemporaryDirectory(prefix='snapshot-',dir=self.directory) as tmp:
            tmp=Path(tmp)
            (tmp/'points.xyz').write_text(str(n)+'\n'+' '.join(map(str,box))+'\n'+'\n'.join(
                'Si %.12g %.12g %.12g'%tuple(p) for p in coordinates)+'\n')
            lines=['UNITS LENGTH=A','ENVIRONMENTSIMILARITYKERNEL ...',
                ' SPECIES=1-%d'%n,' SIGMA='+str(self.settings['sigma']),
                ' LAMBDA='+str(self.settings['lambda_value']),' ORDERS=0-4',
                ' ZETA='+str(self.settings['zeta']),' CRYSTAL_STRUCTURE=CUSTOM',' LABEL=refcv']
            lines+=[' REFERENCE_%d=%s'%(j+1,p) for j,p in enumerate(self.references)]
            lines+=[' MEAN','... ENVIRONMENTSIMILARITYKERNEL',
                'ri_scalar: MCOLV_COMBINE DATA=refcv COEFFICIENTS=1 MEAN',
                'DUMPMULTICOLVAR STRIDE=1 DATA=ri_scalar FILE=scores.xyz','FLUSH STRIDE=1']
            (tmp/'plumed.dat').write_text('\n'.join(lines)+'\n')
            # A serial diagnostic subprocess must not attach to the parent's MPI world.
            env={k:v for k,v in os.environ.items() if not k.startswith(('PMI_','PMIX_','OMPI_','HYDRA_'))}
            env['OMP_NUM_THREADS']='1'
            result=subprocess.run([executable,'driver','--ixyz','points.xyz','--length-units','A','--plumed','plumed.dat'],
                cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if result.returncode:
                (self.directory/'last_failure.log').write_text(result.stdout)
                raise RuntimeError('RIEnv PLUMED driver failed; see '+str(self.directory/'last_failure.log'))
            rows=(tmp/'scores.xyz').read_text().splitlines()
            if int(rows[0])!=n or len(rows)<n+2:raise ValueError('Incomplete RIEnv snapshot')
            data=np.array([list(map(float,row.split()[1:5])) for row in rows[2:n+2]])
            if data.shape!=(n,4) or not np.isfinite(data).all():raise ValueError('Invalid RIEnv output')
            # Explicitly verify row/atom order rather than trusting a stale file.
            delta=data[:,:3]-coordinates;delta-=box*np.round(delta/box)
            if np.max(np.abs(delta))>1e-4:raise ValueError('RIEnv output atom order/coordinates mismatch')
            return data[:,3]
