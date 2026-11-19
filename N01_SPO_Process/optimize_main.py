#!/usr/bin/env python3
"""SPO coordinate optimization with explicit paths and comparable trial losses.

--check validates inputs without launching jobs. --prepare writes one reviewable
trial. --run executes configured local/Slurm jobs. Gaussian generation is a
separate upstream stage; `qm` below always means CG evaluation on QM geometries.
"""
import argparse,copy,hashlib,json,logging,shlex,shutil,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from workflow import load_workflow,check_workflow,validate_parameters
from scripts.params_to_in import replace_lammps_params
from scripts.spo_loss import qm_losses,mle_weights,weighted_loss


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def command(args):return ' '.join(shlex.quote(str(a)) for a in args)


class ParamOptimizer:
    def __init__(self,config_file=None,nthreads=None,current_iteration=1,job_number=None):
        self.config=load_workflow(config_file);self.base_dir=Path(__file__).resolve().parent
        self.paths={k:Path(v) for k,v in self.config['paths'].items()}
        self.runtime=self.config['runtime'];self.options=self.config['optimization'];self.objective=self.config['objective']
        if nthreads is not None:self.runtime['md_ranks']=nthreads
        if job_number is not None:self.runtime['qm_batches']=job_number
        self.parameters=validate_parameters(json.loads(self.paths['parameters'].read_text()))
        self.initial_pairs=copy.deepcopy(self.parameters['pairs']);self.lambda_init=self.parameters['lambda']
        self.optimal_params=self._expand_parameters();self.weights=np.array(self.objective['initial_weights'],dtype=float)
        self.current_iteration=current_iteration;self.history=[];self.samples=[];self.active_samples=[]
        self.rng=np.random.default_rng(self.options['seed']);self.step_ratio=self.options['step_ratio']

    def _expand_parameters(self):
        result={f'pairs_{pair}_{i}':float(v) for pair,values in self.parameters['pairs'].items() for i,v in enumerate(values)}
        result['lambda_0']=float(self.parameters['lambda']);return result

    def _contract_parameters(self,flat):
        return dict(pairs={pair:[float(flat[f'pairs_{pair}_{i}']) for i in range(2)] for pair in self.initial_pairs},
                    **{'lambda':float(flat['lambda_0'])})

    def build_params_for_mode(self,mode,flat):
        if mode not in ('qm','md'):raise ValueError('mode must be qm or md')
        params=self._contract_parameters(flat);validate_parameters(params)
        pairs=params['pairs']
        if mode=='md':
            for pair in ('Na-O','C-O','N-O'):
                if pair in pairs:pairs[pair][0]*=params['lambda']
        return dict(pairs=pairs)

    def load_dataset(self):
        data=json.loads(self.paths['dataset_manifest'].read_text())
        if data.get('version')!=1:raise ValueError('Unsupported dataset manifest version')
        if data.get('units')!=dict(energy='kcal/mol',force='kcal/mol/angstrom'):raise ValueError('Dataset units are not kcal/mol and angstrom')
        self.samples=data['samples']
        if len(self.samples)<2 or len({s['id'] for s in self.samples})!=len(self.samples):raise ValueError('Dataset needs >=2 unique samples')
        for s in self.samples:
            if set(s.get('sha256',{}))!={'data','forces'}:raise ValueError('Rebuild dataset.json with scripts/solve_CG.py to record input hashes')
            if not np.isfinite(s['energy']):raise ValueError('Invalid reference energy')
            for key in ('data','forces'):
                path=Path(s[key]);path=(self.paths['dataset_manifest'].parent/path).resolve();s[key]=str(path)
                if not path.is_file():raise FileNotFoundError(path)

    def select_samples(self):
        count=self.options['sample_size']
        if count<=0 or count>=len(self.samples):self.active_samples=copy.deepcopy(self.samples);return
        # Every family and its fixed DFT minimum must survive sampling.
        anchors={}
        for sample in self.samples:
            if sample['family'] not in anchors or sample['energy']<anchors[sample['family']]['energy']:
                anchors[sample['family']]=sample
        if count<len(anchors):raise ValueError('sample_size is smaller than number of composition families')
        ids={s['id'] for s in anchors.values()};others=[s for s in self.samples if s['id'] not in ids]
        indices=self.rng.choice(len(others),count-len(ids),replace=False)
        self.active_samples=sorted(list(anchors.values())+[others[i] for i in indices],key=lambda s:s['id'])

    def _trial_dir(self,flat):
        payload=dict(parameters=flat,weights=self.weights.tolist(),samples=[s['id'] for s in self.active_samples],iteration=self.current_iteration)
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]
        return self.paths['run']/('iteration_%04d'%self.current_iteration)/('trial_'+digest)

    def _script(self,stage,folder):
        r=self.runtime;work=folder/stage
        lines=['#!/bin/bash',f'#SBATCH --job-name=spo-{stage}',f'#SBATCH --output={shlex.quote(str(work/"slurm-%A_%a.out"))}',
               '#SBATCH --nodes=1',f'#SBATCH --ntasks={r["md_ranks"] if stage=="md" else 1}',
               f'#SBATCH --cpus-per-task={1 if stage=="md" else r["qm_workers"]}']
        if stage=='qm':lines.append(f'#SBATCH --array=0-{r["qm_batches"]-1}')
        for key in ('partition','account'):
            if r.get(key):lines.append(f'#SBATCH --{key}={shlex.quote(r[key])}')
        lines+=['set -euo pipefail','export OMP_NUM_THREADS=1']
        if r.get('environment_setup'):lines.append('source '+shlex.quote(r['environment_setup']))
        lines.append('cd '+shlex.quote(str(work)))
        if stage=='qm':
            args=[r['python'],self.base_dir/'scripts/evaluate_qm_batch.py','--manifest',folder/'dataset.json','--input',work/'lmp_qm.in',
                  '--output',work,'--executable',r['lammps_serial'],'--workers',r['qm_workers'],'--batch-count',r['qm_batches'],
                  '--exclude-types',','.join(map(str,self.objective['excluded_force_types']))]
            lines.append(command(args)+' --batch-index "${SLURM_ARRAY_TASK_ID:-0}"')
        else:
            lines.append(command(r['mpi_launcher']+['-n',str(r['md_ranks']),r['lammps_mpi'],'-in','lmp_md.in']))
            lines.append(command([r['python'],self.base_dir/'scripts/get_traj_info.py','--current',work,'--workers',r['md_ranks'],
                                  '--start-frame',self.objective['rdf_start_frame'],'--r-min',self.objective['rdf_lower_angstrom'],
                                  '--r-max',self.objective['rdf_cutoff_angstrom']]))
            lines.append(command([r['python'],self.base_dir/'scripts/gr_similarity.py','--config',self.paths['run']/'resolved_workflow.json','--current',work]))
        return '\n'.join(lines)+'\n'

    def prepare_trial(self,flat):
        folder=self._trial_dir(flat);folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():return folder
        write_json(folder/'parameters.json',self._contract_parameters(flat))
        write_json(folder/'dataset.json',dict(version=1,samples=self.active_samples))
        for mode in ('qm','md'):
            work=folder/mode;work.mkdir(exist_ok=True)
            for src in self.paths['templates'].iterdir():
                if src.is_file():shutil.copy2(src,work/src.name)
            write_json(work/'parameters.json',self.build_params_for_mode(mode,flat))
            replace_lammps_params(work/f'lmp_{mode}.in',work/f'lmp_{mode}.in',work/'parameters.json')
            (folder/f'submit_{mode}.sh').write_text(self._script(mode,folder))
        return folder

    def _wait_for_job(self,job_ids):
        started=time.monotonic();joined=','.join(job_ids)
        while True:
            query=subprocess.run(['squeue','-h','-j',joined],capture_output=True,text=True,check=True)
            if not query.stdout.strip():break
            if time.monotonic()-started>self.runtime['timeout_seconds']:raise TimeoutError('Slurm wait timed out: '+joined)
            time.sleep(self.runtime['poll_seconds'])
        account=subprocess.run(['sacct','-X','-n','-P','-j',joined,'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True)
        rows=[line.split('|') for line in account.stdout.splitlines() if line.strip()]
        for job in job_ids:
            matches=[r for r in rows if r[0]==job or r[0].startswith(job+'_')]
            if not matches or any(r[1]!='COMPLETED' or r[2]!='0:0' for r in matches):
                raise RuntimeError('Slurm job did not complete successfully: '+job+' '+str(matches))

    def evaluate(self,flat):
        folder=self.prepare_trial(flat);result_file=folder/'result.json'
        if result_file.exists():return json.loads(result_file.read_text())
        # Incomplete trials are not reusable: stale MD/QM output must never score.
        if (folder/'submitted.json').exists():raise RuntimeError('Incomplete existing trial; inspect its logs and use a new run directory: '+str(folder))
        write_json(folder/'submitted.json',dict(backend=self.runtime['backend']))
        if self.runtime['backend']=='slurm':
            jobs=[]
            for mode in ('qm','md'):
                out=subprocess.run(['sbatch','--parsable',str(folder/f'submit_{mode}.sh')],capture_output=True,text=True,check=True)
                jobs.append(out.stdout.strip().split(';')[0])
            write_json(folder/'submitted.json',dict(backend='slurm',jobs=jobs));self._wait_for_job(jobs)
        else:
            # Local execution is explicit and uses the same prepared scripts.
            import os
            for index in range(self.runtime['qm_batches']):
                env=dict(os.environ,SLURM_ARRAY_TASK_ID=str(index))
                subprocess.run(['bash',str(folder/'submit_qm.sh')],env=env,check=True)
            subprocess.run(['bash',str(folder/'submit_md.sh')],check=True)
        predicted={}
        for index in range(self.runtime['qm_batches']):
            for row in json.loads((folder/'qm'/('batch_%04d.json'%index)).read_text()):
                if row['id'] in predicted:raise ValueError('Duplicate QM result')
                predicted[row['id']]=row
        losses=qm_losses(predicted,{s['id']:s for s in self.active_samples})
        md=json.loads((folder/'md/rdf_loss.json').read_text())
        if set(md['pairs'])!=set(self.objective['rdf_pairs']) or not np.isfinite(md['total']):raise ValueError('Invalid RDF result')
        total=weighted_loss(losses,self.weights)+float(md['total'])
        result=dict(qm_losses=losses.tolist(),weights=self.weights.tolist(),md_loss=md['total'],total=total,folder=str(folder))
        write_json(result_file,result);return result

    def _bounds(self,param):return (0.,1.) if param=='lambda_0' else ((.05,float('inf')) if param.endswith('_0') else (1.,float('inf')))

    def _generate_param_range(self,param):
        value=self.optimal_params[param];lo,hi=self._bounds(param)
        step=max(abs(value)*self.step_ratio,self.options['min_step'])
        points=sorted(set(float(np.clip(v,lo,hi)) for v in (value-step,value,value+step)))
        if len(points)<3:points.append(float(np.clip(value+(2*step if value<=lo+step else -2*step),lo,hi)))
        return value,sorted(set(points))

    def _find_optimal(self,values,scores):
        x=np.array(values,dtype=float);y=np.array(scores,dtype=float)
        if not np.isfinite(y).all():raise ValueError('Invalid scan objective')
        if len(x)<3:return float(x[np.argmin(y)])
        center=x.mean();scale=np.std(x)
        a,b,_=np.polyfit((x-center)/scale,y,2)
        return float(np.clip(center-scale*b/(2*a),min(x),max(x))) if a>0 else float(x[np.argmin(y)])

    def optimize_parameter(self,param):
        old,values=self._generate_param_range(param)
        def trial(v):
            flat=dict(self.optimal_params);flat[param]=v
            return self.evaluate(flat)
        with ThreadPoolExecutor(max_workers=min(len(values),self.options['scan_workers'])) as pool:results=list(pool.map(trial,values))
        proposal=self._find_optimal(values,[r['total'] for r in results]);proposals=[proposal]
        if self.options['peak_sigma_proposal'] and param.startswith('pairs_') and param.endswith('_1'):
            key=param[len('pairs_'):]
            peak=json.loads((Path(results[0]['folder'])/'md/rdf_first_sigma_file.json').read_text()).get(key)
            if peak is not None:proposals.append(float(np.clip(peak,*self._bounds(param))))
        for candidate in proposals:
            if not any(abs(candidate-v)<1e-10 for v in values):values.append(candidate);results.append(trial(candidate))
        # An interpolated or RDF-derived value is accepted only after actual evaluation.
        best=min(range(len(values)),key=lambda i:results[i]['total'])
        self.optimal_params[param]=values[best]
        return results[best]

    def _initialize_run(self,resume=False):
        report=check_workflow(self.config)
        if not report['ready']:raise FileNotFoundError('Missing inputs:\n'+'\n'.join(report['missing']))
        self.load_dataset();run=self.paths['run'];run.mkdir(parents=True,exist_ok=True)
        inputs=[self.paths['parameters'],self.paths['dataset_manifest']]+sorted(self.paths['templates'].glob('*'))+sorted(self.paths['rdf_reference'].glob('*.csv'))+[Path(__file__),self.base_dir/'workflow.py']+sorted((self.base_dir/'scripts').glob('*.py'))
        self.config['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.is_file()}
        snapshot=run/'resolved_workflow.json'
        if snapshot.exists():
            if json.loads(snapshot.read_text())!=self.config:raise ValueError('Run configuration changed; use a new paths.run')
            if not resume and ((run/'checkpoint.json').exists() or any(run.glob('iteration_*/trial_*/submitted.json'))):
                raise ValueError('Run already started; use --resume or choose a new paths.run')
        else:write_json(snapshot,self.config)
        logging.basicConfig(filename=run/'optimize.log',level=logging.INFO,format='%(asctime)s %(message)s')
        if resume:
            state=json.loads((run/'checkpoint.json').read_text())
            self.optimal_params=state['parameters'];self.weights=np.array(state['weights']);self.history=state['history']
            self.current_iteration=state['next_iteration'];self.rng.bit_generator.state=state['rng']
        elif self.current_iteration!=1:raise ValueError('Iteration index is not a restart; use --resume')

    def main_optimization(self,resume=False):
        self._initialize_run(resume)
        while self.current_iteration<=self.options['iterations']:
            self.select_samples();before=dict(self.optimal_params);old_weights=self.weights.copy()
            # Keep the same sample subset and weights throughout a complete coordinate sweep.
            initial=self.evaluate(self.optimal_params)
            for param in self.optimal_params:self.optimize_parameter(param)
            final=self.evaluate(self.optimal_params)
            self.weights=mle_weights(final['qm_losses'],self.objective['loss_floor'])
            self.history.append(dict(parameters=dict(self.optimal_params),objective=final))
            change=np.mean([abs(self.optimal_params[k]-before[k])/max(abs(before[k]),1e-3) for k in before])
            logging.info('iteration=%s objective=%s relative_change=%s',self.current_iteration,final,change)
            self.current_iteration+=1
            write_json(self.paths['run']/'checkpoint.json',dict(parameters=self.optimal_params,weights=self.weights.tolist(),history=self.history,
                       next_iteration=self.current_iteration,rng=self.rng.bit_generator.state))
            write_json(self.paths['run']/'optimized_params.json',self._contract_parameters(self.optimal_params))
            weight_change=float(np.max(np.abs(self.weights-old_weights)/old_weights))
            if change<self.options['relative_tolerance'] and weight_change<self.options['relative_tolerance']:break


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--resume',action='store_true')
    actions=p.add_mutually_exclusive_group();actions.add_argument('--check',action='store_true');actions.add_argument('--prepare',action='store_true');actions.add_argument('--run',action='store_true')
    p.add_argument('legacy_ranks',nargs='?',type=int);p.add_argument('legacy_iteration',nargs='?',type=int)
    args=p.parse_args()
    optimizer=ParamOptimizer(args.config,nthreads=args.legacy_ranks,current_iteration=args.legacy_iteration or 1)
    if args.run:optimizer.main_optimization(args.resume)
    elif args.prepare:
        optimizer._initialize_run(args.resume);optimizer.select_samples();folder=optimizer.prepare_trial(optimizer.optimal_params);print(folder)
    else:
        report=check_workflow(optimizer.config);print(json.dumps(report,indent=2));return 0 if report['ready'] else 2
    return 0
if __name__=='__main__':sys.exit(main())
