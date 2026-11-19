"""One path convention: config-relative project root, then project-relative paths."""
import json
import os
from pathlib import Path

DEFAULT_CONFIG=Path(__file__).resolve().parent/'config/workflow.json'
PATH_KEYS=('parameters','qm_source','cg_reference','dataset_manifest','templates','rdf_reference','run')


def load_workflow(path=None):
    config_path=Path(path or DEFAULT_CONFIG).expanduser().resolve()
    config=json.loads(config_path.read_text())
    root=(config_path.parent/Path(os.path.expandvars(config['project_root'])).expanduser()).resolve()
    config['project_root']=str(root);config['config_file']=str(config_path)
    for key in PATH_KEYS:
        value=Path(os.path.expandvars(config['paths'][key])).expanduser()
        config['paths'][key]=str((root/value).resolve())
    setup=config['runtime'].get('environment_setup')
    if setup:config['runtime']['environment_setup']=str((root/Path(os.path.expandvars(setup)).expanduser()).resolve())
    for key in ('environment_setup','scratch'):
        value=config.get('qm_generation',{}).get(key)
        if value:config['qm_generation'][key]=str((root/Path(os.path.expandvars(value)).expanduser()).resolve())
    r=config['runtime'];o=config['objective'];opt=config['optimization']
    if r['backend'] not in ('slurm','local'):raise ValueError('runtime.backend must be slurm or local')
    for key in ('md_ranks','qm_workers','qm_batches'):
        if not isinstance(r[key],int) or r[key]<1:raise ValueError(key+' must be a positive integer')
    if not isinstance(r['mpi_launcher'],list) or not r['mpi_launcher']:raise ValueError('mpi_launcher must be a command argument list')
    if o['rdf_metric'] not in ('squared_l2','paper_signed'):raise ValueError('Unknown rdf_metric')
    if not 0<=o['rdf_lower_angstrom']<o['rdf_cutoff_angstrom']:raise ValueError('Invalid RDF range')
    if len(o['initial_weights'])!=3 or any(w<=0 for w in o['initial_weights']):raise ValueError('Three positive QM weights required: energy, force, zero')
    if o['loss_floor']<=0 or opt['iterations']<1 or opt['scan_workers']<1:raise ValueError('Invalid optimization controls')
    for key in ('parameters','dataset_manifest','qm_source','cg_reference','templates','rdf_reference'):
        try:Path(config['paths'][key]).relative_to(Path(config['paths']['run']))
        except ValueError:continue
        raise ValueError('Reference inputs must not be inside the run directory')
    raw=Path(config['paths']['qm_source']);mapped=Path(config['paths']['cg_reference'])
    if raw==mapped or raw in mapped.parents or mapped in raw.parents:
        raise ValueError('qm_source and cg_reference must be separate, non-nested directories')
    return config


def validate_parameters(params):
    import math
    if not isinstance(params.get('pairs'),dict) or not params['pairs']:raise ValueError('Missing pair parameters')
    if not math.isfinite(params['lambda']) or not 0<=params['lambda']<=1:raise ValueError('lambda must lie in [0,1]')
    for name,values in params['pairs'].items():
        if len(name.split('-'))!=2 or len(values)!=2 or not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('Invalid epsilon/sigma for '+name)
    return params


def check_workflow(config):
    paths={k:Path(v) for k,v in config['paths'].items()};missing=[]
    required=[paths['parameters'],paths['dataset_manifest'],paths['templates']/'lmp_qm.in',paths['templates']/'lmp_md.in',paths['templates']/'lmp.data']
    required += [paths['rdf_reference']/('rdf_'+pair+'.csv') for pair in config['objective']['rdf_pairs']]
    if config['runtime'].get('environment_setup'):required.append(Path(config['runtime']['environment_setup']))
    for path in required:
        if not path.is_file():missing.append(str(path))
    if paths['parameters'].is_file():validate_parameters(json.loads(paths['parameters'].read_text()))
    return dict(config=config['config_file'],project_root=config['project_root'],paths=config['paths'],
        missing=missing,ready=not missing,backend=config['runtime']['backend'],
        rdf_metric=config['objective']['rdf_metric'],
        note='Input-file check only; runtime executables/environment are not tested. QM fitting means CG single-point evaluation on mapped DFT geometries; it does not launch Gaussian.')
