import copy,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from workflow import load_workflow,validate_parameters,check_workflow
from optimize_main import ParamOptimizer
from spo_loss import relative_energies,qm_losses,mle_weights,weighted_loss,rdf_distance
from energy_and_force_diff import read_energy,read_force_table,force_squared_error
from params_to_in import replace_lammps_params
from solve_CG import build_manifest
from gr_similarity import score_rdfs
from map_QM_to_CG import parse_gaussian_output,mapping_rules,convert_sample,write_lammps_data


class NumericalTests(unittest.TestCase):
    def test_relative_energy_same_reference_configuration(self):
        actual,truth=relative_energies([30,10,90,100],[3,1,8,9],['a','a','b','b'])
        np.testing.assert_equal(actual,[20,0,0,10]);np.testing.assert_equal(truth,[2,0,0,1])
    def test_three_terms_and_force_configuration_vector(self):
        ref={'a':dict(energy=1,family='f'),'b':dict(energy=3,family='f')}
        pred={'a':dict(energy=10,force_squared_error=9),'b':dict(energy=14,force_squared_error=16)}
        loss=qm_losses(pred,ref)
        np.testing.assert_allclose(loss,[np.sqrt(2),np.sqrt(12.5),0],atol=1e-14)
        self.assertEqual(weighted_loss([1,2,3],[2,3,4]),20)
    def test_absolute_pearson_as_printed_in_equation_three(self):
        ref={str(i):dict(energy=i,family='f') for i in range(3)}
        pred={str(i):dict(energy=-i,force_squared_error=0) for i in range(3)}
        self.assertAlmostEqual(qm_losses(pred,ref)[2],0.)
        for r in pred.values():r['energy']=0
        with self.assertRaisesRegex(ValueError,'Pearson'):qm_losses(pred,ref)
    def test_missing_or_nonfinite_sample_is_not_a_good_loss(self):
        with self.assertRaises(ValueError):qm_losses({}, {'a':dict(energy=0,family='f')})
        with self.assertRaises(ValueError):mle_weights([1,np.nan,0])
    def test_mle_solution_and_zero_regularization(self):
        np.testing.assert_allclose(mle_weights([2,4,.5]),[.25,.125,1])
        self.assertEqual(mle_weights([0],floor=1e-4)[0],5000.)
    def test_rdf_is_nonnegative_unbounded_and_cutoff_aligned(self):
        r=np.linspace(0,12,101);ref=np.column_stack((r,np.ones(len(r))))
        current=np.column_stack((r,np.ones(len(r))*3))
        self.assertAlmostEqual(rdf_distance(ref,ref,0,12),0)
        self.assertAlmostEqual(rdf_distance(current,ref,0,12),4)
        self.assertAlmostEqual(rdf_distance(current,ref,0,12,'paper_signed'),8)
        with self.assertRaisesRegex(ValueError,'cover'):rdf_distance(ref[20:],ref,0,12)
        with self.assertRaises(ValueError):rdf_distance(ref,np.column_stack((r,r*0)),0,12)


class FileTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='spo with spaces ');self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_named_thermo_column_not_footer(self):
        f=self.p/'log';f.write_text('Step Temp PotEng E_pair\n0 0 -8 -9\n1 0 -7 -10\nLoop time of 0.2 on 2 procs\nTotal neighbors = 123456\n')
        self.assertEqual(read_energy(f),-7)
        f.write_text('Step Temp E_pair\n0 2 -3\n')
        with self.assertRaises(ValueError):read_energy(f)
    def test_force_identity_and_virtual_type_exclusion(self):
        dump=self.p/'force.dump';ref=self.p/'ref.xyz'
        dump.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type fz fx fy\n3 3 100 100 100\n2 5 0 2 0\n1 1 0 0 3\n')
        ref.write_text('3\nForces\n1 0 0 0\n5 0 0 0\n3 0 0 0\n')
        sq,n=force_squared_error(dump,ref);self.assertEqual(sq,13);self.assertEqual(n,2)
        ref.write_text('3\nForces\n1 0 0 0\n4 0 0 0\n3 0 0 0\n')
        with self.assertRaisesRegex(ValueError,'type'):force_squared_error(dump,ref)
    def test_lj_replacement_does_not_overwrite_overlay(self):
        inp=self.p/'lmp.in';params=self.p/'params.json'
        inp.write_text('pair_coeff 2 6 lj/gromacs/coul/gromacs 1 2 9 12 # LJ\npair_coeff 2 6 coul/long\npair_coeff 2 6 lennard/mdf 99 0\n')
        params.write_text(json.dumps(dict(pairs={'Na-O':[3,4]})))
        replace_lammps_params(inp,inp,params)
        text=inp.read_text();self.assertIn('3 4 9 12 # LJ',text);self.assertIn('lennard/mdf 99 0',text)
        params.write_text(json.dumps(dict(pairs={'Si-C':[3,4]})))
        with self.assertRaisesRegex(ValueError,'absent'):replace_lammps_params(inp,inp,params)
    def test_rdf_pairs_are_selected_by_name_not_scan_parameter(self):
        a=self.p/'current';b=self.p/'reference';a.mkdir();b.mkdir()
        r=np.linspace(1.5,12,51)
        for pair in ['Si-Na','Na-O']:
            np.savetxt(a/('rdf_'+pair+'.csv'),np.column_stack((r,np.ones(51)*2)),delimiter=',')
            np.savetxt(b/('rdf_'+pair+'.csv'),np.column_stack((r,np.ones(51))),delimiter=',')
        (b/'rdf_Si-Si.csv').write_text('irrelevant')
        cfg=dict(rdf_pairs=['Si-Na','Na-O'],rdf_lower_angstrom=1.5,rdf_cutoff_angstrom=12,rdf_metric='squared_l2')
        result,_=score_rdfs(a,b,cfg);self.assertEqual(result['total'],2.)
    def gaussian(self,energy,shift,force):
        return f''' Standard orientation:
 -----
 Center Atomic Atomic Coordinates
 Number Number Type X Y Z
 -----
 1 11 0 {shift} 0.0 0.0
 -----
 SCF Done: E(RB3LYP) = {energy} A.U.
 Center Atomic Forces (Hartrees/Bohr)
 Number Number X Y Z
 -----
 1 11 {force} 0.0 0.0
 -----
 Normal termination of Gaussian
'''
    def test_gaussian_last_complete_frame_and_D_exponent(self):
        log=self.p/'qm.log';log.write_text(self.gaussian('-1.0',0,'0.01')+self.gaussian('-2.0D+00',1,'2.0D-02'))
        atoms,e=parse_gaussian_output(log);self.assertEqual(e,-2);self.assertEqual(atoms[0]['X'],1);self.assertEqual(atoms[0]['Fx'],.02)
        log.write_text(log.read_text()+'Error termination\n')
        with self.assertRaises(ValueError):parse_gaussian_output(log)
    def test_mapping_does_not_write_raw_QM_and_index_hashes(self):
        root=self.p/'raw QM';sample=root/'Na-Na'/'point';sample.mkdir(parents=True)
        log=sample/'point.log';text=self.gaussian('-1.0',0,'0.01');log.write_text(text)
        out=convert_sample(sample,root,self.p/'mapped')
        self.assertEqual(log.read_text(),text);self.assertEqual(list(sample.iterdir()),[log])
        self.assertIn('8 atom types',(out/'point.data').read_text())
        records=build_manifest(self.p/'mapped',self.p/'dataset.json');self.assertEqual(records[0]['id'],'Na-Na/point')
        self.assertIn('forces',records[0]['sha256'])
    def test_silicate_mapping_preserves_net_force_and_torque(self):
        directions=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
        positions=np.vstack(([0,0,0],directions*1.6,directions*2.6))
        forces=np.random.default_rng(12).normal(size=positions.shape)
        atoms=[dict(Element=e,X=x[0],Y=x[1],Z=x[2],Fx=f[0],Fy=f[1],Fz=f[2]) for e,x,f in zip(['Si']+['O']*4+['H']*4,positions,forces)]
        cg,_,_,mapped=mapping_rules(atoms);coords=np.array([a[2] for a in cg])
        np.testing.assert_allclose(np.sum(mapped,axis=0),forces.sum(axis=0),atol=1e-12)
        np.testing.assert_allclose(np.cross(coords,mapped).sum(axis=0),np.cross(positions,forces).sum(axis=0),atol=1e-12)
        np.testing.assert_allclose(np.asarray(mapped)[5:],0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='SPO run with spaces ');self.root=Path(self.tmp.name)
        config=json.loads((ROOT/'config/workflow.json').read_text());config['project_root']=str(self.root)
        for key in ('parameters','templates','rdf_reference'):config['paths'][key]=str(ROOT/config['paths'][key])
        config['paths']['dataset_manifest']='mapped/dataset.json';config['paths']['run']='runs/test'
        self.cfg=self.root/'workflow.json';self.cfg.write_text(json.dumps(config))
        records=[]
        for i,e in enumerate([1,3]):
            d=self.root/'mapped'/'Si-Na'/str(i);d.mkdir(parents=True)
            (d/(str(i)+'.data')).write_text('fixture')
            (d/(str(i)+'_force.xyz')).write_text('1\nforce\n1 0 0 0\n')
            (d/(str(i)+'_energy.xyz')).write_text(str(e))
        build_manifest(self.root/'mapped',self.root/'mapped/dataset.json')
    def tearDown(self):self.tmp.cleanup()
    def test_paths_independent_of_cwd_and_preflight(self):
        before=load_workflow(self.cfg);old=os.getcwd()
        try:os.chdir('/tmp');after=load_workflow(self.cfg)
        finally:os.chdir(old)
        self.assertEqual(before,after);self.assertTrue(check_workflow(before)['ready'])
    def test_lambda_only_scales_MD_oxygen_epsilon_once(self):
        opt=ParamOptimizer(self.cfg);flat=dict(opt.optimal_params);flat['lambda_0']=.2
        qm=opt.build_params_for_mode('qm',flat)['pairs'];md=opt.build_params_for_mode('md',flat)['pairs']
        self.assertEqual(md['Na-O'][0],qm['Na-O'][0]*.2);self.assertEqual(md['Si-Na'],qm['Si-Na'])
        self.assertEqual(md['Na-O'][1],qm['Na-O'][1])
        with self.assertRaises(ValueError):validate_parameters(dict(pairs=qm,**{'lambda':1.1}))
    def test_prepare_scripts_and_review_then_run_initialization(self):
        opt=ParamOptimizer(self.cfg);opt._initialize_run();opt.select_samples();folder=opt.prepare_trial(opt.optimal_params)
        for mode in ('qm','md'):
            subprocess.run(['bash','-n',str(folder/f'submit_{mode}.sh')],check=True)
            self.assertTrue((folder/mode/'parameters.json').is_file())
        self.assertNotIn('l/64',(folder/'submit_qm.sh').read_text())
        self.assertIn('--array=0-9',(folder/'submit_qm.sh').read_text())
        again=ParamOptimizer(self.cfg);again._initialize_run()
    def test_lambda_scans_respect_bounds_and_contain_current(self):
        opt=ParamOptimizer(self.cfg)
        for value in [0.,.01,.99,1.]:
            opt.optimal_params['lambda_0']=value;_,points=opt._generate_param_range('lambda_0')
            self.assertIn(value,points);self.assertTrue(all(0<=v<=1 for v in points));self.assertGreaterEqual(len(points),3)
    def test_failed_queue_is_not_success(self):
        opt=ParamOptimizer(self.cfg)
        with patch('optimize_main.subprocess.run',side_effect=subprocess.CalledProcessError(1,'squeue')):
            with self.assertRaises(subprocess.CalledProcessError):opt._wait_for_job(['1'])
    def test_parameter_candidates_are_actually_evaluated(self):
        opt=ParamOptimizer(self.cfg);opt.options['peak_sigma_proposal']=False;calls=[]
        def fake(flat):
            value=flat['lambda_0'];calls.append(value)
            # Interpolation proposes .16, but its actual evaluation is worse.
            loss=(value-.16)**2 if abs(value-.16)>1e-8 else 10.
            return dict(total=loss,folder='unused')
        with patch.object(opt,'evaluate',side_effect=fake):opt.optimize_parameter('lambda_0')
        self.assertTrue(any(abs(v-.16)<1e-8 for v in calls));self.assertNotAlmostEqual(opt.optimal_params['lambda_0'],.16)

    def test_weight_update_does_not_stop_early_and_checkpoint_restores(self):
        opt=ParamOptimizer(self.cfg);opt.options['iterations']=2
        result=dict(qm_losses=[1.,2.,.5],total=3.,folder='fixture')
        with patch.object(opt,'evaluate',return_value=result), patch.object(opt,'optimize_parameter'):
            opt.main_optimization()
        self.assertEqual(len(opt.history),2)
        expected=opt.rng.random()
        resumed=ParamOptimizer(self.cfg);resumed.options['iterations']=2
        resumed._initialize_run(resume=True)
        self.assertEqual(resumed.current_iteration,3)
        self.assertEqual(resumed.optimal_params,opt.optimal_params)
        np.testing.assert_equal(resumed.weights,[.5,.25,1.])
        self.assertEqual(resumed.rng.random(),expected)

    def test_raw_and_mapped_directories_cannot_overlap(self):
        cfg=json.loads(self.cfg.read_text());cfg['paths']['cg_reference']=cfg['paths']['qm_source']+'/mapped'
        self.cfg.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'non-nested'):load_workflow(self.cfg)

if __name__=='__main__':unittest.main()
