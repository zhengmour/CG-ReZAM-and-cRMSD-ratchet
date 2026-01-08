# %%
# ---------------------------
# Module Imports and Setup
# ---------------------------
import logging
import argparse
import json
import sys
from lammps import lammps

def argparse_input():
    parser = argparse.ArgumentParser(
        description='kMC rate constant optimization program',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage examples:
  python variableMD.py -i/--input lmp.in --log MD.log --verbose
        '''
    )
    
    parser.add_argument('-i', '--input',
                       default='lmp.in',
                       help='Enter the file path (default: lmp.in)')

    parser.add_argument('--input_md',
                       default='lmp.md.in',
                       help='Enter the file path (default: lmp.md.in)')
    
    parser.add_argument('--input_annealing',
                       default='lmp.annealing.in',
                       help='Enter the file path (default: lmp.annealing.in)')

    parser.add_argument('--input_plumed',
                       default='lmp.plumed.in',
                       help='Enter the file path (default: lmp.plumed.in)')

    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Shows detailed output')
    
    parser.add_argument('--debug',
                       action='store_true',
                       help='Turn on debugging mode')
    
    parser.add_argument('--log',
                       default='MD.log',
                       help='Output file path (default: MD.log)')
    
    try:
        args = parser.parse_args()
        return args
        
    except SystemExit:
        print("Parameter resolution fails or helps are displayed")
        return None


# ---------------------------
# kMCMD class
# ---------------------------
class EnhanceSampling:
    def __init__(self, args):
        # Read input parameters
        self.lmpfile = args.input
        self.lmp_mdfile = args.input_md
        self.lmp_annealingfile = args.input_annealing
        self.lmp_plumedfile = args.input_plumed
        params_file = args.parameters

        # Configure logging system
        logging.basicConfig(
            filename="MD.log",           
            level=logging.INFO,          
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
            datefmt='%Y-%m-%d %H:%M:%S' 
        )
        self.logger = logging.getLogger("variableMD")
        if self.rank == 0:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)

        simulation_parameters = params['Simulation_Parameters']
        self.NITERATIONS  =  simulation_parameters['niterations']
        self.INIT_ITER    =  simulation_parameters['init_iter']
        self.INIT_TIME    =  simulation_parameters['init_time']         # ns
        self.TIME_STEP    =  simulation_parameters['time_step']
        self.NVT_STEPS    =  simulation_parameters['nvt_steps']         # NVT loops for MD
        self.NPT_STEPS    =  simulation_parameters['npt_steps']         # NPT loops for MD

        thermodynamic_parameters = params['Thermodynamic_Parameters']   
        self.T = thermodynamic_parameters['temperature']     # T (K)
        self.P = thermodynamic_parameters['pressure']     # P (K)
        
        # Main simulation loop
        self.current_time = self.INIT_TIME # ns

        self.enhance_sampling = False    
        self.sampling_methods = []    
        self.sampling_settings= []

        ENHANCE_SAMPLING = params['Enhance_Sampling']
        for sampling_method, sampling_setting in ENHANCE_SAMPLING.items():
            if sampling_setting['status']:
                self.enhance_sampling = True
                self.sampling_methods.append(sampling_method)
                self.sampling_settings.append(sampling_setting) 

    def _set_lmp(self):
        # run lmpfile one line at a time
        lines = open(self.lmpfile,'r').readlines()
        for line in lines: 
            line = line.replace('${T}', str(self.T))                                 
            line = line.replace('${TIME_STEP}', str(self.TIME_STEP))    
            self.lmp.command(line)

    def _write_lmp_md(self):
        self.current_time += (self.NVT_STEPS + self.NPT_STEPS)*self.TIME_STEP*1.0E-6 # ns 
        
        if self.rank == 0:
            lines = open(self.lmp_mdfile,'r').readlines()
            with open(f'lmp_{self.INIT_ITER}.md.in', 'w') as wf:
                for line in lines: 
                    line = line.replace('${T}', str(self.T))                   
                    line = line.replace('${P}', str(self.P))                  
                    line = line.replace('${TIME_STEP}', str(self.TIME_STEP))   
                    line = line.replace('${NVT_STEPS}', str(self.NVT_STEPS))   
                    line = line.replace('${NPT_STEPS}', str(self.NPT_STEPS))   
                    line = line.replace('${current_time}', f"{self.current_time:.4f}")   
                    wf.write(line)         

    def _write_lmp_annealing(self, setting):
        annealing_loop = setting["annealing_loop"]
        heating_steps, cooling_steps, equaling_steps = setting["annealing_step"]
        start_temp, stop_temp = setting["annealing_temp"]
        self.current_time += (annealing_loop*(heating_steps+cooling_steps+equaling_steps))*self.TIME_STEP*1.0E-6 # ns
        
        if self.rank == 0:
            lines = open(self.lmp_annealingfile,'r').readlines()
            with open(f'lmp_{self.INIT_ITER}.annealing.in', 'w') as wf:
                for line in lines: 
                    line = line.replace('${T}', str(self.T))                                
                    line = line.replace('${TIME_STEP}', str(self.TIME_STEP))  
                    line = line.replace('${lmp_in}', str('temp_lmp.annealing.in')) 
                    line = line.replace('${startTemp}', str(start_temp))  
                    line = line.replace('${stopTemp}', str(stop_temp)) 
                    line = line.replace('${annealing_loop}', str(annealing_loop))   
                    line = line.replace('${heating_steps}', str(heating_steps))   
                    line = line.replace('${cooling_steps}', str(cooling_steps))   
                    line = line.replace('${equaling_steps}', str(equaling_steps))   
                    line = line.replace('${current_time}', f"{self.urrent_time:.4f}") 
                    wf.write(line)    

    def _write_lmp_plumed(self, setting):
        plumed_file = setting["plumed_file"]
        md_steps = setting["md_steps"]
        self.current_time += md_steps*self.TIME_STEP*1.0E-6   

        if self.rank == 0:
            lines = open(self.lmp_plumedfile,'r').readlines()
            with open(f'lmp_{self.INIT_ITER}.plumed.in', 'w') as wf:
                for line in lines: 
                    line = line.replace('${T}', str(self.T))                                   
                    line = line.replace('${TIME_STEP}', str(self.TIME_STEP))   
                    line = line.replace('${plumed_file}', str(plumed_file))     
                    line = line.replace('${md_steps}', str(md_steps))
                    line = line.replace('${current_time}', f"{self.current_time:.4f}")   
                    wf.write(line)

    def _clear_lmp(self):
        self.lmp.command("clear")

    def _restart_lmp(self):
        self._clear_lmp()
        self._set_lmp()

    # ---------------------------
    # Minimization Process
    # ---------------------------
    def Min_process(self):
        # soft minimize
        self.lmp.command(f"min_style cg")
        # self.lmp.command(f"min_modify dmax 0.5 integrator verlet")
        # self.lmp.command(f"minimize 1e-2 1e-3 100 1000")
        # # strict minimize
        # self.lmp.command(f"min_modify line quadratic")
        self.lmp.command(f"minimize 1e-5 1e-7 500 1000")

    # ---------------------------
    # CGMD Process
    # ---------------------------
    def CGMD_process(self):
        self.logger.info(f"Starting major-iteraten, which contains {self.NVT_STEPS} NVT loops and {self.NPT_STEPS} NPT loops ")

        self.lmp.command("write_data lmp.data nocoeff")
        self._write_lmp_md()

    # ---------------------------
    # Enhance Sampling 
    # ---------------------------
    def enhance_sampling_process(self):
        for sampling_method, sampling_setting in zip(self.sampling_methods, self.sampling_settings):
            if sampling_method == "Annealing":
                self.lmp.command('write_data lmp.data nocoeff')
                self._write_lmp_annealing(sampling_setting)

            elif sampling_method == 'PIRMSD':
                with open(f'pirmsd_{self.INIT_ITER}.json', 'w') as f:
                    json.dump(sampling_setting, f)

            elif sampling_method == "PIEnvCV":
                self.lmp.command("write_data lmp.data nocoeff")
                self._write_lmp_plumed(sampling_setting)

    # ---------------------------
    # Core Simulation Logic
    # ---------------------------
    def main_process(self):
        # ---------------------------
        # System State Initialization
        # ---------------------------
        self.lmp = lammps("mpi")
        self._set_lmp()

        # ---------------------------
        # Molecular Dynamic  
        # ---------------------------
        if self.simulation_type == 'CGMD':
            try:
                self.CGMD_process()
            except Exception as e:
                self.logger.error(f"CGMD Simulation failed: {e}")
                sys.exit(1)

        # ---------------------------
        # Enhance Sampling 
        # ---------------------------
        if self.enhance_sampling:
            try:
                self.enhance_sampling_process()
            except Exception as e:
                self.logger.error(f"Enhance sampling failed: {e}")
                sys.exit(1)

if __name__ == "__main__":
    args = argparse_input()
    vMD = EnhanceSampling(args)
    vMD.main_process()