"""
1. Draw the simulated Qn distribution, scale the abscissa and compare it with the reference Qn, and return the similarity of the two curves
https://zhuanlan.zhihu.com/p/610535464
https://github.com/nelsonwenner/shape-similarity.git
2. Compare the similarity between the Si-O-Si angular distribution and the reference angular distribution
3. Si-Si distance distribution and experimental values 3.10. Similarity of reference distribution
"""
import sys, os
import math
import numpy as np
from shapesimilarity import shape_similarity
from sko.PSO import PSO
from sko.GA import RCGA
from sko.tools import set_run_mode
from scripts.aq_info import InfoFile
import subprocess
from time import sleep

GPU_SERVER = False
CPU_SERVER = True

#Valores experimentais de Devreux et al. (q, c)
d_max_c       = .9
d_max_q0      = np.array([1.0, .00])
d_max_q1      = np.array([.58, .25])
d_max_q2      = np.array([.58, .52])
d_max_q3      = np.array([.60, .77])
d_max_q4      = np.array([.47, .90])
d_cross_q0_q1 = np.array([.45, .17])
d_cross_q1_q2 = np.array([.45, .41])
d_cross_q2_q3 = np.array([.44, .64])
d_cross_q3_q4 = np.array([.50, .87])
d_cross_q0_q2 = np.array([.20, .26])
d_cross_q1_q3 = np.array([.22, .32])
d_cross_q2_q4 = np.array([.22, .75])
d_cross_q0_q3 = np.array([.06, .36])
d_cross_q1_q4 = np.array([.07, .62])
d_cross_q0_q4 = np.array([.01, .46])

# The Si-O-Si bond angle and Si-Si bond length in the experiment
r_ss = 3.05
t_sos = 153.0
k = 100.0

variables = [
  # [0.2000, 0.3200], # sigQ
  [2.0, 15.0],        # epsQ
  [0.05, 0.25],     # sigD
  [40.0, 150.0],      # epsD
  [0.15, 0.25],       # dVS 
  [0.1e-7, 0.1e-5],   # rep 
]

class OptimizeCoeffs():
  def __init__(self, directory="./workspace", script_path="./scripts", logfile='./optimize.log', gmx_ntomp=64, pool_threads=6, refer_gmx=None, refer_lmp=None):
    self.lfs = None
    self.directory=directory
    self.script_path=script_path
    self.logfile = logfile
    self.gmx_ntomp=gmx_ntomp
    self.pool_threads=pool_threads

    # Whether to use gromacs and reaxff simulation results as a comparison
    if not refer_gmx is None:
      self.rfgmx = InfoFile(refer_gmx)
      self.rfgmx.calc_rdf_qn()
    else:
      self.rfgmx = None

    if not refer_lmp is None:
      self.rflmp = InfoFile(refer_lmp)
      self.rflmp.calc_rdf_qn()
    else:
      self.rflmp = None

  # The number of LFS and RFS nodes must be the same
  def curve_similarity(self, lfs, rfs, k):
    shape1 = np.column_stack((lfs[0], lfs[1]))
    shape2 = np.column_stack((rfs[0], rfs[1]))
    return k*(1.0-shape_similarity(shape1, shape2))

  def numerical_similarity(self, lfv, rfv, k):
    return k*np.linalg.norm(lfv-rfv)

  def log_numerical_similarity(self, lfv, rfv, k):
    return -k*math.log(np.linalg.norm(lfv-rfv)/2)

  def func(self, coeffs):
    new_coeffs = []
    for coeff, variable in zip(coeffs, variables):
      new_coeffs.append(coeff*(variable[1]-variable[0])+variable[0])
    new_coeffs.insert(0, 0.27172)
    fitness = self.run_md(new_coeffs)
    return fitness

  def fitness(self):
    # r_ss, θ_sos becomes an order of magnitude with qn
    result_r_ss  = self.numerical_similarity(self.lfs.r_ss/r_ss,   1.0, k)
    result_t_sos = self.numerical_similarity(self.lfs.t_sos/t_sos, 1.0, k)

    result_max_c  = self.numerical_similarity(self.lfs.max_c      , d_max_c      , k)
    result_max_q0 = self.numerical_similarity(self.lfs.max_q0     , d_max_q0     , k)
    result_max_q1 = self.numerical_similarity(self.lfs.max_q1     , d_max_q1     , k)
    result_max_q2 = self.numerical_similarity(self.lfs.max_q2     , d_max_q2     , k)
    result_max_q3 = self.numerical_similarity(self.lfs.max_q3     , d_max_q3     , k)
    result_max_q4 = self.numerical_similarity(self.lfs.max_q4     , d_max_q4     , k)
    result_max_q5 = self.numerical_similarity(self.lfs.max_q5[0]  , 0.0          , k*20)

    result_cross_q0_q1 = result_cross_q1_q2 = result_cross_q2_q3 = result_cross_q3_q4 = 0.0
    result_cross_q0_q2 = result_cross_q1_q3 = result_cross_q2_q4 = 0.0
    result_cross_q0_q3 = result_cross_q1_q4 = result_cross_q0_q4 = 0.0
    for cross_q0_q1 in self.lfs.cross_q0_q1:  result_cross_q0_q1 += self.numerical_similarity(cross_q0_q1, d_cross_q0_q1, 2*k)
    for cross_q1_q2 in self.lfs.cross_q1_q2:  result_cross_q1_q2 += self.numerical_similarity(cross_q1_q2, d_cross_q1_q2, 2*k)
    for cross_q2_q3 in self.lfs.cross_q2_q3:  result_cross_q2_q3 += self.numerical_similarity(cross_q2_q3, d_cross_q2_q3, k)
    for cross_q3_q4 in self.lfs.cross_q3_q4:  result_cross_q3_q4 += self.numerical_similarity(cross_q3_q4, d_cross_q3_q4, k)
    for cross_q0_q2 in self.lfs.cross_q0_q2:  result_cross_q0_q2 += self.numerical_similarity(cross_q0_q2, d_cross_q0_q2, 2*k)
    for cross_q1_q3 in self.lfs.cross_q1_q3:  result_cross_q1_q3 += self.numerical_similarity(cross_q1_q3, d_cross_q1_q3, k)
    for cross_q2_q4 in self.lfs.cross_q2_q4:  result_cross_q2_q4 += self.numerical_similarity(cross_q2_q4, d_cross_q2_q4, k)
    for cross_q0_q3 in self.lfs.cross_q0_q3:  result_cross_q0_q3 += self.numerical_similarity(cross_q0_q3, d_cross_q0_q3, 2*k)
    for cross_q1_q4 in self.lfs.cross_q1_q4:  result_cross_q1_q4 += self.numerical_similarity(cross_q1_q4, d_cross_q1_q4, k)
    for cross_q0_q4 in self.lfs.cross_q0_q4:  result_cross_q0_q4 += self.numerical_similarity(cross_q0_q4, d_cross_q0_q4, 2*k)
    with open(self.logfile, 'a') as logf:
      logf.write(
        "*** Infomation of Fitness: \n"
        "r_ss         = {:.2f} {:.2f}\n"
        "t_sos        = {:.2f} {:.2f}\n"
        "max_c        = {:.2f} {:.2f}\n"
        "max_q0       = {:.2f} {:.2f} {:.2f}\n"
        "max_q1       = {:.2f} {:.2f} {:.2f}\n"
        "max_q2       = {:.2f} {:.2f} {:.2f}\n"
        "max_q3       = {:.2f} {:.2f} {:.2f}\n"
        "max_q4       = {:.2f} {:.2f} {:.2f}\n"
        "max_q5       = {:.2f} {:.2f} {:.2f}\n"
        "cross_q0_q1  = {:.2f}\n"
        "cross_q1_q2  = {:.2f}\n"
        "cross_q2_q3  = {:.2f}\n"
        "cross_q3_q4  = {:.2f}\n"
        "cross_q0_q2  = {:.2f}\n"
        "cross_q1_q3  = {:.2f}\n"
        "cross_q2_q4  = {:.2f}\n"
        "cross_q0_q3  = {:.2f}\n"
        "cross_q1_q4  = {:.2f}\n"
        "cross_q0_q4  = {:.2f}\n".format(
          self.lfs.r_ss, result_r_ss, self.lfs.t_sos, result_t_sos, self.lfs.max_c, result_max_c, # type: ignore
          self.lfs.max_q0[0], self.lfs.max_q0[1], result_max_q0, self.lfs.max_q1[0], self.lfs.max_q1[1], result_max_q1, self.lfs.max_q2[0], self.lfs.max_q2[1], result_max_q2, self.lfs.max_q3[0], self.lfs.max_q3[1], result_max_q3, # type: ignore
          self.lfs.max_q4[0], self.lfs.max_q4[1], result_max_q4, self.lfs.max_q5[0], self.lfs.max_q5[1], result_max_q5, result_cross_q0_q1, result_cross_q1_q2, result_cross_q2_q3, result_cross_q3_q4, # type: ignore
          result_cross_q0_q2, result_cross_q1_q3, result_cross_q2_q4, result_cross_q0_q3, result_cross_q1_q4, result_cross_q0_q4
        ))
    result = result_r_ss+ result_t_sos+ result_max_c + result_max_q0+ result_max_q1+ result_max_q2+ result_max_q3+ \
          result_max_q4+ result_max_q5+ result_cross_q0_q1+ result_cross_q1_q2+ result_cross_q2_q3+ result_cross_q3_q4+ \
          result_cross_q0_q2+ result_cross_q1_q3+ result_cross_q2_q4+ result_cross_q0_q3+ result_cross_q1_q4+ result_cross_q0_q4

    # Comparison with RDF of reference structures
    if not self.rflmp is None:
      # result += self.curve_similarity(self.lfs.Tsos, self.rflmp.Tsos)
      result += self.curve_similarity(self.lfs.Rss , self.rflmp.Rss , k)
    if not self.rfgmx is None:
      result += self.curve_similarity(self.lfs.Rsc , self.rfgmx.Rsc , k)
      result += self.curve_similarity(self.lfs.Rsn , self.rfgmx.Rsn , k)
      result += self.curve_similarity(self.lfs.Rso , self.rfgmx.Rso , k)
      result += self.curve_similarity(self.lfs.Rsna, self.rfgmx.Rsna, k)

    return result

  def run_md(self, coeffs):
    # coeffs: [sigma_Q, epsion_Q, sigma_D, epsilon_D, d_vs, epsilon_rep]
    path = "{:s}/sigQ_{:010.6f}_epsQ_{:010.6f}_sigD_{:010.6f}_epsD_{:010.6f}_dVS_{:010.6f}_rep_{:010.6e}_lambda_{:010.6f}".format(self.directory, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6])
    coeffs_str = "{:010.6f} {:010.6f} {:010.6f} {:010.6f} {:010.6f} {:010.6e} {:010.6f}".format(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6])

    # If the folder does not exist, the simulation is performed
    if not os.path.exists(path):
      # Enter the PID of the process as a flag for each task
      os.system("mkdir {:s}".format(path))
      p = subprocess.Popen("sbatch --job-name=zd-opt-{:d} --ntasks={:d} -o {:s}/job.%j.out -e {:s}/job.%j.err {:s}/run_gmx.sh {:s} {:s} {:d}".format(
          os.getpid(), self.gmx_ntomp, path, path, self.script_path, coeffs_str, self.directory, self.gmx_ntomp), shell=True)
      while True:
        sleep(10)
        # Use sacct to view the details of the task
        sacct_output = subprocess.getoutput("sacct --name zd-opt-{:d} -X -o JobID,JobName%20,State".format(os.getpid()))
        with open(self.logfile, 'a') as logf:
          logf.write("waiting {:d} completed...\n{:s}\n".format(os.getpid(), sacct_output))
        status = sacct_output.split('\n')[-1].split()[2]
        if status.startswith("COMPLETED") or status.startswith("FAILED") or status.startswith("CANCELLED"):
          break
        sleep(110)
      p.wait()

    if os.path.exists(path+"/md_cg_si.xtc"):
      self.lfs = InfoFile(path+"/md_cg_si.gro", path+"/md_cg_si.xtc", path+"/round.log", self.pool_threads, is_gmx=True)
      self.lfs.calc_rdf_qn()
      fitness = self.fitness()
    else:
      fitness = math.inf

    with open(self.logfile, 'a') as of:
      print("=== coeffs ===\n sigQ: {:010.6f}; epsQ: {:010.6f};\n sigD: {:010.6f}; epsD: {:010.6f};\n dVS: {:010.6f}; rep: {:010.6e}\n}".format(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]), file=of)
      print("*** Fitness: {:f}  ***\n".format(fitness), file=of)
    return fitness

  def optimize_PSO(self, pop_number, max_iter):
    # https://zhuanlan.zhihu.com/p/346355572
    # https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    # pyPSO、scikit-opt、deap
    set_run_mode(self.func, 'multiprocessing')
    pso = PSO(func=self.func, n_dim=6, pop=pop_number, max_iter=max_iter, lb=[0.0, 0.0, 0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0, 1.0, 1.0], w=0.8, c1=0.5, c2=0.5)
    pso.run()
    return pso.gbest_x, pso.gbest_y

  def optimize_GA(self, pop_number, max_iter):
    set_run_mode(self.func, 'multiprocessing')
    ga = RCGA(func=self.func, n_dim=6, size_pop=pop_number, max_iter=max_iter, prob_mut=0.1, prob_cros=0.8, lb=[0.0, 0.0, 0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0, 1.0, 1.0], n_processes=pop_number)
    best_x, best_y = ga.run()
    return best_x, best_y

if __name__ == "__main__":
  pop_number = int(sys.argv[1])
  max_iter = int(sys.argv[2])
  gmx_ntomp = int(sys.argv[3])
  pool_threads = int(sys.argv[4])

  directory = sys.argv[5]
  script_path = sys.argv[6]
  logfile = sys.argv[7]

  optimize = OptimizeCoeffs(directory=directory, script_path=script_path, logfile=logfile, gmx_ntomp=gmx_ntomp, pool_threads=pool_threads)
  best_coeffs, best_fitness = optimize.optimize_GA(pop_number, max_iter)
  best_norm_coeffs = []
  for coeff, variable in zip(best_coeffs, variables):
    best_norm_coeffs.append(coeff*(variable[1]-variable[0])+variable[0])
  best_norm_coeffs.insert(0, 0.27172)
  with open(logfile, 'a') as logf:
    print(best_norm_coeffs, file=logf)
    print(best_fitness, file=logf)
    print("best_norm_coeffs: {:s} ; best_fitness: {:f}".format(' '.join(map(str, best_coeffs.tolist())), best_fitness[0]), file=logf)
