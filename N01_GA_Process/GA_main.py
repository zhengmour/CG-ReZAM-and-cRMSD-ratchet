"""Genetic-algorithm driver for CG parameter fitting.

The fitness combines three structural targets extracted from the CG trajectory:
1. Qn distribution topology landmarks.
2. Si-D-D-Si angle distribution.
3. Si-Si distance distribution.
"""
import sys, os
import math
import numpy as np
from sko.GA import RCGA
from sko.tools import set_run_mode
from scripts.aq_info import InfoFile
import subprocess
from time import sleep

# Experimental targets from Devreux et al. in (q, c) space.
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

VARIABLE_BOUNDS = [
  # [0.2000, 0.3200], # sigQ
  [2.0, 15.0],        # epsQ
  [0.05, 0.25],     # sigD
  [40.0, 150.0],      # epsD
  [0.15, 0.25],       # dVS 
  [0.1e-7, 0.1e-5],   # rep 
]

class OptimizeCoeffs():
  """Coordinate the GA search and submit MD jobs for each candidate."""

  def __init__(self, directory="./workspace", script_path="./scripts", logfile='./optimize.log', gmx_ntomp=64, pool_threads=6):
    self.lfs = None
    self.directory=directory
    self.script_path=script_path
    self.logfile = logfile
    self.gmx_ntomp=gmx_ntomp
    self.pool_threads=pool_threads

  def numerical_similarity(self, lfv, rfv, k):
    return k*np.linalg.norm(lfv-rfv)

  def _denormalize_coeffs(self, normalized_coeffs):
    """Map unit-interval GA parameters into the physical coefficient range."""
    physical_coeffs = []
    for coeff, bounds in zip(normalized_coeffs, VARIABLE_BOUNDS):
      physical_coeffs.append(coeff * (bounds[1] - bounds[0]) + bounds[0])
    physical_coeffs.insert(0, 0.27172)
    return physical_coeffs

  def _get_latest_slurm_status(self, job_name):
    """Return the latest Slurm state string, or None if no usable row exists yet."""
    sacct_output = subprocess.getoutput(
      "sacct --name {:s} -X -o JobID,JobName%20,State".format(job_name)
    )
    with open(self.logfile, 'a') as logf:
      logf.write("Waiting for {:s} to finish...\n{:s}\n".format(job_name, sacct_output))

    lines = [line.split() for line in sacct_output.splitlines() if line.strip()]
    for parts in reversed(lines):
      if len(parts) >= 3 and parts[0] != "JobID":
        return parts[2]
    return None

  def func(self, coeffs):
    physical_coeffs = self._denormalize_coeffs(coeffs)
    fitness = self.run_md(physical_coeffs)
    return fitness

  def fitness(self):
    # Scale structural observables to the same order of magnitude as the Qn terms.
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
        "*** Fitness details:\n"
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

    return result

  def run_md(self, coeffs):
    # coeffs: [sigma_Q, epsilon_Q, sigma_D, epsilon_D, d_vs, epsilon_rep]
    path = "{:s}/sigQ_{:010.6f}_epsQ_{:010.6f}_sigD_{:010.6f}_epsD_{:010.6f}_dVS_{:010.6f}_rep_{:010.6e}_lambda_{:010.6f}".format(self.directory, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6])
    coeffs_str = "{:010.6f} {:010.6f} {:010.6f} {:010.6f} {:010.6f} {:010.6e} {:010.6f}".format(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6])

    # Each coefficient set writes to a deterministic folder so repeated
    # evaluations can reuse an existing finished trajectory.
    if not os.path.exists(path):
      # Use a deterministic directory name so repeated evaluations can reuse results.
      os.makedirs(path, exist_ok=True)
      job_name = "zd-opt-{:d}".format(os.getpid())
      p = subprocess.Popen("sbatch --job-name={:s} --ntasks={:d} -o {:s}/job.%j.out -e {:s}/job.%j.err {:s}/run_gmx.sh {:s} {:s} {:d}".format(
          job_name, self.gmx_ntomp, path, path, self.script_path, coeffs_str, self.directory, self.gmx_ntomp), shell=True)
      while True:
        sleep(10)
        # Poll Slurm until the submitted job reaches a terminal state.
        status = self._get_latest_slurm_status(job_name)
        if status is None:
          sleep(110)
          continue
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
      print("=== Coefficients ===\nsigQ: {:010.6f}; epsQ: {:010.6f};\nsigD: {:010.6f}; epsD: {:010.6f};\ndVS: {:010.6f}; rep: {:010.6e}\n".format(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]), file=of)
      print("*** Fitness: {:f}  ***\n".format(fitness), file=of)
    return fitness

  def optimize_GA(self, pop_number, max_iter):
    set_run_mode(self.func, 'multiprocessing')
    ga = RCGA(func=self.func, n_dim=5, size_pop=pop_number, max_iter=max_iter, prob_mut=0.1, prob_cros=0.8, lb=[0.0, 0.0, 0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0, 1.0, 1.0], n_processes=pop_number)
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
  best_norm_coeffs = optimize._denormalize_coeffs(best_coeffs)
  with open(logfile, 'a') as logf:
    print(best_norm_coeffs, file=logf)
    print(best_fitness, file=logf)
    print("best_norm_coeffs: {:s} ; best_fitness: {:f}".format(' '.join(map(str, best_coeffs.tolist())), best_fitness[0]), file=logf)
