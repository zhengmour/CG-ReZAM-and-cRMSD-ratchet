#!/home/zhengda/anaconda3/envs/mda/bin/python
import os
import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis import distances, rdf
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from scipy.ndimage import median_filter
from scipy import interpolate
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial 
import itertools

n_dummies = 4

class InfoFile():
  """Read a trajectory and extract RDF / Qn descriptors used by the GA stage."""

  def __init__(self, topfile, trjfile=None, logfile=None, pool_threads=4, is_refer=None, is_gmx=True):
    self.dirname = os.path.dirname(topfile) or "."
    self.filename = os.path.basename(topfile).rsplit('.', 2)[0]
    self.log  = logfile or os.path.join(self.dirname, "info.log")
    self.pool_threads = pool_threads

    if is_refer is None:
      is_refer = trjfile is None and logfile is None

    if trjfile is None:
      trjfiles = ()
    elif isinstance(trjfile, str):
      trjfiles = (trjfile,) 
    else:
      trjfiles = tuple(trjfile)

    self.ftsos= self.dirname+"/condensation_Si-D-D-Si_angles_rdf_"+self.filename+'.xvg'
    self.fss= self.dirname+"/condensation_Si-Si_rdf_"+self.filename+'.xvg'
    self.fqn  = self.dirname+"/condensation_Qn_"+self.filename+'.xvg'

    if is_refer:
      # GROMACS reference structure
      if is_gmx:
        self.system = mda.Universe(topfile)
        self.u_Si: AtomGroup = self.system.select_atoms('name Si')
        self.u_D : AtomGroup = self.system.select_atoms('name DO1 DO2 DO3 DO4')
        self.u_C : AtomGroup = self.system.select_atoms('name C')
        self.u_N : AtomGroup = self.system.select_atoms('name N')
        self.u_Na: AtomGroup = self.system.select_atoms("name Na")
      # LAMMPS reference structure
      else:
        self.system = mda.Universe(topfile, topology_format='DATA', atom_style='id resid type charge x y z')
        self.u_Si: AtomGroup = self.system.select_atoms('type 1')
        self.u_D : AtomGroup = self.system.select_atoms('type 2')
        self.u_C : AtomGroup = self.system.select_atoms('type 4')        
        self.u_N : AtomGroup = self.system.select_atoms('type 5')
        self.u_Na: AtomGroup = self.system.select_atoms("type 6")
    else:
      if trjfile is None:
        raise ValueError("Trajectory mode requires 'trjfile'.")
      if is_gmx:
        self.system = mda.Universe(topfile, trjfiles)       
        self.u_Si: AtomGroup = self.system.select_atoms('name Si')
        self.u_D : AtomGroup = self.system.select_atoms('name DO1 DO2 DO3 DO4')
        self.u_C : AtomGroup = self.system.select_atoms('name C1 C2 C3 C4')
        self.u_N : AtomGroup = self.system.select_atoms('name N1')
        self.u_Na: AtomGroup = self.system.select_atoms('name NA*')
      else:
        self.system = mda.Universe(topfile, trjfiles, topology_format='DATA', format='lammpsdump', atom_style='id resid type charge x y z')
        self.u_Si: AtomGroup = self.system.select_atoms('type 1')
        self.u_D : AtomGroup = self.system.select_atoms('type 2')
        self.u_C : AtomGroup = self.system.select_atoms('type 4')
        self.u_N : AtomGroup = self.system.select_atoms('type 5')
        self.u_Na: AtomGroup = self.system.select_atoms("type 6")

  def calc_angle(self, si1, d1, si2, d2):
    """Return the Si-D-D-Si angle in radians."""
    v1 = d1 - si1
    v1_norm = v1.dot(v1)**0.5
    v2 = d2 - si2
    v2_norm = v2.dot(v2)**0.5
    return np.arccos(np.dot(v1,v2)/(v1_norm*v2_norm))

  def write_rdf_xvg(self, of, lst, bins, rdf_values, note):
    print("# Information", file=of)                                                          
    print(note, file=of)          
    for bin, value in zip(bins, rdf_values):
      lst[0].append(bin)
      lst[1].append(value)
      print("{:f} {:f}".format(bin, value), file=of)

  def _get_valid_savgol_window(self, data_length, preferred_window):
    """Return a valid odd window length for Savitzky-Golay smoothing."""
    if data_length < 3:
      return None

    window = min(preferred_window, data_length)
    if window % 2 == 0:
      window -= 1
    if window < 3:
      return None
    return window

  def _smooth_curve(self, values, preferred_window, polyorder=1):
    """Smooth data when enough points are available, else return raw values."""
    values = np.asarray(values)
    window = self._get_valid_savgol_window(len(values), preferred_window)
    if window is None or window <= polyorder:
      return values
    return savgol_filter(values, window, polyorder)

  def analyze_frame_qn(self, r_ss, r_dd, index=None):
    """Analyze one frame and return Tsos values together with Qn fractions."""
    self.system.trajectory[index]
    dims = self.system.dimensions
    si_coords = self.u_Si.positions
    d_coords  = self.u_D.positions
    si_mat = distances.contact_matrix(si_coords, cutoff=r_ss, box=dims)
    d_mat = distances.contact_matrix(d_coords, cutoff=r_dd, box=dims)
    for i in range(0, len(d_mat), n_dummies):
      d_mat[i:i+n_dummies, i:i+n_dummies] = False
    # Find neighboring dummy sites after removing intramolecular pairs.
    nb_dummies = [np.where(ele) for ele in d_mat]
    
    Tsos = []	
    for i, nb in enumerate(nb_dummies):
      if len(nb[0]) == 0:
        continue
      else:
        for j in nb[0]:
          try:
            Tsos.append(self.calc_angle(si_coords[int(np.floor(i/4))], d_coords[i], si_coords[int(np.floor(j/4))], d_coords[int(j)]))
          except Exception as error:
            print("Frame analysis error:", type(error).__name__, "-", error)

    Q0 = Q1 = Q2 = Q3 = Q4 = Q5 = 0 
    for ele in si_mat:
      # Remove self contacts when counting Qn connectivity.
      Q = np.sum(ele==True) - 1
      if Q == 0:
        Q0 += 1
      elif Q == 1:
        Q1 += 1
      elif Q == 2:
        Q2 += 1
      elif Q == 3:
        Q3 += 1
      elif Q == 4:
        Q4 += 1
      elif Q >= 5:
        Q5 += 1
    n_atoms = len(si_coords)
    C = 1/4*((Q1/n_atoms+2*(Q2/n_atoms)+3*(Q3/n_atoms)+4*(Q4/n_atoms)))
    if index is None:
      return (Tsos, None, Q0/n_atoms, Q1/n_atoms, Q2/n_atoms, Q3/n_atoms, Q4/n_atoms, Q5/n_atoms, C)
    return (Tsos, self.system.trajectory.ts.time, Q0/n_atoms, Q1/n_atoms, Q2/n_atoms, Q3/n_atoms, Q4/n_atoms, Q5/n_atoms, C)

  def calc_rdf(self):
    self.Rss_rdf = [[],[]]
    self.Tsos_rdf = [[],[]]

    with open(self.fss, 'w') as of, open(self.log, 'a') as logf:
      logf.write("# Starting RDF analysis...\n")
      ss_rdf = rdf.InterRDF(self.u_Si, self.u_Si, nbins=500, range=(0.5,15.0))
      ss_rdf.run(start=-100)
      self.r_ss_rdf = [ss_rdf.bins, ss_rdf.rdf]	

      self.write_rdf_xvg(of, self.Rss_rdf, ss_rdf.bins, ss_rdf.rdf, '@    title RDF of Si-Si\n@    xaxis  label "distance"\n@    yaxis  label "RDF"\n'
          '@TYPE xy\n@ view 0.15, 0.15, 0.75, 0.85\n')
      max_index = np.argmax(ss_rdf.rdf)
      self.r_ss = ss_rdf.bins[max_index]
      logf.write("*** Peak Si-Si distance: {:4.2f}\n".format(self.r_ss))
      plt.plot(ss_rdf.bins, ss_rdf.rdf)
      plt.savefig(self.dirname+"/condensation_Si-Si_"+self.filename+'.png', dpi=1200)
      plt.cla()

    dd_rdf = rdf.InterRDF(self.u_D, self.u_D, nbins=500, range=(0.5,15.0))
    dd_rdf.run(start=-100)
    max_value_index = np.argmax(dd_rdf.rdf)
    self.r_dd = dd_rdf.bins[max_value_index]
    
    extra_frame = len(self.system.trajectory[-100:])
    n_frames = len(self.system.trajectory)
    start_frame = n_frames - extra_frame
    indices = list(range(start_frame, n_frames))
    with Pool(processes = self.pool_threads) as pool:
      result = list(tqdm.tqdm(pool.imap(partial(self.analyze_frame_qn, self.r_ss+0.2, self.r_dd+0.12), indices), total=extra_frame))

    Tsos_all = []
    q5 = []
    for Tsos, t, Q0, Q1, Q2, Q3, Q4, Q5, C in result:  
      Tsos_all.append(Tsos)
      q5.append(Q5)
    self.max_q5 = np.max(q5)

    with open(self.ftsos, 'w') as of, open(self.log, 'a') as logf:
      Tsos_all = np.array(list(itertools.chain(*Tsos_all)))*180/np.pi
      hist, bin_edges = np.histogram(Tsos_all, bins=880, range=(0,220.0))
      bins = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0) 
      self.write_rdf_xvg(of, self.Tsos_rdf, bins, hist, '@    title RDF of Si-D-D-Si\n@    xaxis  label "degree"\n@    yaxis  label "RDF"\n'
          '@TYPE xy\n@ view 0.15, 0.15, 0.75, 0.85\n')

      hist_smooth = self._smooth_curve(hist, 51)
      peaks,props = find_peaks(hist_smooth,prominence=max(hist_smooth)*0.02)
      valleys,props = find_peaks(-hist_smooth,prominence=max(hist_smooth)*0.02)
      peak_intensity = []
      for peak in peaks:
        lv = np.array([v for v in valleys if v < peak])
        if lv.any():
          lv = np.max(lv)
        else:
          lv = 0
        rv = np.array([v for v in valleys if v > peak])
        if rv.any():
          rv = np.min(rv)
        else:
          rv = -1
        peak_intensity.append(simpson(hist_smooth[lv:rv], x=bins[lv:rv]))
      plt.plot(bins, hist_smooth)
      plt.scatter(bins[peaks], hist_smooth[peaks], color="red")
      plt.scatter(bins[valleys], hist_smooth[valleys], color="green")
      plt.savefig(self.dirname+"/condensation_Si-D-D-Si_"+self.filename+'.png', dpi=1200)
      plt.cla()

      if peak_intensity:
        max_index = max(enumerate(peak_intensity), key=lambda x: x[1])[0]
        self.t_sos = bins[peaks[max_index]]
        logf.write("*** Peak Si-D-D-Si angle: {:.2f}\n".format(self.t_sos))
      else:
        self.t_sos = 0.00
        logf.write("*** ERROR: unable to determine a peak Si-D-D-Si angle.\n")
    
  def calc_env(self, filename):
    with open(filename, 'r') as rf:
      words = rf.readlines()[-1].split()
      self.mean = float(words[1])
      self.morethan = float(words[2])
    
  def calc_rdf_qn(self):
    self.Rss_rdf = [[],[]]
    self.Tsos_rdf = [[],[]]
    times = []
    q0 = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    c = []
    self.max_c       =  0.0
    self.max_q0      =  np.array([0.0, 0.0])
    self.max_q1      =  np.array([0.0, 0.0])
    self.max_q2      =  np.array([0.0, 0.0])
    self.max_q3      =  np.array([0.0, 0.0])
    self.max_q4      =  np.array([0.0, 0.0])
    self.max_q5      =  np.array([0.0, 0.0])
    self.cross_q0_q1 =  []
    self.cross_q1_q2 =  []
    self.cross_q2_q3 =  []
    self.cross_q3_q4 =  []
    self.cross_q0_q2 =  []
    self.cross_q1_q3 =  []
    self.cross_q2_q4 =  []
    self.cross_q0_q3 =  []
    self.cross_q1_q4 =  []
    self.cross_q0_q4 =  []

    with open(self.fss, 'w') as of, open(self.log, 'a') as logf:
      logf.write("# Starting RDF analysis...\n")
      ss_rdf = rdf.InterRDF(self.u_Si, self.u_Si, nbins=500, range=(0.5,15.0))
      ss_rdf.run(start=-100)
      self.r_ss_rdf = [ss_rdf.bins, ss_rdf.rdf]	

      self.write_rdf_xvg(of, self.Rss_rdf, ss_rdf.bins, ss_rdf.rdf, '@    title RDF of Si-Si\n@    xaxis  label "distance"\n@    yaxis  label "RDF"\n'
          '@TYPE xy\n@ view 0.15, 0.15, 0.75, 0.85\n')
      max_index = np.argmax(ss_rdf.rdf)
      self.r_ss = ss_rdf.bins[max_index]
      logf.write("*** Peak Si-Si distance: {:4.2f}\n".format(self.r_ss))
      plt.plot(ss_rdf.bins, ss_rdf.rdf)
      plt.savefig(self.dirname+"/condensation_Si-Si_"+self.filename+'.png', dpi=1200)
      plt.cla()

    dd_rdf = rdf.InterRDF(self.u_D, self.u_D, nbins=500, range=(0.5,15.0))
    dd_rdf.run(start=-100)
    max_value_index = np.argmax(dd_rdf.rdf)
    self.r_dd = dd_rdf.bins[max_value_index]

    Tsos_all = []
    with open(self.fqn, 'w') as of, open(self.log, 'a') as logf:          
      print("# Andre Carvalho\n# MolModel Group\n# Universidade de Aveiro\n# andre.dc@ua.pt", file=of)
      print('@    title "Silica condensation"\n@    xaxis  label "time"\n@    yaxis  label "qi"\n@TYPE xy\n@ view 0.15, 0.15, 0.75, 0.85\n@ legend on\n@ legend box on\n@ legend loctype view\n@ legend 0.78, 0.8\n@ legend length 2\n@ s0 legend "Q0"\n@ s1 legend "Q1"\n@ s2 legend "Q2"\n@ s3 legend "Q3"\n@ s4 legend "Q4"\n@ s5 legend "Qn"\n@ s6 legend "C"\n@ s0 line color "black"\n@ s1 line color "red"\n@ s2 line color "blue"\n@ s3 line color "magenta"\n@ s4 line color "brown"\n@ s5 line color "yellow"\n@ s6 line color "green"\n', file=of)
      
      logf.write("# Starting Qn analysis...\n")
      
      with Pool(processes = self.pool_threads) as pool:
        result = list(tqdm.tqdm(pool.imap(partial(self.analyze_frame_qn, self.r_ss+0.2, self.r_dd+0.12), list(range(len(self.system.trajectory)))), total=len(self.system.trajectory)))

      logf.write("# Aggregating per-frame analysis results...\n")
      for Tsos, t, Q0, Q1, Q2, Q3, Q4, Q5, C in result:  
        Tsos_all.append(Tsos)
        times.append(t)
        q0.append(Q0)
        q1.append(Q1)
        q2.append(Q2)
        q3.append(Q3)
        q4.append(Q4)
        q5.append(Q5)
        c.append(C)
        print("{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format( t, Q0, Q1, Q2, Q3, Q4, Q5, C), file=of)

    with open(self.ftsos, 'w') as of, open(self.log, 'a') as logf:
      Tsos_all = np.array(list(itertools.chain(*Tsos_all)))*180/np.pi
      hist, bin_edges = np.histogram(Tsos_all, bins=880, range=(0,220.0))
      bins = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0) 
      self.write_rdf_xvg(of, self.Tsos_rdf, bins, hist, '@    title RDF of Si-D-D-Si\n@    xaxis  label "degree"\n@    yaxis  label "RDF"\n'
          '@TYPE xy\n@ view 0.15, 0.15, 0.75, 0.85\n')
      hist_smooth = self._smooth_curve(hist, 51)
      peaks,props = find_peaks(hist_smooth,prominence=max(hist_smooth)*0.02)
      valleys,props = find_peaks(-hist_smooth,prominence=max(hist_smooth)*0.02)
      peak_intensity = []
      for peak in peaks:
        lv = np.array([v for v in valleys if v < peak])
        if lv.any():
          lv = np.max(lv)
        else:
          lv = 0
        rv = np.array([v for v in valleys if v > peak])
        if rv.any():
          rv = np.min(rv)
        else:
          rv = -1
        peak_intensity.append(simpson(hist_smooth[lv:rv], x=bins[lv:rv]))
      plt.plot(bins, hist_smooth)
      plt.scatter(bins[peaks], hist_smooth[peaks], color="red")
      plt.scatter(bins[valleys], hist_smooth[valleys], color="green")
      plt.savefig(self.dirname+"/condensation_Si-D-D-Si_"+self.filename+'.png', dpi=1200)
      plt.cla()
      if peak_intensity:
        max_index = max(enumerate(peak_intensity), key=lambda x: x[1])[0]
        self.t_sos = bins[peaks[max_index]]
        logf.write("*** Peak Si-D-D-Si angle: {:.2f}\n".format(self.t_sos))
      else:
        self.t_sos = 0.00
        logf.write("*** ERROR: unable to determine a peak Si-D-D-Si angle.\n")

      self.max_c = np.max(c)
      self.max_q0 = np.array([np.max(q0), c[q0.index(np.max(q0))]])
      self.max_q1 = np.array([np.max(q1), c[q1.index(np.max(q1))]]) 
      self.max_q2 = np.array([np.max(q2), c[q2.index(np.max(q2))]]) 
      self.max_q3 = np.array([np.max(q3), c[q3.index(np.max(q3))]]) 
      self.max_q4 = np.array([np.max(q4), c[q4.index(np.max(q4))]]) 
      self.max_q5 = np.array([np.max(q5), c[q5.index(np.max(q5))]])   
      # Compare the simulated (c, q) curves across all Qn states.
      plt.plot(c, q0, label='q0')
      plt.plot(c, q1, label='q1')
      plt.plot(c, q2, label='q2')
      plt.plot(c, q3, label='q3')
      plt.plot(c, q4, label='q4')
      plt.plot(c, q5, label='q5')
      plt.legend()
      plt.savefig(self.dirname+"/condensation_c-q_"+self.filename+'.png', dpi=1200)
      plt.cla()

      c_index_delete = []
      for i in range(len(c)):
        if c[i] in c[:i]:
          c_index_delete.append(i)
      for counter, index in enumerate(c_index_delete):
        index = index - counter
        c.pop(index)
        q0.pop(index)
        q1.pop(index)
        q2.pop(index)
        q3.pop(index)
        q4.pop(index)
        q5.pop(index)
      c_q0_zip_sorted = sorted(zip(c, q0))
      c_q1_zip_sorted = sorted(zip(c, q1))
      c_q2_zip_sorted = sorted(zip(c, q2))
      c_q3_zip_sorted = sorted(zip(c, q3))
      c_q4_zip_sorted = sorted(zip(c, q4))
      c_q5_zip_sorted = sorted(zip(c, q5))
      c = sorted(c)
      _, q0 = zip(*c_q0_zip_sorted)
      _, q1 = zip(*c_q1_zip_sorted)
      _, q2 = zip(*c_q2_zip_sorted)
      _, q3 = zip(*c_q3_zip_sorted)
      _, q4 = zip(*c_q4_zip_sorted)
      _, q5 = zip(*c_q5_zip_sorted)      
      q0_filter = median_filter(q0, 10) 
      q1_filter = median_filter(q1, 10) 
      q2_filter = median_filter(q2, 10) 
      q3_filter = median_filter(q3, 10) 
      q4_filter = median_filter(q4, 10)
      q5_filter = median_filter(q5, 10)   
      q0_smooth = self._smooth_curve(q0_filter, 11)
      q1_smooth = self._smooth_curve(q1_filter, 11)
      q2_smooth = self._smooth_curve(q2_filter, 11)
      q3_smooth = self._smooth_curve(q3_filter, 11)
      q4_smooth = self._smooth_curve(q4_filter, 11)
      q5_smooth = self._smooth_curve(q5_filter, 11)
      plt.plot(c, q0_smooth, label='q0')
      plt.plot(c, q1_smooth, label='q1')
      plt.plot(c, q2_smooth, label='q2')
      plt.plot(c, q3_smooth, label='q3')
      plt.plot(c, q4_smooth, label='q4')
      plt.plot(c, q5_smooth, label='q5')
      plt.legend()
      plt.savefig('c-q-smooth.png', dpi=1200)
      plt.cla()
      X_Y_cross = self.crossing_points_list(c, q0, c, q1, "cross_q0_q1")
      if X_Y_cross.size == 0:         self.cross_q0_q1 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]
        self.cross_q0_q1.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q1, c, q2, "cross_q1_q2")
      if X_Y_cross.size == 0:         self.cross_q1_q2 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]
        self.cross_q1_q2.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q2, c, q3, "cross_q2_q3")
      if X_Y_cross.size == 0:         self.cross_q2_q3 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q2_q3.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q3, c, q4, "cross_q3_q4")
      if X_Y_cross.size == 0:         self.cross_q3_q4 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q3_q4.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q0, c, q2, "cross_q0_q2")
      if X_Y_cross.size == 0:         self.cross_q0_q2 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q0_q2.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q1, c, q3, "cross_q1_q3")
      if X_Y_cross.size == 0:         self.cross_q1_q3 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q1_q3.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q2, c, q4, "cross_q2_q4")
      if X_Y_cross.size == 0:         self.cross_q2_q4 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q2_q4.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q0, c, q3, "cross_q0_q3")
      if X_Y_cross.size == 0:         self.cross_q0_q3 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q0_q3.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q1, c, q4, "cross_q1_q4")
      if X_Y_cross.size == 0:         self.cross_q1_q4 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q1_q4.append([y, x])
      X_Y_cross = self.crossing_points_list(c, q0, c, q4, "cross_q0_q4")
      if X_Y_cross.size == 0:         self.cross_q0_q4 = [[0.0, 1.0]]
      else:
        x, y = X_Y_cross[0]  
        self.cross_q0_q4.append([y, x])
      self.cross_q0_q1 = np.array(self.cross_q0_q1) 
      self.cross_q1_q2 = np.array(self.cross_q1_q2)
      self.cross_q2_q3 = np.array(self.cross_q2_q3)
      self.cross_q3_q4 = np.array(self.cross_q3_q4)
      self.cross_q0_q2 = np.array(self.cross_q0_q2)
      self.cross_q1_q3 = np.array(self.cross_q1_q3)
      self.cross_q2_q4 = np.array(self.cross_q2_q4)
      self.cross_q0_q3 = np.array(self.cross_q0_q3)
      self.cross_q1_q4 = np.array(self.cross_q1_q4)
      self.cross_q0_q4 = np.array(self.cross_q0_q4)

      print("*** Qn summary:\n"
                    "max_c =  ", self.max_c,  " \n"
                    "max_q0 = ", self.max_q0, " \n"
                    "max_q1 = ", self.max_q1, " \n"
                    "max_q2 = ", self.max_q2, " \n"
                    "max_q3 = ", self.max_q3, " \n"
                    "max_q4 = ", self.max_q4, " \n"
                    "max_q5 = ", self.max_q5, " \n", file=logf)
      print("cross_q0_q1 = ", self.cross_q0_q1, " \n"
            "cross_q1_q2 = ", self.cross_q1_q2, " \n"
            "cross_q2_q3 = ", self.cross_q2_q3, " \n"
            "cross_q3_q4 = ", self.cross_q3_q4, " \n"
            "cross_q0_q2 = ", self.cross_q0_q2, " \n"
            "cross_q1_q3 = ", self.cross_q1_q3, " \n"
            "cross_q2_q4 = ", self.cross_q2_q4, " \n"
            "cross_q0_q3 = ", self.cross_q0_q3, " \n"
            "cross_q1_q4 = ", self.cross_q1_q4, " \n"
            "cross_q0_q4 = ", self.cross_q0_q4, " \n", file=logf)

  def find_crossing_points(self, X1, Y1, X2, Y2, gname):
    """Smooth two curves and return their intersections."""
    Y1_smooth = self._smooth_curve(Y1, 51)
    Y2_smooth = self._smooth_curve(Y2, 51)
    
    plt.plot(X1, Y1_smooth)
    plt.plot(X2, Y2_smooth)
    plt.savefig(self.dirname+"/condensation_" + gname + '_' + self.filename+'.png', dpi=1200)
    plt.cla()

    X, Y = self.find_intersections_from_curves(X1, Y1_smooth, X2, Y2_smooth)
    return np.array([[x, y] for x, y in zip(X, Y)])      

  def crossing_points_list(self, X1, Y1, X2, Y2, gname):
    """Backward-compatible wrapper around :meth:`find_crossing_points`."""
    return self.find_crossing_points(X1, Y1, X2, Y2, gname)

  def find_intersections_from_curves(self, X1, Y1, X2, Y2):
    """Return the intersections of two curves defined on increasing X grids."""
    # https://zhuanlan.zhihu.com/p/358435456
    if np.all(np.diff(X1) > 0) == True:
        pass
    else:
        raise Exception('X1 must be strictly monotonically incremented data!')
    
    if np.all(np.diff(X2) > 0) == True:
        pass
    else:
        raise Exception('X2 must be strictly monotonically incremented data!')
    X_new = X1
    Y_new = Y1 - Y2 

    inersections_X = self.find_roots_from_xy(X=X_new, Y=Y_new)
    inersections_Y = np.interp(inersections_X, X1, Y1)
    return inersections_X, inersections_Y

  def numpy_scipy_find_inersections_by_X1Y1X2Y2(self, X1, Y1, X2, Y2):
    """Backward-compatible wrapper around :meth:`find_intersections_from_curves`."""
    return self.find_intersections_from_curves(X1, Y1, X2, Y2)

  def find_roots_from_xy(self, X, Y):
    """Find roots of a 1D curve by converting a linear spline into a PPoly."""
    if np.all(np.diff(X) > 0) == True:
        pass
    else:
        raise Exception('X It must be strictly monotonous and incremental data!')

    # BSpline(builtins.object)
    # tck: A spline, as returned by `splrep` or a BSpline object.
    tck = interpolate.make_interp_spline(x=X, y=Y, k=1) 

    # class PPoly(_PPolyBase); Construct a piecewise polynomial from a spline
    piecewise_polynomial = interpolate.PPoly.from_spline(tck, extrapolate=None)

    roots_X_ = piecewise_polynomial.roots() # class ndarray(builtins.object)
    roots_X = roots_X_[np.where(np.logical_and(roots_X_>=X[0], roots_X_<=X[-1]))]

    return roots_X

  def numpy_scipy_find_roots_by_XY(self, X, Y):
    """Backward-compatible wrapper around :meth:`find_roots_from_xy`."""
    return self.find_roots_from_xy(X, Y)
  
if __name__ == "__main__":
  raise SystemExit(
    "This module is intended to be imported. Instantiate InfoFile with explicit "
    "topology and trajectory paths from another script."
  )
  
