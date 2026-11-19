# N02 · Refine the silicate parameters

Run from `N02_GA_Process` with Slurm, GROMACS (`gmx_mpi`), Packmol, and a Python environment with NumPy, SciPy, MDAnalysis, Matplotlib, and tqdm.

1. Apply the N01 interactions to `workspace/files/system.itp` in GROMACS units. Set the five search ranges in `GA_main.py` and provide the N01 scaling factor separately. The batch script applies this factor to the selected interaction coefficients. Its default is 1.0.
2. Configure the executables and start the search. The first four arguments set population size, generations, GROMACS threads per trial, and analysis workers per trial.

   ```bash
   export PYTHONPATH="$PWD/scripts/scikit-opt${PYTHONPATH:+:$PYTHONPATH}"
   export PACKMOL_BIN='<path-to-packmol-executable>'
   export N02_LAMBDA='<fitted-lambda>'
   export N02_JOB_NAME='<slurm-job-name>'
   python GA_main.py 20 50 64 6 ./workspace ./scripts ./optimize.log
   ```

3. Read the fitted coefficients and fitness from `optimize.log`. Transfer the selected trial's silicate parameters and interactions to N03, retaining the N03 mixture composition.
