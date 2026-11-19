# N01 · Fit coarse-grained interactions

Run from `N01_SPO_Process`.

1. Set the reference-data directories, LAMMPS commands, and scheduler resources in `config/workflow.json`. Set the initial parameters in `config/elements.json`. Supply the reference RDFs and the starting LAMMPS system in the configured directories.
2. Place completed Gaussian calculations under `data_set/refer_QM/`. To generate inputs first, run the command below for each required species pair and submit the generated `sbatch.sh` from each sample directory.

   ```bash
   python scripts/expand_eight_directions.py --center opt_gro/SiH.gro --around opt_gro/Na.gro --charge 1
   ```

3. Map the samples, index the reference data, check the inputs, and run the fit.

   ```bash
   bash scripts/mapping.sh --workers 8
   python scripts/solve_CG.py
   python optimize_main.py --check
   python optimize_main.py --prepare
   python optimize_main.py --run
   ```

4. To continue a checkpointed fit, use `python optimize_main.py --run --resume`. Transfer the fitted interactions and scaling factor from `results/spo_run/optimized_params.json` to N02, converting units for GROMACS.
