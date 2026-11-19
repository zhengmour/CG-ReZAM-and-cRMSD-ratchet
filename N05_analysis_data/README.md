# N05 · Analyze trajectories

Run from `N05_analysis_data` with NumPy, SciPy, NetworkX, Matplotlib, and tqdm. Install MDAnalysis for GROMACS trajectories and LAMMPS data files.

1. Select one trajectory per invocation. For N03, use the GROMACS run directory containing `init.gro`, `eq2_cg.xtc`, and the subsequent NPT/NVT trajectories. For N04, select a `PIRMSD_<iteration>` segment, its parent, or one run containing `cycle_*/PIRMSD_<iteration>`. Keep each segment's `history.log`, `latest.json`, referenced state, and dump fragments together.
2. Export the N04 bias history and plot both PIRMSDs. `Analysis__pirmsd.csv` reports `cluster_pirmsd_A` in Å and dimensionless `multidomain_pirmsd`. If the run includes `cycle_metrics/`, `Analysis__pirmsd_stages.csv` reports both metrics at the start and after MD, bias, and relaxation.

   ```bash
   python silicate_analysis.py rmsd_lammpstrj 1 '<n04-run-directory>' --metrics-only --output '<metrics-directory>'
   ```

3. Run full analysis for either trajectory format. The N04 command also exports PIRMSD. For a single N04 segment, add `--dt-fs <timestep-in-fs>` to use segment-local time.

   ```bash
   python silicate_analysis.py xtc 8 '<n03-run-directory>' --output '<n03-analysis-directory>'
   python silicate_analysis.py rmsd_lammpstrj 8 '<n04-run-directory>' --output '<n04-analysis-directory>'
   ```

   For LAMMPS, the default types are Si 1, OB 2, C 4, N 5, and Na 6. If the input uses other types, pass `--type-map '<type-map.json>'` with those five keys and integer IDs or lists. For example, the 14-type Si/Al input with Si oxygen sites 2–4 uses `{"Si":1,"OB":[2,3,4],"C":10,"N":11,"Na":12}`.
4. Read the CSVs and figures in the output directory for bonded clusters, Qn, linear/branched/cyclic chains, rings, and reaction counts between adjacent saved frames within each segment. Use `Analysis__frames.csv` to identify the source frames.
