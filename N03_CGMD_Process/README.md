# N03 · Prepare the gel by coarse-grained MD

Run from `N03_CGMD_Process` with Packmol and MPI-enabled GROMACS on your path.

1. Apply the fitted N01/N02 parameters to the GROMACS inputs. Check that the species, order, counts, and box in `packmol_water.inp` match `topol_water.top` and `sbatch.sh`.
2. Set the run lengths and thermodynamic conditions in the MDP files. Set the Slurm resources in `sbatch.sh` and submit the run.

   ```bash
   sbatch sbatch.sh
   ```

3. Follow minimization (`em_cg`, `em2_cg`), equilibration (`eq_cg`, `eq2_cg`), NPT, and NVT. Inspect the resulting trajectory for gel formation. N05 can analyze its bonded clusters and Qn distribution.
4. Select a gel configuration and convert it with its topology to a LAMMPS `atom_style full` data file for N04. Preserve molecule IDs, atom types, charges, bonds, angles, and periodic box dimensions, and convert GROMACS lengths from nm to Å.
