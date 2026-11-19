# N04 · Run PIRMSD sampling from the gel

1. Use a working copy of this directory for each trajectory. Place the converted N03 gel in `lmp.data`. Check the atom types and force-field coefficients in the LAMMPS inputs against the gel topology.
2. Activate the Python environment containing MPI-enabled LAMMPS, mpi4py, and pybind11. Build the extension with the same MPI installation as LAMMPS.

   ```bash
   EIGEN_INCLUDE_DIR='<path-to-eigen3>' MPI_HOME='<path-to-mpi>' python setup.py build_ext --inplace --force
   ```

3. In `params.json`, set the temperature, timestep, reference XYZ, PLUMED executable, bias strength, and segment length. For continuous linear bias use `bias_algorithm="continuous"`, `ratchet_enabled=false`, and `bias_potential="linear"`. For ratchet sampling use `bias_algorithm="ratchet"` and `ratchet_enabled=true`.
4. To run a single bias segment from the prepared gel, set `init_iter` to 0 and run the following commands inside an MPI allocation. To resume the same segment, use the same settings, rank count, and checkpoint directory.

   ```bash
   ranks='<mpi-ranks>'
   python -c 'import json; print(json.dumps(json.load(open("params.json"))["Enhance_Sampling"]["PIRMSD"]))' > pirmsd_0.json
   mpirun -np "$ranks" python PIRMSD.py pirmsd_0.json
   mpirun -np "$ranks" lmp_mpi -in lmp_0.relax.in
   ```

5. For repeated bias, relaxation, and CGMD cycles, configure the software paths and resources in `sbatch.sh` and run `sbatch sbatch.sh` from the prepared working directory. Keep the `PIRMSD_<iteration>/` directories with their history, trajectories, and checkpoints for N05.
