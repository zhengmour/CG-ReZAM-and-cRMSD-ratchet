# CG-ReZAM and PIRMSD sampling

This workflow fits a coarse-grained reactive model, uses it to form a silicate gel, and starts crystal-reference enhanced sampling from that gel. Run each stage from its module directory. Replace values in angle brackets with your paths, executables, or scheduler settings.

| Stage | Run | Output for the next stage |
| --- | --- | --- |
| [N01 · Fit interactions](N01_SPO_Process/README.md) | Map the QM reference structures and fit interaction parameters to energies, forces, and radial distribution functions. | Fitted pair parameters and scaling factor. |
| [N02 · Refine silicate parameters](N02_GA_Process/README.md) | Use GROMACS trials and a genetic algorithm to fit silicate geometry and interactions against structural observables, including Qn. | Selected coefficients and topology parameters for the gel simulation. |
| [N03 · Form the gel](N03_CGMD_Process/README.md) | Pack the mixture, minimize, equilibrate, and run NPT and NVT coarse-grained MD. Check the trajectory and select a gel configuration. | Gel coordinates, periodic box, and topology. Convert these to LAMMPS `lmp.data` for N04. |
| [N04 · Sample from the gel](N04_cRMSD_Process/README.md) | Supply the converted gel and target lattice, then run continuous linear bias or ratchet sampling. | Biased trajectories, committed histories, and checkpoints. |
| [N05 · Analyze trajectories](N05_analysis_data/README.md) | Analyze the N03 gel trajectory or the N04 sampling output. | Bonded-cluster sizes, Qn, chains, rings, reaction counts, and both N04 PIRMSD series when available. |

Transfer fitted parameters between force-field formats with the corresponding unit conversions. The N03-to-N04 handoff also requires a topology-aware coordinate conversion.

## Reading the N04 observables

Both bias modes drive **cluster PIRMSD** (`cluster_pirmsd_A`), the weighted RMS displacement of matched Si sites in eligible spatial clusters. It is measured in Å and can exceed 1 Å. **Multidomain PIRMSD** (`multidomain_pirmsd`) is a dimensionless diagnostic that includes coverage, vacancies, and bounded positional mismatch. N04 records both at the same observation. Compare ratchet minima within the same segment and reference epoch. N05 reports sizes of chemically bonded clusters separately.

## Related publication

*Combining Coarse-Grained Reactive Molecular Dynamics with an Enhanced-Sampling Method for MFI Zeolite Crystallization.* Journal of Chemical Theory and Computation 22, 1409–1425 (2026). [Publication](https://doi.org/10.1021/acs.jctc.5c01745)
