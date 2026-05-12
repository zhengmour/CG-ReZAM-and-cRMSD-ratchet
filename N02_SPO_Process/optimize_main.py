#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified parameter optimization master script (currently MD-focused; QM Slurm logic
has already been wrapped and can be conveniently integrated later)

- Read config/elements.json:

    {
      "pairs": {
        "Si-Na": [eps, sigma],
        "Si-C":  [eps, sigma],
        "Si-N":  [eps, sigma],
        "Na-O":  [eps, sigma],
        "C-O":   [eps, sigma],
        "N-O":   [eps, sigma]
      },
      "lambda": 0.24
    }

- Expand into flat parameters:
    pairs_Si-Na_0, pairs_Si-Na_1, ..., pairs_Na-O_0, ..., lambda_0

- For each parameter perform 1D scan + quadratic fit:
    For a trial value of some parameter:
      1) Write results/.../params_[md|qm].json
      2) Generate corresponding Slurm submit.sh
      3) Wait for the job to finish
      4) Read md_loss from score.txt
"""

import os
import sys
import json
import time
import logging
import subprocess
from scipy.optimize import minimize
from scipy.stats import pearsonr, zscore
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


class ParamOptimizer:
    def __init__(self, nthreads: int = 64, current_iteration: int = 1,
                 logfile: str = "./optimize.log", job_number: int = 10):
        """
        nthreads: total number of LAMMPS / MPI ranks
        current_iteration: iteration index to start from
        job_number: number of QM Slurm array jobs (reserved for future QM use)
        """

        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.base_dir = Path(__file__).parent.absolute()
        self.nthreads = nthreads
        self.current_iteration = current_iteration
        # Reserved for QM Slurm array usage (to be used later)
        self.job_number = job_number

        # Read config/elements.json
        self.config = self._load_config()
        self.initial_pairs = self.config["pairs"]
        self.lambda_init = float(self.config.get("lambda", 0.24))

        # Expand into 1D flat parameter dictionary
        self.expanded_params = self._expand_parameters()
        self.optimal_params = self.expanded_params.copy()
        self.history = []

        # Optimization control parameters
        self.max_iterations = 200
        # Consider convergence if average relative change < 1%
        self.convergence_thresh = 0.01

        self.step_ratio = 0.2            # Initial step-size ratio
        self.min_step = 0.01             # Minimum absolute step size
        self.step_adjust_factors = {
            "change_threshold": 0.15,
            "increase_factor": 1.05,
            "decrease_factor": 0.95,
            "max_ratio": 2.0,
            "min_ratio": 0.05,
            "global_decrease_factor": 0.97,
        }
        self.param_step_ratios = {p: self.step_ratio for p in self.expanded_params}

        logging.info(f"Working directory: {self.base_dir}")
        logging.info(f"Initial pairs: {self.initial_pairs}")
        logging.info(f"Initial lambda: {self.lambda_init}")
        logging.info(f"Expanded flat parameters: {self.expanded_params}")

        # QM data directory (for later QM integration if needed)
        self.refer_dir = self.base_dir / "data_set" / "refer_CG"
        self.QM_datas, self.aa_zero_energies = self._load_QM_datas()

        # log_sigma_E, log_sigma_F, log_sigma_B, log_zero
        self.weights = [1.0, 0.1, 1.0, 0.1, 1.0, 0.1]

    # ----------------------- Config and parameter expansion -----------------------

    def _load_config(self):
        cfg_path = self.base_dir / "config" / "elements.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "pairs" not in cfg or "lambda" not in cfg:
            raise KeyError("elements.json must contain both 'pairs' and 'lambda' keys")
        if not isinstance(cfg["pairs"], dict):
            raise TypeError("'pairs' must be a dict, e.g., 'Na-O': [eps, sigma]")
        return cfg

    def _expand_parameters(self):
        """
        Expand config["pairs"] and lambda into scalar parameters:
          - 'pairs_Si-Na_0', 'pairs_Si-Na_1', ...
          - 'lambda_0'
        """
        expanded = {}
        for pair_name, values in self.initial_pairs.items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            for i, v in enumerate(values):
                expanded[f"pairs_{pair_name}_{i}"] = float(v)

        expanded["lambda_0"] = self.lambda_init
        return expanded

    def _contract_parameters(self, params_flat):
        """
        Contract flat parameters back to the same structure as elements.json:
            {
              "pairs": {...},
              "lambda": ...
            }
        """
        pairs_out = {}
        for pair_name, init_vals in self.initial_pairs.items():
            if not isinstance(init_vals, (list, tuple)):
                init_vals = [init_vals]
            vals = []
            for i in range(len(init_vals)):
                key = f"pairs_{pair_name}_{i}"
                vals.append(float(params_flat.get(key, init_vals[i])))
            pairs_out[pair_name] = vals

        lambda_out = float(params_flat.get("lambda_0", self.lambda_init))
        return {"pairs": pairs_out, "lambda": lambda_out}

    def _parse_parameter(self, name):
        """
        'pairs_Na-O_0' -> ('pairs', 'Na-O', 0)
        'lambda_0'     -> ('lambda', '', 0)
        """
        if name.startswith("pairs_"):
            _, rest = name.split("pairs_", 1)
            key, idx = rest.rsplit("_", 1)
            return "pairs", key, int(idx)
        elif name.startswith("lambda_"):
            _, idx = name.split("_", 1)
            return "lambda", "", int(idx)
        elif name.startswith("weights"):
            return "weights", "", 0
        else:
            return "other", name, 0

    def _read_energy(self, logfile):
        # Read the last energy (or PE) value from a LAMMPS log file
        with open(logfile) as f:
            lines = f.readlines()
        energy_lines = [line for line in lines if "Loop time" not in line]
        for line in reversed(energy_lines):
            if line.strip().startswith("Step") or line.strip() == "":
                continue
            try:
                return float(line.strip().split()[-1])
            except Exception:
                continue
        return None

    def _load_QM_datas(self):
        energy_file = self.refer_dir / "all_energy_files.txt"                # kcal/A
        # forces_file = self.refer_dir / "all_force_files.txt"                 # kcal/mol,A
        aa_energy_forces_file = self.refer_dir / "aa_energy_forces.json"     # kcal/mol,A
        zero_file = self.refer_dir / "aa_zero_energies.json"                 # kcal/mol

        if aa_energy_forces_file.exists() and zero_file.exists():
            with open(aa_energy_forces_file) as f:
                points = json.load(f)
            with open(zero_file) as f:
                aa_zero_energies = json.load(f)
        else:
            with energy_file.open("r", encoding="utf-8") as f:
                content = f.read()
            energy_blocks = [block.strip() for block in content.strip().split("\n\n") if block.strip()]

            # with forces_file.open("r", encoding="utf-8") as f:
            #     content = f.read()
            # forces_blocks = [block.strip() for block in content.strip().split("\n\n") if block.strip()]

            points = {}
            energies = {}

            for energy in energy_blocks:
                point = {}
                energy_lines = energy.split("\n")

                data_dir = energy_lines[0]
                elem_pair, direction = data_dir.split("/")

                point["energy"] = float(energy_lines[1])
                if elem_pair not in energies.keys():
                    energies[elem_pair] = {}
                energies[elem_pair][data_dir] = point["energy"]

                if data_dir not in points.keys():
                    points[data_dir] = point
                else:
                    points[data_dir].update(point)

            aa_zero_energies = {}
            for elem_pair, infos in energies.items():
                keys = np.array(list(infos.keys()))
                values = np.array(list(infos.values()))

                # Use z-scores to filter out outliers
                z_scores = zscore(values)
                mask = np.abs(z_scores) < 3

                keys_filtered = keys[mask]
                values_filtered = values[mask]
                min_index = np.argmin(values_filtered)

                zero_point = {}
                zero_point["path"] = keys_filtered[min_index]
                zero_point["energy"] = values_filtered[min_index]
                elem_pair, direction = keys_filtered[min_index].split("/")
                aa_zero_energies[elem_pair] = zero_point

            # Subtract zero-point energy for all configurations
            for path, infos in points.items():
                elem_pair, direction = path.split("/")
                points[path]["energy"] -= aa_zero_energies[elem_pair]["energy"]

            with open(aa_energy_forces_file, "w") as wf:
                json.dump(points, wf)

            with open(zero_file, "w") as wf:
                json.dump(aa_zero_energies, wf)

        return points, aa_zero_energies

    # ----------------------- Slurm script generation -----------------------

    def build_params_for_mode(self, mode: str, params_flat: dict) -> dict:
        """
        Construct parameter dictionaries for QM / MD based on the current flat parameters.

        mode = "qm":
            - All pairs directly use values from params_flat (or initial_pairs defaults)
            - Does not include the lambda field

        mode = "md":
            - First recover all pairs as in QM
            - Then multiply epsilon (index 0) of Na-O / C-O / N-O by lambda
            - Does not include the lambda field (lambda is encoded into epsilon)
        """
        # Recover unscaled pairs
        pairs = {}
        excluded_pairs = {"Si-Si"}
        for pair_name, init_vals in self.initial_pairs.items():
            if not isinstance(init_vals, (list, tuple)):
                init_vals = [init_vals]
            vals = []
            for i in range(len(init_vals)):
                key = f"pairs_{pair_name}_{i}"
                vals.append(float(params_flat.get(key, init_vals[i])))

            # For QM, remove selected interactions (e.g., Si-Si) if needed
            if mode == "qm":
                if pair_name not in excluded_pairs:
                    pairs[pair_name] = vals
            else:
                pairs[pair_name] = vals

        if mode == "md":
            # Read lambda
            lambda_val = float(params_flat.get("lambda_0", self.lambda_init))
            # Scale epsilon for O-Na/C/N
            for p in ("Na-O", "C-O", "N-O"):
                if p in pairs and len(pairs[p]) > 0:
                    eps = pairs[p][0]
                    pairs[p][0] = lambda_val * eps

        # Returned dict does not contain lambda, only pairs
        return {"pairs": pairs}

    def _build_md_slurm_script(self, param, value, job_dir: Path, base_key: str):
        """
        Slurm script for MD g(r)-based optimization
        """
        script = f"""#!/bin/bash
#SBATCH --job-name={param}_{value:.2f}
#SBATCH --output={job_dir}/slurm.out
#SBATCH -N 1
#SBATCH -n {self.nthreads}
#SBATCH -c 1

#TODO
LAMMPS=<Path to LAMMPS>
export PATH=$LAMMPS/src:$PATH
export LD_LIBRARY_PATH=$LAMMPS/src:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LAMMPS/src:$LIBRARY_PATH

cd {job_dir}
mpirun -np {self.nthreads} lmp_mpi -in lmp_md.in
python {self.base_dir}/scripts/get_traj_info.py ./ {self.nthreads}
python {self.base_dir}/scripts/gr_similarity.py ./ {self.base_dir} {base_key}
cd {self.base_dir}
"""
        return script

    def _build_qm_slurm_script(self, param, value, job_dir: Path):
        """
        Slurm array script used for QM optimization (optimize_point_energy_main.py)
        """
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        script = f"""#!/bin/bash
#SBATCH --job-name={param}_{value:.2f}
#SBATCH --output={job_dir}/logs/slurm_array_%A_%a.out
#SBATCH --error={job_dir}/logs/slurm_array_%A_%a.err
#SBATCH -N 1
#SBATCH -n {self.nthreads}
#SBATCH -c 1
#SBATCH --array=0-{self.job_number-1}

JOB_TAG="spo-qm"

LIST_ID=$SLURM_ARRAY_TASK_ID
DATA_DIR={self.refer_dir}
SCRIPT_DIR={self.base_dir}/scripts
RESULT_DIR={job_dir}
PART_DIR=part_${{LIST_ID}} 
LIST_FILE=$DATA_DIR/data_dirs_${{LIST_ID}}.list

#TODO
LAMMPS=<Path to LAMMPS>
export PATH=$LAMMPS/src:$PATH
export LD_LIBRARY_PATH=$LAMMPS/src:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LAMMPS/src:$LIBRARY_PATH

monitor_cpu() {{
  echo "Start monitoring CPU usage (PID $$)..."
  while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    cpu=$(ps -p $$ -o %cpu=)
    echo "[$timestamp] Total CPU usage by this script: ${{cpu}}%" 
    sleep 10
  done
}}

cd $RESULT_DIR
mkdir $PART_DIR 
cd $PART_DIR 

# Split the list into 64 parts
split -n l/64 -d "$LIST_FILE" split_${{LIST_ID}}_

run_worker() {{
  id=$(printf "%02d" $1)
  part_list=split_${{LIST_ID}}_${{id}}

  if [[ ! -f $part_list ]]; then
    echo "ERROR: File $part_list not found!"
    return
  fi

  while read data_dir || [[ -n "$data_dir" ]]; do
    filename=$(basename "$data_dir")
    full_datafile="$DATA_DIR/$data_dir/$filename"

    echo "[$id] Processing: $filename"
    lmp_serial -in ../lmp_qm.in -var datafile $full_datafile -var name $filename > ${{filename}}.log 
    python $SCRIPT_DIR/energy_and_force_diff.py $DATA_DIR $data_dir $RESULT_DIR/$PART_DIR 
    rm ${{filename}}.log ${{filename}}.lammpstrj
  done < "$part_list"

  echo "[worker $id] Done."
}}

# monitor_cpu &  # start monitoring
# MONITOR_PID=$!

# Launch multiple workers in parallel
for ((i=0; i<{self.job_number}; i++)); do
  run_worker $i &
done

wait
# kill $MONITOR_PID

# Combine errors
cat *Na*.score > ../Na_score_${{LIST_ID}}.txt
cat *TPA*.score > ../TPA_score_${{LIST_ID}}.txt
cat *Si*.score > ../SIV_score_${{LIST_ID}}.txt

# Remove temporary files
cd $RESULT_DIR
rm -rf $PART_DIR 
"""
        return script

    def _get_elem_loss_values(self, job_dir, molname):
        energies, refer_energies, dfs = [], [], []

        # *.score must end with an empty line
        score_files = list(job_dir.glob(molname))

        for score_file in score_files:
            with open(score_file) as f:
                for line in f:
                    data_dir, energy, df = line.strip().split()
                    refer_energies.append(self.QM_datas[data_dir]["energy"])
                    energies.append(float(energy))
                    delta_E = self.QM_datas[data_dir]["energy"] - float(energy)
                    if abs(delta_E) > 500:
                        logging.error(
                            f"Energy difference for {data_dir} is very large, ΔE = {delta_E}"
                        )
                    dfs.append(float(df))

        energies = np.array(energies)
        refer_energies = np.array(refer_energies)
        des = energies - refer_energies
        dfs = np.array(dfs)

        energy_loss = np.sqrt(np.mean(des**2))
        forces_loss = np.sqrt(np.mean(dfs**2))

        return [energy_loss, forces_loss]

    def _collect_qm_loss(self, job_dir):
        patterns = ["Na_score_*.txt", "TPA_score_*.txt", "SIV_score_*.txt"]
        all_losses = []
        for pattern in patterns:
            losses = self._get_elem_loss_values(job_dir, pattern)
            all_losses.extend(losses)

        cg_zero_energies_file = job_dir / "cg_zero_energies.json"
        with open(cg_zero_energies_file) as f:
            cg_zero_energies = json.load(f)
        keys = sorted(set(self.aa_zero_energies.keys()) & set(cg_zero_energies.keys()))
        pred_vals = np.array([self.aa_zero_energies[k]["energy"] for k in keys])
        true_vals = np.array([cg_zero_energies[k] for k in keys])

        min_index = np.argmin(true_vals)
        pred_vals -= pred_vals[min_index]
        true_vals -= true_vals[min_index]
        pearson_corr, _ = pearsonr(true_vals, pred_vals)  # [-1.0, 1.0]
        zero_loss = 1.0 - pearson_corr

        logging.debug(
            f"[job_dir] E&F losses are {all_losses} and zero loss is {zero_loss}"
        )
        return all_losses, zero_loss

    # ----------------------- Slurm submission and waiting -----------------------

    def _wait_for_job(self, job_ids):
        """
        Wait for one or multiple Slurm jobs to finish.
        Args:
            job_ids: str or list[str]
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]

        # Convert to comma-separated string for squeue to query multiple jobs at once
        job_ids_str = ",".join(job_ids)

        logging.debug(f"Waiting for jobs to finish: {job_ids_str}")
        while True:
            result = subprocess.run(
                ["squeue", "-h", "-j", job_ids_str],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logging.warning(f"squeue query failed: {result.stderr}")
                break

            active_jobs = [
                line.strip().split()[0]
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if not active_jobs:
                logging.debug(f"All jobs have finished: {job_ids_str}")
                break

            logging.debug(f"Jobs still running: {', '.join(active_jobs)}")
            time.sleep(60)

    # ----------------------- Single evaluation: current version = QM+MD -----------------------

    def _run_single_simulation(self, param, value):
        """
        Assign a value to param, generate params.json for QM and MD respectively,
        then submit QM+MD Slurm jobs, and return total_loss = qm_loss + md_loss.

        - QM uses "raw parameters": epsilon is not scaled by lambda
        - MD uses "scaled parameters": epsilon of Na-O / C-O / N-O is multiplied by lambda
        """

        ptype, base_key, idx = self._parse_parameter(param)

        if ptype == "pairs":
            element = base_key.split("-")[0]
            job_dir = (
                self.base_dir
                / "results"
                / f"iteration_{self.current_iteration}"
                / "pairs"
                / element
                / base_key
                / str(idx)
                / f"{value:.6f}"
            )
        elif ptype == "lambda":
            job_dir = (
                self.base_dir
                / "results"
                / f"iteration_{self.current_iteration}"
                / "lambda"
                / f"{value:.6f}"
            )
        elif ptype == "weights":
            job_dir = (
                self.base_dir
                / "results"
                / f"iteration_{self.current_iteration}"
                / "weights"
            )
        else:
            job_dir = (
                self.base_dir
                / "results"
                / f"iteration_{self.current_iteration}"
                / "other"
                / param
                / f"{value:.6f}"
            )

        job_dir.mkdir(parents=True, exist_ok=True)

        # 1) Current flat parameters
        trial_params = self.optimal_params.copy()
        if value is not None:
            trial_params[param] = float(value)
        else:
            # Placeholder for later use
            value = -1

        # 2) Build parameter dictionaries for QM / MD
        params_qm = self.build_params_for_mode("qm", trial_params)  # no lambda, unscaled
        params_md = self.build_params_for_mode("md", trial_params)  # no lambda, scaled

        # Write to different files
        params_qm_json = job_dir / "params_qm.json"
        params_md_json = job_dir / "params_md.json"
        with params_qm_json.open("w", encoding="utf-8") as f:
            json.dump(params_qm, f, indent=2)
        with params_md_json.open("w", encoding="utf-8") as f:
            json.dump(params_md, f, indent=2)

        # 3) Copy module_files into job_dir (shared by QM + MD)
        module_dir = self.base_dir / "jobs" / "module_files"
        if module_dir.exists():
            for src in module_dir.iterdir():
                dst = job_dir / src.name
                if src.is_file():
                    dst.write_bytes(src.read_bytes())
        else:
            logging.warning(f"module_files directory not found: {module_dir}")

        # Create lmp_md.in and lmp_qm.in
        cmd = f"cp {self.base_dir}/jobs/module_files/* {job_dir}"
        subprocess.run(cmd, shell=True, check=True)
        cmd = f"python {self.base_dir}/scripts/params_to_in.py {job_dir}/lmp_md.in {job_dir}/lmp_md.in {job_dir}/params_md.json"
        subprocess.run(cmd, shell=True, check=True)
        cmd = f"python {self.base_dir}/scripts/params_to_in.py {job_dir}/lmp_qm.in {job_dir}/lmp_qm.in {job_dir}/params_qm.json"
        subprocess.run(cmd, shell=True, check=True)

        # ----------------- QM part -----------------
        if ptype != "lambda" and base_key != "Si-Si":
            qm_loss = 1.0e6
            job_id_qm = None
            try:
                # Compute zero-point energies for molecular pairs
                zero_paths = [info["path"] for info in self.aa_zero_energies.values()]
                for path in zero_paths:
                    name, direction = path.split("/")
                    datafile = f"{self.refer_dir}/{path}/{direction}"
                    logfile = f"{job_dir}/{name}.log"
                    cmd = [
                        "lmp_mpi",
                        "-in",
                        f"{job_dir}/lmp_qm.in",
                        "-var",
                        "datafile",
                        datafile,
                        "-var",
                        "name",
                        f"{job_dir}/{name}",
                        "-log",
                        logfile,
                    ]
                    subprocess.run(cmd)

                log_files = list(job_dir.glob("*.log"))
                cg_zero_energies = {}
                for log_file in log_files:
                    molecule = log_file.stem
                    energy = self._read_energy(log_file)
                    cg_zero_energies[molecule] = energy

                with open(job_dir / "cg_zero_energies.json", "w") as wf:
                    json.dump(cg_zero_energies, wf)

                # Submit QM jobs (submission only; waiting is done later)
                qm_slurm_path = job_dir / "submit_qm.sh"
                qm_slurm = self._build_qm_slurm_script(param, value, job_dir)
                with qm_slurm_path.open("w", encoding="utf-8") as f:
                    f.write(qm_slurm)

                result_qm = subprocess.run(
                    ["sbatch", str(qm_slurm_path)],
                    capture_output=True,
                    text=True,
                )
                if result_qm.returncode != 0:
                    raise RuntimeError(f"QM job submission failed: {result_qm.stderr}")

                job_id_qm = result_qm.stdout.strip().split()[-1]
                logging.debug(
                    f"Submitted QM job: {param}={value:.4f}, JobID={job_id_qm}"
                )

            except Exception as e:
                logging.error(
                    f"QM part encountered an error; qm_loss set to 1.0e6: {e}"
                )
                qm_loss = 1.0e6
        else:
            qm_loss = 0.0
            job_id_qm = None

        # ----------------- MD part -----------------
        if ptype != "weights":
            md_loss = 1.0e6
            job_id_md = None
            try:
                md_slurm_path = job_dir / "submit_md.sh"
                md_slurm = self._build_md_slurm_script(param, value, job_dir, base_key)
                with md_slurm_path.open("w", encoding="utf-8") as f:
                    f.write(md_slurm)

                result_md = subprocess.run(
                    ["sbatch", str(md_slurm_path)],
                    capture_output=True,
                    text=True,
                )
                if result_md.returncode != 0:
                    raise RuntimeError(f"MD job submission failed: {result_md.stderr}")

                job_id_md = result_md.stdout.strip().split()[-1]
                logging.debug(
                    f"Submitted MD job: {param}={value:.4f}, JobID={job_id_md}"
                )

            except Exception as e:
                logging.error(
                    f"MD submission failed; md_loss set to 1.0e6: {e}"
                )
                md_loss = 1.0e6
        else:
            md_loss = 0.0
            job_id_md = None

        # ----------------- Wait for both -----------------
        job_ids = [j for j in (job_id_qm, job_id_md) if j is not None]
        if job_ids:
            self._wait_for_job(job_ids)

        # ----------------- Unified result processing -----------------
        # Collect QM loss
        losses = [1e6] * len(self.weights)
        if job_id_qm is not None:
            try:
                losses, zero_loss = self._collect_qm_loss(job_dir)
                qm_loss = self._auto_weighted_loss(losses) + zero_loss
            except Exception as e:
                logging.error(f"Collecting QM loss failed; qm_loss set to 1.0e6: {e}")
                qm_loss = 1.0e6

        # Read MD loss from score.txt (assume second column of last line is the score)
        if job_id_md is not None:
            score_file = job_dir / "score.txt"
            if not score_file.exists():
                logging.error(
                    f"{score_file} does not exist; MD loss set to 1.0e6"
                )
                md_loss = 1.0e6
            else:
                try:
                    with score_file.open("r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    last = lines[-1].split()
                    md_loss = float(last[1])
                except Exception as e:
                    logging.error(f"Failed to parse MD score.txt: {e}")
                    md_loss = 1.0e6

        total_loss = qm_loss + md_loss
        logging.info(f"[{param}={value:.4f}] QM-losses(raw)={losses}")
        logging.info(
            f"[{param}={value:.4f}] QM={qm_loss:.3f}  MD={md_loss:.3f}  total={total_loss:.3f}"
        )

        return losses, md_loss, total_loss

    # ----------------------- 1D scan + quadratic fit -----------------------

    def _get_param_min(self, param_name):
        # Index 0 is treated as epsilon with lower bound 0.05
        # Index 1 is treated as sigma with lower bound 1.0
        if param_name.startswith("pairs_"):
            if param_name.endswith("_0"):
                return 0.05
            elif param_name.endswith("_1"):
                return 1.0
        return 0.0

    def _generate_param_range(self, param):
        """
        Generate a set of candidate test points such that:
        - All candidates >= max(0.0, param_min)
        - If the original interval crosses 0, include 0.0 as a candidate
          (as long as that parameter allows 0)
        - If current value is too close to the lower bound to obtain 3 points,
          shift the interval to the right
        """
        current = float(self.optimal_params[param])

        # Lower bound: no negative values and not below the parameter's own minimum
        param_min = float(self._get_param_min(param))
        minb = max(0.0, param_min)

        # Step size
        if abs(current) < max(param_min, 1e-12):
            abs_step = max(self.min_step, 0.2)
        else:
            step_ratio = self.param_step_ratios[param]
            abs_step = max(abs(current) * step_ratio, self.min_step)

        # Original 3 points (before clipping, used to check if interval crosses 0)
        raw_left = current - abs_step
        raw_mid = current
        raw_right = current + abs_step

        # If left point hits lower bound, shift the whole interval to the right
        if raw_left <= minb + 1e-12:
            candidates = [minb, minb + abs_step, minb + 2 * abs_step]
        else:
            candidates = [raw_left, raw_mid, raw_right]

        # If interval crosses 0 and 0 is allowed, add 0.0
        crosses_zero = raw_left <= 0.0 <= raw_right
        if crosses_zero and minb <= 0.0:
            candidates.append(0.0)

        # Clip to lower bound, remove duplicates, sort
        candidates = sorted(set(round(max(v, minb), 6) for v in candidates))

        # If fewer than 3 candidates, extend to the right
        while len(candidates) < 3:
            if not candidates:
                base = max(minb, 0.0)
                candidates = [base]
            candidates.append(round(candidates[-1] + abs_step, 6))

        # Take the three smallest candidates
        candidates = candidates[:3]

        return current, candidates

    def _find_optimal(self, values, scores):
        """
        Quadratic fitting + vertex search;
        if the fit fails, return the best value among the discrete points.
        """
        values = np.array(values, dtype=float)
        scores = np.array(scores, dtype=float)

        if len(values) < 3:
            return float(values[np.argmin(scores)])

        try:
            mean_val = values.mean()
            std_val = values.std() if values.std() > 1e-8 else 1.0
            x_scaled = (values - mean_val) / std_val

            coeffs = np.polyfit(x_scaled, scores, 2)
            a, b, c = coeffs
            if a <= 0:
                return float(values[np.argmin(scores)])

            vertex_scaled = -b / (2 * a)
            vertex = vertex_scaled * std_val + mean_val

            min_bound = float(values.min() * 0.8)
            max_bound = float(values.max() * 1.2)
            vertex = float(np.clip(vertex, min_bound, max_bound))
            return vertex
        except Exception as e:
            logging.warning(
                f"Quadratic fit failed; falling back to best discrete value: {e}"
            )
            return float(values[np.argmin(scores)])

    def _collect_rdf_first_sigmas(self, param, test_values):
        """
        Only for LJ sigma parameters (pairs_XXX_1):
        For each test_value, read rdf_first_sigma_file.json in the corresponding
        directory and extract the sigma inferred from the first peak of the RDF.
        """
        ptype, base_key, idx = self._parse_parameter(param)

        # Only active for pairs_X-Y_1; for other parameters, return an empty array
        if not (ptype == "pairs" and idx == 1):
            return np.array([], dtype=float)

        element = base_key.split("-")[0]  # e.g. "Na-O" -> "Na"
        rdf_first_sigmas = []

        for v in test_values:
            job_dir = (
                self.base_dir
                / "results"
                / f"iteration_{self.current_iteration}"
                / "pairs"
                / element
                / base_key
                / str(idx)
                / f"{v:.6f}"
            )
            rdf_first_sigma_file = job_dir / "rdf_first_sigma_file.json"

            try:
                with rdf_first_sigma_file.open("r", encoding="utf-8") as f:
                    rdf_config = json.load(f)

                # In gr_similarity.py the key is written as {"Na-O_1": value, ...}
                key = f"{base_key}_{idx}"  # e.g. "Na-O_1"
                val = rdf_config.get(key, np.nan)

                try:
                    if not np.isnan(val):
                        rdf_first_sigmas.append(val)
                    else:
                        logging.info(
                            f"For {param}, key={key} in {rdf_first_sigma_file} is NaN, skip it."
                        )
                except TypeError:
                    # val is not a numeric type; skip it
                    logging.info(
                        f"For {param}, key={key} in {rdf_first_sigma_file} is non-numeric, skip it."
                    )
            except Exception as e:
                logging.info(
                    f"Warning: could not read rdf_first_sigma_file {rdf_first_sigma_file}: {e}"
                )

        return np.array(rdf_first_sigmas, dtype=np.float64)

    def _update_step_ratio(self, param, old_value, new_value):
        """
        Adaptively update the step-size ratio based on the magnitude of the
        parameter change in the current iteration.
        """
        current_ratio = self.param_step_ratios.get(param, self.step_ratio)
        if old_value == 0:
            delta_rel = 0.0
        else:
            delta_rel = abs(new_value - old_value) / abs(old_value)

        if delta_rel > self.step_adjust_factors["change_threshold"]:
            new_ratio = min(
                current_ratio * self.step_adjust_factors["increase_factor"],
                self.step_adjust_factors["max_ratio"],
            )
        else:
            new_ratio = max(
                current_ratio * self.step_adjust_factors["decrease_factor"],
                self.step_adjust_factors["min_ratio"],
            )

        logging.debug(f"Step update for {param}: {current_ratio:.3f}->{new_ratio:.3f}")
        return new_ratio

    def optimize_parameter(self, param):
        """
        Perform 1D scan + quadratic fit for a single parameter and return the
        optimal value for that parameter.
        """
        old_value = float(self.optimal_params[param])
        current, test_values = self._generate_param_range(param)

        # --- Evaluate each test_value in parallel ---
        def eval_value(v):
            _, _, score = self._run_single_simulation(param, v)
            return v, score

        scores_dict = {}
        # Note: use a thread pool here to avoid nesting process pools
        max_workers = len(test_values)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_v = {executor.submit(eval_value, v): v for v in test_values}
            for fut in as_completed(future_to_v):
                v, score = fut.result()
                scores_dict[v] = score

        scores = [scores_dict[v] for v in test_values]
        optimal_val = self._find_optimal(test_values, scores)
        logging.info(
            f"[{param}] -> {optimal_val:.4f} (losses={scores}, avg loss={np.mean(scores):.3f})"
        )

        # ========= Merge sigma estimated from the first RDF peak =========
        rdf_first_sigmas = self._collect_rdf_first_sigmas(param, test_values)
        if rdf_first_sigmas.size > 0:
            rdf_avg = np.nanmean(rdf_first_sigmas)
            if not np.isnan(rdf_avg) and abs(rdf_avg - optimal_val) > 1.0:
                logging.info(
                    f"{param} adjusted to RDF first-peak sigma: {optimal_val:.6f} -> {rdf_avg:.6f}"
                )
                optimal_val = rdf_avg
        # =================================================================

        new_step_ratio = self._update_step_ratio(param, old_value, optimal_val)

        return {param: optimal_val}, {param: new_step_ratio}

    # ----------------------- Weight and reference QM generation -----------------------

    def split_into_n_files(self, input_dir: Path, n_files: int = 50):
        """
        Split Na_data_dirs.list, TPA_data_dirs.list, SIV_data_dirs.list into
        n_files chunks respectively, and for each output file merge the three
        categories into one.
        """
        # Paths of the three category data files
        categories = ["Na", "TPA", "SIV"]

        # Read all lines for each category
        data = {}
        for cat in categories:
            list_file = input_dir / f"{cat}_data_dirs.list"
            if not list_file.exists():
                raise FileNotFoundError(f"{list_file} does not exist")
            lines = list_file.read_text().splitlines()
            data[cat] = lines
            logging.debug(f"{cat}: total {len(lines)} lines")

        # Compute chunk size for each category
        chunk_sizes = {
            cat: (len(lines) + n_files - 1) // n_files for cat, lines in data.items()
        }

        # Create n_files output files
        for i in range(n_files):
            merged_chunk = []
            for cat in categories:
                start = i * chunk_sizes[cat]
                end = min(start + chunk_sizes[cat], len(data[cat]))
                chunk = data[cat][start:end]
                merged_chunk.extend(chunk)
            output_file = input_dir / f"data_dirs_{i}.list"
            Path(output_file).write_text("\n".join(merged_chunk) + "\n")
            logging.debug(
                f"Wrote file: {output_file}; per-category chunk size {chunk_sizes}, "
                f"actual merged lines {len(merged_chunk)}"
            )

    def sampling_data(self, sample_size=40000):
        """
        Randomly sample data directories from all_data_dirs.list to data_dirs.list.
        If sample_size <= 0 or exceeds the available data size, copy all entries.
        """
        input_file = self.refer_dir / "all_data_dirs.list"
        output_file = self.refer_dir / "data_dirs.list"
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} does not exist")

        lines = [line for line in input_file.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"{input_file} is empty")

        if sample_size > 0 and sample_size < len(lines):
            rng = np.random.default_rng()
            selected_indices = np.sort(rng.choice(len(lines), size=sample_size, replace=False))
            sampled_lines = [lines[i] for i in selected_indices]
            logging.debug(
                f"Randomly sampled file generated: {output_file} (n={sample_size})"
            )
        else:
            sampled_lines = lines
            logging.debug(
                f"Copied all data directories to {output_file} (n={len(sampled_lines)})"
            )

        output_file.write_text("\n".join(sampled_lines) + "\n")

    # -------------------- Automatic weight update --------------------------

    def _auto_weighted_loss(self, losses):
        return float(np.sum(np.array(self.weights) * np.array(losses))) * len(
            self.weights
        )

    def _minimize_weighted(self, log_sigmas, Losses):
        total_loss = 0.0
        for log_sigma, Loss in zip(log_sigmas, Losses):
            total_loss += Loss / (2.0 * np.exp(2.0 * log_sigma)) + log_sigma
        return total_loss

    def update_weights(self):
        """
        Update the loss weights using a one-shot QM evaluation and
        the log-sigma formulation from heteroscedastic regression.
        """
        # Submit jobs and wait for completion
        qm_losses, _, _ = self._run_single_simulation("weights", None)

        log_sigmas = [-0.5 * np.log(2.0 * weight) for weight in self.weights]
        res = minimize(self._minimize_weighted, x0=log_sigmas, args=(qm_losses,))
        self.weights = [1.0 / (2.0 * np.exp(2.0 * log_sigma)) for log_sigma in res.x]

        logging.info(
            f"[Step {self.current_iteration}] losses = {qm_losses},\n weights = {self.weights}"
        )

    # ----------------------- Convergence check and main loop -----------------------

    def check_convergence(self):
        """
        Simple convergence criterion:
        average relative change of all parameters between the last two
        iterations < convergence_thresh.
        """
        if len(self.history) < 2:
            return False
        prev = self.history[-2]
        curr = self.history[-1]
        deltas = [
            abs(curr[k] - prev[k]) / (abs(prev[k]) + 1e-3) for k in curr.keys()
        ]
        mean_delta = float(np.mean(deltas))
        logging.debug(
            f"Average relative parameter change this iteration: {mean_delta:.4f}"
        )
        return mean_delta < self.convergence_thresh

    def main_optimization(self):
        """
        Main optimization loop: sequentially optimize all parameters until
        convergence or until reaching the maximum number of iterations.
        """
        while self.current_iteration <= self.max_iterations:
            logging.info(
                f"\n=== Iter {self.current_iteration}/{self.max_iterations} ==="
            )

            # Sample N data points (for initial weight update pass)
            self.sampling_data(-1)
            self.split_into_n_files(input_dir=self.refer_dir, n_files=self.job_number)

            # Update loss function weights
            self.update_weights()
            logging.info(f"[Iter {self.current_iteration}] Weights updated.")

            # Sample N data points (for parameter optimization pass)
            self.sampling_data()
            self.split_into_n_files(input_dir=self.refer_dir, n_files=self.job_number)

            new_params = {}
            new_step_ratios = {}

            # Number of workers can be tuned based on number of parameters and cluster load
            max_workers = len(self.expanded_params)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit optimization tasks for all parameters
                _futures = {
                    executor.submit(self.optimize_parameter, param): param
                    for param in self.expanded_params.keys()
                }

                # Wait and collect results
                for fut in as_completed(_futures):
                    p = _futures[fut]
                    try:
                        param_update, ratio_update = fut.result()
                        new_params.update(param_update)
                        new_step_ratios.update(ratio_update)
                    except Exception as e:
                        logging.error(f"Parameter {p} optimization failed: {e}")

            # Update global optimal parameters and step-size ratios
            for k, v in new_params.items():
                self.optimal_params[k] = float(v)
            for k, v in new_step_ratios.items():
                self.param_step_ratios[k] = float(v)

            # Record history for convergence check
            self.history.append(self.optimal_params.copy())

            # Convergence check
            if self.check_convergence():
                logging.info("Convergence criterion met; stopping optimization early.")
                break

            # Apply a slight global decay to step-size ratios
            for k in self.param_step_ratios:
                self.param_step_ratios[k] *= self.step_adjust_factors[
                    "global_decrease_factor"
                ]

            self.current_iteration += 1

        # Output final results
        logging.info("\n=== Optimization finished ===")
        for param, value in self.optimal_params.items():
            logging.info(f"{param}: {value:.6f}")

        final_params = self._contract_parameters(self.optimal_params)
        out_path = self.base_dir / "optimized_params.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(final_params, f, indent=2)
        logging.info(f"Final parameters have been written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        nthreads = int(sys.argv[1])
        current_iteration = int(sys.argv[2])
    else:
        nthreads = 64
        current_iteration = 1

    optimizer = ParamOptimizer(
        nthreads=nthreads, current_iteration=current_iteration, job_number=10
    )
    optimizer.main_optimization()
