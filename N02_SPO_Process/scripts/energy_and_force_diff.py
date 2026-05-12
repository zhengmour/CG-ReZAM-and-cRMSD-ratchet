import sys
from pathlib import Path
import json
import numpy as np

def read_energy(logfile):
    # Read the last line of energy (or PE) from the LAMMPS log file
    with open(logfile) as f:
        lines = f.readlines()
    energy_lines = [line for line in lines if "Loop time" not in line]
    for line in reversed(energy_lines):
        if line.strip().startswith("Step") or line.strip() == "":
            continue
        try:
            return float(line.strip().split()[-1])
        except:
            continue
    return None

def read_forces(dump_file):
    lines = Path(dump_file).read_text().splitlines()
    data = []
    for line in lines[9:]:
        if line:
            words = line.split()
            if int(words[1]) != 5:
                data.append([float(x) for x in words[5:8]])
    return np.array(data)

def read_refer_forces(dir, name):
    with open(dir / (name + "_force.xyz")) as ff:
        forces = []
        for line in ff.readlines()[2:]:
            if line:
                words = line.split()
                if int(words[0]) != 5:
                    forces.append([float(x) for x in words[1:]])
    return np.array(forces)

if __name__ == '__main__':
    main_dir = Path(sys.argv[1])
    data_dir = sys.argv[2]
    job_dir = Path(sys.argv[3])

    try:
        with open(job_dir.parent / f"cg_zero_energies.json") as f:
            cg_zero_energies = json.load(f)
        data_dir = Path(data_dir)

        for part in data_dir.parts:
            if part not in ('.', '/', '..'):
                elem_pair = part
                break
        cg_zero_energy = cg_zero_energies[elem_pair]

        dataname = data_dir.name
        energy = read_energy(job_dir / (dataname + ".log")) 
        energy -= cg_zero_energy
        
        forces = read_forces(job_dir / (dataname + ".lammpstrj"))
        refer_forces = read_refer_forces(main_dir / data_dir, dataname)
        df = np.sqrt(np.mean(np.linalg.norm(forces-refer_forces, axis=1)**2))

        with open(job_dir / (dataname + '.score'), 'w') as wf:
            wf.write(f'{data_dir} {energy} {df}\n') 
        print(f"{data_dir}: {energy} {df}")
    except Exception as e:
        print(f"energy_and_force_diff.py solving {data_dir} Error: {e}")
        sys.exit(1)


