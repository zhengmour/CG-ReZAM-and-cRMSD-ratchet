import sys
import re
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree

SIV_DQ = "related to RSi model"
SIV_VQ = "related to RSi model"

# OH OB ON => SP
atom_types = {
    "Si" : 1,
    "OH" : 2,
    "OB" : 2,
    "ON" : 2,
    "V"  : 3,
    "C"  : 4,
    "N"  : 5,
    "Na" : 6,
    "Br" : 7,
    "W"  : 8
}

elem_masses = {
    "Si" : 28.084,
    "O"  : 15.999,
    "H"  : 1.008,
    "C"  : 12.0096,
    "N"  : 14.00643,
    "Na" : 22.98977 
}

bond_types = {
    "Si-D": 1,
    "D-V" : 2,
    "D-D" : 3,
    "N-C" : 4,
    "B-B" : 5
}

angle_types = {
    "SIV" : 1,
    "TPA" : 2
}

def parse_gaussian_output(log_file):
    # Mapping of atomic types to element symbols
    atom_type_map = {
         1: 'H',   2: 'He',  6: 'C',   7: 'N',  8: 'O', 
         9: 'F',  11: 'Na', 14: 'Si', 15: 'P', 16: 'S', 
        17: 'Cl', 35: 'Br'
    }

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract the SCF iteration energy
    scf_match = re.search(r'SCF Done:\s+E\(.*?\)\s+=\s+(-?\d+\.\d+)', content)
    energy = float(scf_match.group(1))


    # Extract the final structure coordinates
    coord_pattern = re.compile(
        r"Standard orientation:.*?(\n\s*-+)+"    # Match the title and any number of dividers
        r"\s*Center.*?Type\s+X\s+Y\s+Z\s*\n"     # Match the column header rows
        r"(\s*-+.*?\n)"                          # Match the divider
        r"(.*?)"                                 # Match the data core area
        r"(\n\s*-+\n)",                          # End the divider
        re.DOTALL | re.IGNORECASE
    )
    coord_match = coord_pattern.search(content)
    data_block = coord_match.group(3)
    coordinates = []
    for line in data_block.strip().split('\n'):
        cols = line.split()
        if len(cols) == 6:
            atom_type = int(cols[1])
            x, y, z = map(float, cols[3:6])
            element = atom_type_map.get(atom_type, 'X')
            coordinates.append((element, x, y, z))

    # Get forces of atoms
    force_pattern = r"^\s*\d+\s+\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s*$"
    force_match = re.findall(force_pattern, content, re.MULTILINE)
    forces = []
    for fx, fy, fz in force_match:
        forces.append((float(fx), float(fy), float(fz)))

    # Combine data
    results = []
    for (elem, x, y, z), (fx, fy, fz) in zip(coordinates, forces):
        results.append({
            'Element': elem,
            'X': float(x), 'Y': float(y), 'Z': float(z),
            'Fx': float(fx), 'Fy': float(fy), 'Fz': float(fz)
        })

    return results, energy

def mapping_rules(atoms_info):
    cg_atoms, cg_bonds, cg_angles = [], [], []
    cg_forces = []

    silicas, oxygens, hydrogens = [], [], []
    nitrogens, carbons = [], []
    sodiums = []

    silica_positions, oxygen_positions, hydrogen_positions = [], [], []
    nitrogen_positions, carbon_positions = [], []
    sodium_positions = []

    silica_forces, oxygen_forces, hydrogen_forces = [], [], []
    nitrogen_forces, carbon_forces = [], []
    sodium_forces = []    

    for atom in atoms_info:
        if atom['Element'] == 'Si':
            silicas.append(atom)
            silica_positions.append([atom['X'], atom['Y'], atom['Z']])
            silica_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])
        elif atom['Element'] == 'O':
            oxygens.append(atom)
            oxygen_positions.append([atom['X'], atom['Y'], atom['Z']])
            oxygen_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])
        elif atom['Element'] == 'N':
            nitrogens.append(atom)
            nitrogen_positions.append([atom['X'], atom['Y'], atom['Z']])
            nitrogen_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])
        elif atom['Element'] == 'C':
            carbons.append(atom)
            carbon_positions.append([atom['X'], atom['Y'], atom['Z']])
            carbon_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])
        elif atom['Element'] == 'Na':
            sodiums.append(atom)
            sodium_positions.append([atom['X'], atom['Y'], atom['Z']])
            sodium_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])
        elif atom['Element'] == 'H':
            hydrogens.append(atom)
            hydrogen_positions.append([atom['X'], atom['Y'], atom['Z']]) 
            hydrogen_forces.append([atom['Fx'], atom['Fy'], atom['Fz']])   
    
    molid = 1
    silica_positions = np.array(silica_positions)
    nitrogen_positions = np.array(nitrogen_positions)
    carbon_positions = np.array(carbon_positions)
    oxygen_positions = np.array(oxygen_positions)
    hydrogen_positions = np.array(hydrogen_positions)
    sodium_positions = np.array(sodium_positions)

    silica_forces = np.array(silica_forces)
    oxygen_forces = np.array(oxygen_forces)
    nitrogen_forces = np.array(nitrogen_forces)
    carbon_forces = np.array(carbon_forces)    
    hydrogen_forces = np.array(hydrogen_forces)
    sodium_forces = np.array(sodium_forces)

    # Construct an atomic KD tree
    if silica_positions.size != 0:
        si_tree = KDTree(silica_positions)
    if oxygen_positions.size != 0:
        o_tree = KDTree(oxygen_positions)
    if hydrogen_positions.size != 0:
        h_tree = KDTree(hydrogen_positions)
    if carbon_positions.size != 0:
        c_tree = KDTree(carbon_positions)

    # Find all the O's near the silicon, find all the H connected to the O
    AB_indices = []
    AB_positions = []
    for si_idx, si_pos in enumerate(silica_positions):
        # Find the 4 nearest O atoms
        o_dists, o_indices = o_tree.query(si_pos, k=4)
        
        # Verification Key Length (Tolerance±0.1Å)
        valid_o = [o_idx for o_idx, d in zip(o_indices, o_dists) if 1.5 <= d <= 1.8]
        
        cluster = {"Si": si_idx, "O": [], "H": []}
        for o_idx in valid_o:            
            # Find O-linked H atoms (tolerance±0.1Å)
            h_dists, h_indices = h_tree.query(oxygen_positions[o_idx], k=2)  # 最多找1个H
            connected_h = [
                h_idx for h_idx, h_d in zip(h_indices, h_dists) if 0.8 <= h_d <= 1.2
            ]
            
            cluster["O"].append(o_idx)
            if connected_h:
                cluster["H"].extend(connected_h)
            else:
                cluster["H"].append(None)

        # Map the whole atom to SIV according to the aboveom to SIV according to the above
        idx = len(cg_atoms) + 1
        cg_atoms.append(('Si', molid, silica_positions[si_idx]))
        cg_forces.append(silica_forces[si_idx])

        idx_copy = idx + 1
        for o_idx, h_idx in zip(cluster['O'], cluster['H']):
            v_Si_O = oxygen_positions[o_idx] - si_pos
            norm_V_Si_O = v_Si_O / np.linalg.norm(v_Si_O)
            rotate_ratio = SIV_DQ / np.linalg.norm(v_Si_O)

            if h_idx is not None:
                cg_atoms.append(('OH', molid, si_pos + norm_V_Si_O * SIV_DQ))
                force = (oxygen_forces[o_idx]*elem_masses['O'] + hydrogen_forces[h_idx]*elem_masses['H']) / (elem_masses['O'] + elem_masses['H'])
                force *= rotate_ratio
                cg_forces.append(force)
            else:
                si_dists, si_indices = si_tree.query(oxygen_positions[o_idx], k=2)
                connected_si = [si_idx for si_idx, d in zip(si_indices, si_dists) if 1.5 <= d <= 1.8]
                if len(connected_si) == 2:
                    cg_atoms.append(('OB', molid, si_pos + norm_V_Si_O * SIV_DQ))
                    cg_forces.append(oxygen_forces[o_idx]*0.5*rotate_ratio)
                    AB_indices.append(idx_copy)
                    AB_positions.append(si_pos + norm_V_Si_O * SIV_DQ)
                else:
                    cg_atoms.append(('ON', molid, si_pos + norm_V_Si_O * SIV_DQ))
                    cg_forces.append(oxygen_forces[o_idx]*rotate_ratio)
            
            idx_copy += 1

        for o_idx, h_idx in zip(cluster['O'], cluster['H']):
            v_Si_O = oxygen_positions[o_idx] - si_pos
            norm_V_Si_O = v_Si_O / np.linalg.norm(v_Si_O)
            cg_atoms.append(('V', molid, si_pos - norm_V_Si_O * SIV_VQ)) 
            cg_forces.append(np.array([0., 0., 0.]))

        bond_type = bond_types["Si-D"]
        cg_bonds.extend([(bond_type, idx, idx+1),
                        (bond_type, idx, idx+2),
                        (bond_type, idx, idx+3),
                        (bond_type, idx, idx+4)])
        bond_type = bond_types["D-V"]
        cg_bonds.extend([(bond_type, idx+1, idx+5),
                        (bond_type, idx+2, idx+6),
                        (bond_type, idx+3, idx+7),
                        (bond_type, idx+4, idx+8)])        
        bond_type = bond_types["D-D"]
        cg_bonds.extend([(bond_type, idx+1, idx+2),
                        (bond_type, idx+1, idx+3),
                        (bond_type, idx+1, idx+4),
                        (bond_type, idx+2, idx+3),
                        (bond_type, idx+2, idx+4),
                        (bond_type, idx+3, idx+4)])   
        
        angle_type = angle_types['SIV']
        cg_angles.extend([(angle_type, idx, idx+1, idx+5),
                        (angle_type, idx, idx+2, idx+6),
                        (angle_type, idx, idx+3, idx+7),
                        (angle_type, idx, idx+4, idx+8)])    

        molid += 1

    # Find all the C's connected to N, then find the C-C connected to C, and record the coordinates of N and C-C
    for n_idx, n_pos in enumerate(nitrogen_positions):
        cluster = {"N": n_idx, "C1": [], "C2": [], "C3": []}
        
        # The first stage: find the N-C connection and find the C in the range of 1.3-1.5Å around the N atom
        dists, c_indices = c_tree.query(n_pos, k=4)
        valid_c = [i for i, d in zip(c_indices, dists) if 1.5 <= d <= 1.8]
        assert len(valid_c) == 4, "N旁边连接C的数目不为4"
        cluster['C1'] = valid_c
        
        # The second stage: find the C-C connection
        for c_idx in valid_c:
            # Find other Cs in the range of 1.4-1.6Å around the C atom
            dists, neighbor_indices = c_tree.query(carbon_positions[c_idx], k=2)
            valid_c1 = [
                i for i, d in zip(neighbor_indices, dists) if 1.5 <= d <= 1.8 and i != c_idx 
            ]
            
            dists, neighbor_indices = c_tree.query(carbon_positions[valid_c1[0]], k=3)
            valid_c2 = [
                i for i, d in zip(neighbor_indices, dists) if 1.5 <= d <= 1.8 and i != c_idx and i != valid_c1[0]
            ]

            assert len(valid_c1) == 1 and len(valid_c2) == 1, "TPA structure is unreasonable"
            cluster['C2'].extend(valid_c1)
            cluster['C3'].extend(valid_c2)

        idx = len(cg_atoms) + 1
        cg_atoms.append(('N', molid, n_pos))
        force, weight = 0.0, 0.0
        for c_idx in cluster['C1']:
            force += carbon_forces[c_idx] * elem_masses['C']
            weight += elem_masses['C']
            
            dists, neighbor_indices = h_tree.query(carbon_positions[c_idx], k=2)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx] * elem_masses['H']
                weight += elem_masses['H']
        cg_forces.append(force/weight)

        for c2_idx, c3_idx in zip(cluster['C2'],cluster['C3']):
            force, weight = 0.0, 0.0
            center_pos = (carbon_positions[c2_idx] + carbon_positions[c3_idx]) / 2.0
            cg_atoms.append(('C', molid, center_pos))

            force += (carbon_forces[c2_idx] + carbon_forces[c3_idx]) * elem_masses['C']
            weight += 2 * elem_masses['C']

            dists, neighbor_indices = h_tree.query(carbon_positions[c2_idx], k=2)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx] * elem_masses['H']
                weight += elem_masses['H']

            dists, neighbor_indices = h_tree.query(carbon_positions[c3_idx], k=2)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx] * elem_masses['H']
                weight += elem_masses['H']

            cg_forces.append(force/weight)

        bond_type = bond_types["N-C"]
        angle_type = angle_types["TPA"]
        cg_bonds.extend([(bond_type, idx, idx+1), 
                            (bond_type, idx, idx+2), 
                            (bond_type, idx, idx+3), 
                            (bond_type, idx, idx+4)])
        cg_angles.extend([(angle_type, idx+1, idx, idx+2), 
                            (angle_type, idx+1, idx, idx+3), 
                            (angle_type, idx+1, idx, idx+4), 
                            (angle_type, idx+2, idx, idx+3), 
                            (angle_type, idx+2, idx, idx+4), 
                            (angle_type, idx+3, idx, idx+4)])
        
        molid += 1

    for na_idx, na_pos in enumerate(sodium_positions):
        cg_atoms.append(('Na', molid, na_pos))
        force = sodium_forces[na_idx]
        cg_forces.append(force)

        molid += 1

    return cg_atoms, cg_bonds, cg_angles, cg_forces

def write_lammps_data(filename, atoms, bonds, angles, box_margin=20.0):
    coords = np.array([atom[2] for atom in atoms])
    xlo, ylo, zlo = coords.mean(axis=0) - box_margin
    xhi, yhi, zhi = coords.mean(axis=0) + box_margin
    
    with open(filename, 'w') as f:
        f.write("LAMMPS Data File (Generated by Script)\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n")
        f.write(f"{len(atom_types.keys())} atom types\n")
        f.write(f"{len(bond_types.keys())} bond types\n")
        f.write(f"{len(angle_types.keys())} angle types\n\n")

        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
                
        f.write("Atoms\n\n")
        for i, atom in enumerate(atoms, 1):
            if atom[0] == 'Na':
                f.write(f"{i} {atom[1]} {atom_types[atom[0]]} 1.0000 {atom[2][0]:.6f} {atom[2][1]:.6f} {atom[2][2]:.6f}\n")
            elif atom[0] == 'N':
                f.write(f"{i} {atom[1]} {atom_types[atom[0]]} 0.3540 {atom[2][0]:.6f} {atom[2][1]:.6f} {atom[2][2]:.6f}\n")
            elif atom[0] == 'C':
                f.write(f"{i} {atom[1]} {atom_types[atom[0]]} 0.1615 {atom[2][0]:.6f} {atom[2][1]:.6f} {atom[2][2]:.6f}\n")
            else:
                f.write(f"{i} {atom[1]} {atom_types[atom[0]]} 0.0000 {atom[2][0]:.6f} {atom[2][1]:.6f} {atom[2][2]:.6f}\n")
        
        if bonds:
            f.write("\nBonds\n\n")
            for i, bond in enumerate(bonds, 1):
                f.write(f"{i} {bond[0]} {bond[1]} {bond[2]}\n")
        
        if angles:
            f.write("\nAngles\n\n")
            for i, angle in enumerate(angles, 1):
                f.write(f"{i} {angle[0]} {angle[1]} {angle[2]} {angle[3]}\n")

if __name__ == "__main__":

    g16dir = Path(sys.argv[1])
    outparentdir = Path(sys.argv[2])
    
    g16log = list(g16dir.glob('*.log'))[0]
    g16name = g16log.stem

    g16position = g16dir / f"{g16name}_position.xyz"
    g16force = g16dir / f"{g16name}_force.xyz"
    g16energy = g16dir / f"{g16name}_energy.xyz"

    outdir = outparentdir / g16dir
    outdir.mkdir(parents=True, exist_ok=True)
    cgposition = outdir / f"{g16name}.data"
    cgforce = outdir / f"{g16name}_force.xyz"
    cgenergy = outdir / f"{g16name}_energy.xyz"
    cgmapping = outdir / f"{g16name}_3d.html"

    atoms_info, energy = parse_gaussian_output(g16log)
    cg_atoms, cg_bonds, cg_angles, cg_forces = mapping_rules(atoms_info)

    energy_conversion_factor = 627.509474
    forces_conversion_factor = 627.509474 / 0.529177

    atoms_number = len(atoms_info)
    cg_atoms_number = len(cg_atoms)
    with open(g16position, 'w') as gp_wf, \
             open(g16force, 'w') as gf_wf, \
             open(g16energy, 'w') as ge_wf, \
             open(cgforce, 'w') as cf_wf, \
             open(cgenergy, 'w') as ce_wf:
        gp_wf.write(f"{atoms_number}\nPositions\n")
        gf_wf.write(f"{atoms_number}\nForces\n")

        cf_wf.write(f"{cg_atoms_number}\nForces\n")
        for atom in atoms_info:
            gp_wf.write(f"{atom['Element']} {atom['X']} {atom['Y']} {atom['Z']}\n")
            gf_wf.write(f"{atom['Element']} {atom['Fx']} {atom['Fy']} {atom['Fz']}\n")
        for atom, force in zip(cg_atoms, cg_forces):
            cf_wf.write(f"{atom_types[atom[0]]} {force[0]*forces_conversion_factor} {force[1]*forces_conversion_factor} {force[2]*forces_conversion_factor}\n")    # convert to kcal/(mol.A)

        ge_wf.write(f'{energy}')
        ce_wf.write(f'{energy*energy_conversion_factor}')    # convert to kcal/mol

    write_lammps_data(cgposition, cg_atoms, cg_bonds, cg_angles)





        