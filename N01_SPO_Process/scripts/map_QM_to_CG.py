import sys
import re
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree

SIV_DQ = 1.03171  # Si-SP bond length in jobs/module_files/lmp_qm.in
SIV_VQ = 2.32650 - SIV_DQ  # opposite Si-VS radius from SP-VS bond length

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
    content=Path(log_file).read_text();lines=content.splitlines()
    symbols={1:'H',6:'C',7:'N',8:'O',11:'Na',14:'Si',35:'Br'}
    number=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?'
    def val(x):return float(x.replace('D','E').replace('d','e'))
    orientations=[];forces=[];energies=[]
    for i,line in enumerate(lines):
        match=re.search(r'SCF Done:.*?=\s*('+number+')',line)
        if match:energies.append((i,val(match.group(1))))
        if 'Standard orientation:' in line or 'Input orientation:' in line:
            rows=[]
            for row in lines[i+1:]:
                cols=row.split()
                if len(cols)==6 and cols[0].isdigit() and cols[1].isdigit():
                    rows.append((int(cols[0]),int(cols[1]),[val(v) for v in cols[3:6]]))
                elif rows:break
            if rows:orientations.append((i,'Standard' in line,rows))
        if 'Forces (Hartrees/Bohr)' in line:
            rows=[]
            for row in lines[i+1:]:
                cols=row.split()
                if len(cols)==5 and cols[0].isdigit() and cols[1].isdigit():
                    rows.append((int(cols[0]),int(cols[1]),[val(v) for v in cols[2:5]]))
                elif rows:break
            if rows:forces.append((i,rows))
    if content.rfind('Error termination')>content.rfind('Normal termination'):raise ValueError('Gaussian ended with an error')
    if not forces or not energies:raise ValueError('No complete Gaussian energy/force evaluation')
    position,force=forces[-1]
    if not any('Normal termination' in line for line in lines[position:]):raise ValueError('Gaussian did not terminate normally')
    available=[o for o in orientations if o[0]<position]
    standard=[o for o in available if o[1]]
    if not available:raise ValueError('No coordinates for force frame')
    coords=(standard or available)[-1][2]
    energy=[e for i,e in energies if i<position][-1]
    if [(i,z) for i,z,_ in coords]!=[(i,z) for i,z,_ in force]:raise ValueError('Gaussian coordinate/force identities differ')
    atoms=[]
    for (_,z,x),(_,_,f) in zip(coords,force):
        if z not in symbols:raise ValueError('Unsupported element atomic number '+str(z))
        if not np.isfinite(x+f).all():raise ValueError('Nonfinite Gaussian data')
        atoms.append(dict(Element=symbols[z],X=x[0],Y=x[1],Z=x[2],Fx=f[0],Fy=f[1],Fz=f[2]))
    return atoms,energy


def conserve_wrench(positions,initial_forces,source_positions,source_forces):
    """Minimum correction preserving molecular net force and torque.

    This fixes a declared rigid-body force projection; it is not a unique
    atomistic-to-CG force map for arbitrary internal deformations.
    """
    positions=np.asarray(positions);origin=positions[0];r=positions-origin
    target_f=np.asarray(source_forces).sum(axis=0)
    target_t=np.cross(np.asarray(source_positions)-origin,source_forces).sum(axis=0)
    matrix=np.zeros((6,3*len(r)))
    for i,(x,y,z) in enumerate(r):
        matrix[:3,3*i:3*i+3]=np.eye(3)
        matrix[3:,3*i:3*i+3]=[[0,-z,y],[z,0,-x],[-y,x,0]]
    initial=np.asarray(initial_forces).ravel();target=np.r_[target_f,target_t]
    corrected=initial+np.linalg.lstsq(matrix,target-matrix@initial,rcond=1e-12)[0]
    if not np.allclose(matrix@corrected,target,atol=1e-9,rtol=1e-9):raise ValueError('CG sites cannot represent source force/torque')
    return corrected.reshape(-1,3)


def mapping_rules(atoms_info, si_sp=SIV_DQ, si_vs=SIV_VQ):
    unsupported={a['Element'] for a in atoms_info}-{'Si','O','H','C','N','Na'}
    if unsupported:raise ValueError('Unsupported AA mapping elements: '+str(sorted(unsupported)))
    SIV_DQ,SIV_VQ=float(si_sp),float(si_vs)
    if min(SIV_DQ,SIV_VQ)<=0:raise ValueError("Mapping radii must be positive")
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
        
        if len(valid_o)!=4:raise ValueError('Silicate mapping requires four oxygen neighbors')
        source_positions=[si_pos];source_forces=[silica_forces[si_idx]]
        cluster = {"Si": si_idx, "O": [], "H": []}
        for o_idx in valid_o:            
            # Find O-linked H atoms (tolerance±0.1Å)
            h_dists, h_indices = h_tree.query(oxygen_positions[o_idx], k=2) if hydrogen_positions.size else ([],[])  # 最多找1个H
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
            source_positions.append(oxygen_positions[o_idx])
            source_forces.append(oxygen_forces[o_idx].copy())

            if h_idx is not None:
                cg_atoms.append(('OH', molid, si_pos + norm_V_Si_O * SIV_DQ))
                force = oxygen_forces[o_idx]+hydrogen_forces[h_idx]
                source_positions.append(hydrogen_positions[h_idx]);source_forces.append(hydrogen_forces[h_idx])
                cg_forces.append(force)
            else:
                si_dists, si_indices = si_tree.query(oxygen_positions[o_idx], k=2)
                connected_si = [si_idx for si_idx, d in zip(si_indices, si_dists) if 1.5 <= d <= 1.8]
                if len(connected_si) == 2:
                    cg_atoms.append(('OB', molid, si_pos + norm_V_Si_O * SIV_DQ))
                    cg_forces.append(oxygen_forces[o_idx]*0.5)
                    source_forces[-1]*=.5
                    AB_indices.append(idx_copy)
                    AB_positions.append(si_pos + norm_V_Si_O * SIV_DQ)
                else:
                    cg_atoms.append(('ON', molid, si_pos + norm_V_Si_O * SIV_DQ))
                    cg_forces.append(oxygen_forces[o_idx])
            
            idx_copy += 1

        cg_forces[idx-1:idx+4]=conserve_wrench([a[2] for a in cg_atoms[idx-1:idx+4]],
            cg_forces[idx-1:idx+4],source_positions,source_forces).tolist()

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
        force, weight = nitrogen_forces[n_idx].copy(), 0.0
        for c_idx in cluster['C1']:
            force += carbon_forces[c_idx]
            weight += elem_masses['C']
            
            dists, neighbor_indices = h_tree.query(carbon_positions[c_idx], k=2)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx]
                weight += elem_masses['H']
        cg_forces.append(force)

        for c2_idx, c3_idx in zip(cluster['C2'],cluster['C3']):
            force, weight = 0.0, 0.0
            center_pos = (carbon_positions[c2_idx] + carbon_positions[c3_idx]) / 2.0
            cg_atoms.append(('C', molid, center_pos))

            force += carbon_forces[c2_idx] + carbon_forces[c3_idx]
            weight += 2 * elem_masses['C']

            dists, neighbor_indices = h_tree.query(carbon_positions[c2_idx], k=2)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx]
                weight += elem_masses['H']

            dists, neighbor_indices = h_tree.query(carbon_positions[c3_idx], k=3)
            connected_h = [i for i, d in zip(neighbor_indices, dists) if 1.0 <= d <= 1.2]
            for h_idx in connected_h:
                force += hydrogen_forces[h_idx]
                weight += elem_masses['H']

            cg_forces.append(force)

        carbon_ids=sorted(set(cluster['C1']+cluster['C2']+cluster['C3']))
        hydrogen_ids=sorted({h for c in carbon_ids for h in h_tree.query_ball_point(carbon_positions[c],1.2)})
        source_positions=[n_pos]+list(carbon_positions[carbon_ids])+list(hydrogen_positions[hydrogen_ids])
        source_forces=[nitrogen_forces[n_idx]]+list(carbon_forces[carbon_ids])+list(hydrogen_forces[hydrogen_ids])
        cg_forces[idx-1:idx+4]=conserve_wrench([a[2] for a in cg_atoms[idx-1:idx+4]],
            cg_forces[idx-1:idx+4],source_positions,source_forces).tolist()

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

    aa_positions=np.array([[a['X'],a['Y'],a['Z']] for a in atoms_info])
    aa_forces=np.array([[a['Fx'],a['Fy'],a['Fz']] for a in atoms_info])
    cg_positions=np.array([a[2] for a in cg_atoms])
    if not cg_atoms or not np.allclose(np.sum(cg_forces,axis=0),aa_forces.sum(axis=0),atol=1e-9,rtol=1e-9):
        raise ValueError('Unassigned atoms or nonconservative mapped force')
    if not np.allclose(np.cross(cg_positions,cg_forces).sum(axis=0),np.cross(aa_positions,aa_forces).sum(axis=0),atol=1e-8,rtol=1e-8):
        raise ValueError('Mapped molecular torque mismatch')
    return cg_atoms, cg_bonds, cg_angles, cg_forces

def write_lammps_data(filename, atoms, bonds, angles, box_margin=20.0):
    coords = np.array([atom[2] for atom in atoms])
    xlo, ylo, zlo = coords.min(axis=0) - box_margin
    xhi, yhi, zhi = coords.max(axis=0) + box_margin
    
    with open(filename, 'w') as f:
        f.write("LAMMPS Data File (Generated by Script)\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n")
        f.write(f"{max(atom_types.values())} atom types\n")
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

def convert_sample(g16dir,qm_root,output_root,si_sp=SIV_DQ,si_vs=SIV_VQ):
    g16dir=Path(g16dir).resolve();relative=g16dir.relative_to(Path(qm_root).resolve())
    candidates=sorted(g16dir.glob('*.log'))
    expected=g16dir/(g16dir.name+'.log')
    if expected.is_file():log=expected
    elif len(candidates)==1:log=candidates[0]
    else:raise ValueError('Ambiguous Gaussian logs in '+str(g16dir))
    name=log.stem;out=Path(output_root).resolve()/relative;out.mkdir(parents=True,exist_ok=True)
    atoms,energy=parse_gaussian_output(log)
    cg,bonds,angles,forces=mapping_rules(atoms,si_sp,si_vs)
    factor=627.509474/0.529177
    write_lammps_data(out/(name+'.data'),cg,bonds,angles)
    (out/(name+'_energy.xyz')).write_text(str(energy*627.509474)+'\n')
    (out/(name+'_force.xyz')).write_text(str(len(cg))+'\nForces kcal/mol/angstrom; rows follow LAMMPS atom IDs\n'+
        ''.join('{} {} {} {}\n'.format(atom_types[a[0]],*(np.asarray(f)*factor)) for a,f in zip(cg,forces)))
    import json
    (out/'mapping.json').write_text(json.dumps(dict(source=str(log),si_sp=si_sp,si_vs=si_vs,
        force_projection='group sums with minimum correction preserving molecular force and torque; virtual-site target forces zero'),indent=2)+'\n')
    return out


if __name__=='__main__':
    import argparse
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from workflow import load_workflow
    parser=argparse.ArgumentParser(description='Map one Gaussian sample; raw QM inputs remain read-only')
    parser.add_argument('sample',type=Path);parser.add_argument('--config');parser.add_argument('--qm-root',type=Path)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--si-sp',type=float);parser.add_argument('--si-vs',type=float)
    args=parser.parse_args();cfg=load_workflow(args.config)
    # Geometry has one source: the active QM LAMMPS template's Si-SP and SP-VS bonds.
    template=(Path(cfg['paths']['templates'])/'lmp_qm.in').read_text()
    bonds={int(m[0]):float(m[1]) for m in re.findall(r'^bond_coeff\s+(\d+)\s+\S+\s+([0-9.]+)',template,re.M)}
    si_sp=args.si_sp if args.si_sp is not None else bonds[1]
    si_vs=args.si_vs if args.si_vs is not None else bonds[2]-si_sp
    print(convert_sample(args.sample,args.qm_root or cfg['paths']['qm_source'],args.output_root or cfg['paths']['cg_reference'],si_sp,si_vs))
