import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from pathlib import Path
import string

# Element attribute properties
ELEMENT_PROPERTIES = {
    'C': {'color': '#808080', 'vdw': 1.70},  # unit: Å
    'O': {'color': '#FF0000', 'vdw': 1.52},
    'N': {'color': '#0000FF', 'vdw': 1.55},
    'H': {'color': '#FFFFFF', 'vdw': 1.20},
   'SI': {'color': '#FFFF00', 'vdw': 2.10},
   'Al': {'color': '#FFFF00', 'vdw': 2.10},
   'Na': {'color': '#800080', 'vdw': 2.27}
}
# Default properties
DEFAULT_PROPS = {'color': '#808080', 'vdw': 1.50}

def generate_sphere(center, radius, resolution=20):
    # Generate Sphere Surface Mesh
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    
    # Spherical coordinates to Cartesian coordinates
    x = center[0] + radius * np.outer(np.cos(phi), np.sin(theta))
    y = center[1] + radius * np.outer(np.sin(phi), np.sin(theta))
    z = center[2] + radius * np.outer(np.ones(resolution), np.cos(theta))
    
    # Generate a triangular patch
    verts = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v1 = (x[i,j], y[i,j], z[i,j])
            v2 = (x[i+1,j], y[i+1,j], z[i+1,j])
            v3 = (x[i+1,j+1], y[i+1,j+1], z[i+1,j+1])
            v4 = (x[i,j+1], y[i,j+1], z[i,j+1])
            verts.append([v1, v2, v3, v4])
    return verts

def plot_molecules(a_atoms, positions, cutoff, element_props=ELEMENT_PROPERTIES, scale=0.5, bg_color='white', shell_alpha=0.2, resolution=15):
    """Visualize the molecular distribution in 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    coords = a_atoms['positions']
    elements = a_atoms['elements']

    # Prepare the legend handle
    legend_handles = []

    # Traverse all atoms
    for elem, pos in zip(elements, coords):
        # Get atomic properties
        props = element_props.get(elem, DEFAULT_PROPS)
        color = props.get('color', DEFAULT_PROPS['color'])
        vdw_radius = props.get('vdw', DEFAULT_PROPS['vdw']) * scale
        
        # Spawn shell spheres
        sphere = generate_sphere(pos, vdw_radius, resolution)
        shell = Poly3DCollection(sphere, 
                                alpha=shell_alpha, 
                                facecolors=color,
                                edgecolors='none')
        ax.add_collection3d(shell)
        
        # Draw the center point of the atom
        ax.scatter(*pos, s=20, c=color, alpha=0.9, depthshade=False)         

    # Calculate the distance from the B molecule to the center of mass A
    distances = np.linalg.norm(positions, axis=1)
    
    # Plot the B molecule (color mapped for distance)
    sc = ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                    c=distances, cmap='coolwarm', s=15, alpha=0.6,
                    label='Molecule B')
    legend_handles.append(
        Line2D([0], [0], 
               marker='X', 
               color='#FF00FF',
               label='B',
               markersize=10,
               linestyle='')
    )    
    
    # Add cutoff spheres (translucent)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = cutoff * np.outer(np.cos(u), np.sin(v))
    y = cutoff * np.outer(np.sin(u), np.sin(v))
    z = cutoff * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5,
                     label='Cutoff Sphere')
    
    # Set the axis
    ax.set_xlim([-cutoff*1.1, cutoff*1.1])
    ax.set_ylim([-cutoff*1.1, cutoff*1.1])
    ax.set_zlim([-cutoff*1.1, cutoff*1.1])
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Add legends and color scales
    plt.colorbar(sc, ax=ax, label='Distance from A (Å)')
    ax.legend()
    
    # Add a custom legend
    leg = ax.legend(handles=legend_handles, 
                    loc='upper right', 
                    bbox_to_anchor=(1.05, 1),
                    fontsize=10, 
                    frameon=False)
    for text in leg.get_texts():
        text.set_color('black')

    plt.title('Molecular Distribution Visualization')
    plt.savefig("A-B.png", dpi=1200)

def read_gro(filename, element_props=ELEMENT_PROPERTIES):
    """Read the GRO file and return the atomic coordinates and centroid"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip the header line and read the atomic number
    num_atoms = int(lines[1].strip())
    
    # Resolve atomic coordinates
    atoms = {'positions': [], 'elements': [], 'vdws': []}
    for line in lines[2:2+num_atoms]:  
        if len(line) < 44:  
            continue
        try:
            atom_name = line[10:15].strip()
            element = atom_name.strip(string.digits)  
            atoms['elements'].append(element.upper())
            x = float(line[20:28].strip()) * 10.0
            y = float(line[28:36].strip()) * 10.0
            z = float(line[36:44].strip()) * 10.0
            atoms['positions'].append([x, y, z])
            props = element_props.get(element, DEFAULT_PROPS)
            final_vdw = props.get('vdw', DEFAULT_PROPS['vdw'])
            atoms['vdws'].append(final_vdw)
        except:
            continue
    
    atoms['positions'] = np.array(atoms['positions'])
    atoms['vdws'] = np.array(atoms['vdws'])

    # Move the centroid to [0,0,0]
    atoms['positions'] = atoms['positions'] - np.mean(atoms['positions'], axis=0)

    return atoms

def regular_sphere_sampling(n):
    """
    Spherical Equal Spacing Sampling Algorithm (Fibonacci Grid Improvement)
    :p aram n: The exact number of points that need to be generated
    :return: (n,3) unit spherical coordinate array
    """
    indices = np.arange(n, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Improved isometric parameterization formula
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2*(indices + 0.5) / n)
    
    # Convert to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.column_stack((x, y, z))

def adaptive_sphere_sampling(cutoff, total_points, level_number=None):
    """
    Radius adaptive spherical sampling algorithm
    :p aram cutoff: The maximum sampling radius
    :p aram total_points: Total number of sampled points
    :p aram level_number: The number of layers given
    :return: (N,3) 3D coordinate array
    """
    # Generate radius distribution (exponential decay)
    if level_number:
        r_base = np.geomspace(2.0, cutoff, num=level_number, endpoint=True)
    else:
        r_base = np.geomspace(2.0, cutoff, num=int(total_points**0.5), endpoint=True)

    print(f"r: {r_base}")
    # The number of points on each layer is the same
    density_weights = np.repeat(1, r_base.shape[0])
    points_dist = (density_weights / density_weights.sum() * total_points).astype(int)
    
    # Generate spherical sampling points
    points = []
    for r, n in zip(r_base, points_dist):
        if n < 4: 
            n = 4
        sphere = regular_sphere_sampling(n)
        points.append(sphere * r)
    
    return np.concatenate(points)

def generate_rotations(N):
    """ generates N evenly distributed rotation direction matrices
    Parameter: N: The number of rotation directions that need to be generated
    Return: rotations: (N,3,3) rotate the matrix array
    """
    directions = regular_sphere_sampling(N)
    
    # Generate a rotation matrix
    rotations = []
    for d in directions:
        # Align to the Z-axis direction
        rot = Rotation.align_vectors([[0, 0, 1]], [d])[0]
        rotations.append(rot.as_matrix())
    
    return np.array(rotations)

def filter_positions(a_atoms, b_atoms, positions, rotations, scale=0.80, batch_size=100):
    """
    Screen for spatial locations that meet van der Waals radius constraints
    Parameter:
        a_atoms: (N,3) atomic coordinates of the A molecule and (N,) atomic van der Waals radius of the A molecule
        b_atoms: (M,3) Relative coordinates of B atoms and (M,) van der Waals radius of B atoms
        positions: (K,3) Candidate position coordinates
        rotaions: (L,3,3) rotation matrix
        batch_size: Batch size (memory optimization)
    Return:
        valid_positions: Array of effective position coordinates
    """
    valid_mask = np.ones(len(positions), dtype=bool)
    a_coords = a_atoms['positions']
    b_coords = b_atoms['positions']
    a_vdws = a_atoms['vdws'] * scale
    b_vdws = b_atoms['vdws'] * scale

    valid_positions = []
    valid_rotations = []

    # Batch processing avoids memory overflows
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i+batch_size]
        
        for rotation in rotations:
            # Apply rotation to B coordinates (M,3)
            rotated_b = np.dot(b_coords, rotation.T)

            # 3D Broadcast Calculation (Batch, M, 3) = (Batch,1,3) + (M,3)
            batch_b_coords = batch[:, np.newaxis] + rotated_b
            
            # Flattening the Batch Dimension (Batch*M, 3)
            flat_batch_b = batch_b_coords.reshape(-1, 3)
            
            # Calculate all atomic pair spacing (Batch*M, N)
            dist_matrix = cdist(flat_batch_b, a_coords)
            
            # Generate van der Waals radius and matrix (M, N) -> (Batch*M, N)
            vdw_sums = (b_vdws[:, np.newaxis] + a_vdws).repeat(len(batch), axis=0)
            
            # Conflict detection (Batch*M， N)
            collision_flags = dist_matrix < vdw_sums
            
            # Packet by B Atom (Batch, M, N) -> (Batch, M)
            group_collisions = collision_flags.reshape(len(batch), len(b_coords), -1)
            valid_mask = ~np.any(np.any(group_collisions, axis=2), axis=1)
            
            # Record valid combinations
            valid_pos = batch[valid_mask]
            valid_positions.append(valid_pos)
            valid_rotations.append(np.repeat(rotation[np.newaxis, ...], len(valid_pos), axis=0))

            # If it is a single atom, exit the loop
            if len(b_atoms) == 1:
                break
    
    valid_positions = np.concatenate(valid_positions)
    valid_rotations = np.concatenate(valid_rotations)
    return valid_positions, valid_rotations

def generate_g16_input(a_atoms, b_atoms, position, rotation, charge, dir, filename):
    a_coords = a_atoms['positions']
    b_coords = np.dot(b_atoms['positions'], rotation.T) + position
    a_elems = a_atoms['elements']
    b_elems = b_atoms['elements']
    elements = np.concatenate((a_elems, b_elems))
    positions = np.concatenate((a_coords, b_coords), axis=0)

    with open(dir / f'{filename}.com', 'w') as wf:
        wf.write(f"""%nproc=32
%mem=40GB
%chk={filename}.chk
#P b3lyp/6-311G** force em=gd3bj SCRF=(Solvent=Water, SMD)

Title: Force and Energy

""")
        wf.write(f"{charge} 1\n")
        for element, position in zip(elements, positions):
            wf.write(f"{element} {position[0]} {position[1]} {position[2]}\n")
        wf.write(f"\n")

    with open(dir / 'sbatch.sh', 'w') as wf:
        wf.write(f"#!/bin/bash\n")
        wf.write(f"#SBATCH --job-name=zd-qm-{filename}\n")

        wf.write("""
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1

#TODO
export COM_DIR=<Path of G16>
export root=$COM_DIR
export PATH=$root:$PATH
source $root/bsd/g16.profile
export GAUSS_SCRDIR=<Path of G16 tmpdir>
export GAUSS_EXEDIR=$root

start_time=$(date +%s)

filename=$1
log=${filename[@]%.*}.log
g16 < $filename > $log

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "---------------------------------------------" >> ${log}
echo "Job started at: $(date -d @$start_time)" >> ${log}
echo "Job ended at:   $(date -d @$end_time)" >> ${log}
echo "Total elapsed time: $elapsed_time seconds" >> ${log}
echo "---------------------------------------------" >> ${log}

echo "Gaussian job completed in $elapsed_time seconds"

""")

if __name__ == '__main__':
    center_gro = Path(sys.argv[1])
    around_gro = Path(sys.argv[2])
    charge = int(sys.argv[3])
    N = 8

    center_name = center_gro.stem
    around_name = around_gro.stem

    center_atoms = read_gro(center_gro)
    around_atoms = read_gro(around_gro)

    initial_positions = adaptive_sphere_sampling(cutoff=12.0, total_points=4000, level_number=40)

    # Generate a rotation matrix
    rotation_matrices = generate_rotations(N)

    # Perform filtering
    filtered_positions, filter_rotations = filter_positions(
        a_atoms=center_atoms,
        b_atoms=around_atoms,
        positions=initial_positions,
        rotations=rotation_matrices,
        batch_size=100
    )

    # Perform visualizations
    plot_molecules(center_atoms, filtered_positions, cutoff=12.0)

    for position, rotation in zip(filtered_positions, filter_rotations):
        position_str = f"{position[0]:.4f}_{position[1]:.4f}_{position[2]:.4f}"
        # Calculate the absolute value of the matrix difference
        diff = np.abs(rotation_matrices - rotation)
        # Along the last two dimensions, check that all elements are less than tolerance
        matches = np.all(diff < 1e-6, axis=(1,2))
        # Get the matching index
        indices = np.where(matches)[0].tolist()
        rotation_str = f"{indices[0]}_{N}"

        filename = f"{center_name}-{around_name}-{position_str}-{rotation_str}"

        dir = Path(f"./data_set/QM2E/{center_name}-{around_name}/{filename}")
        dir.mkdir(parents=True, exist_ok=True)
        generate_g16_input(center_atoms, around_atoms, position, rotation, charge, dir, filename)

