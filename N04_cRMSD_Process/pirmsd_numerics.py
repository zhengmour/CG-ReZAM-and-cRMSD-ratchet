"""Numerical helpers for the legacy global/per-cluster PIRMSD coordinate.

Distances and impulses use LAMMPS real units.
"""
import numpy as np

FTM2V_REAL = 1.0 / 48.88821291**2


def centered_gradient(entries, atoms):
    """Differentiate S and W through y_i=x_i-mean(x)+reference_center.

    Preserve matched weights, and include unmatched atoms' anchor reaction.
    Membership, periodic images and the selected rotation/mapping are frozen.
    """
    result = np.zeros((len(atoms), 9))
    result[:, :2] = np.asarray(atoms)[:, :2]
    lookup = {int(row[0]): i for i, row in enumerate(atoms)}
    for entry in entries:
        result[lookup[int(entry[0])], 2:] = entry[2:9]
    result[:, 2:8] -= result[:, 2:8].sum(axis=0) / len(atoms)
    return result.tolist()
