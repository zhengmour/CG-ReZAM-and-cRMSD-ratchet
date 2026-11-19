"""LAMMPS force callback with the ratchet history and checkpoint on rank zero."""
import numpy as np
from mpi4py import MPI

from ratchet_force import RatchetForce, orthogonal_box


class DistributedRatchetForce(RatchetForce):
    """Workers retain the current diagnostics and receive their owned atom forces."""

    def state(self):
        if self.pirmsd.rank != 0:
            raise RuntimeError("Only rank zero owns the MPI bias checkpoint")
        return super().state()

    def callback(self, caller, timestep, nlocal, tags, x, forces):
        forces[:] = 0.
        lmp = self.lmp
        if not self.enabled:
            lmp.fix_external_set_energy_global(self.fix_id, 0.)
            lmp.fix_external_set_virial_global(self.fix_id, [0.] * 6)
            return

        comm = self.pirmsd.comm
        root = self.pirmsd.rank == 0
        # Atom ownership and local order can change at every neighbor rebuild.
        counts = comm.gather(nlocal, root=0)
        id_recv = xyz_recv = force_send = None
        if root:
            natoms = sum(counts)
            atom_ids = np.empty(natoms, dtype=np.int64)
            coords = np.empty((natoms, 3))
            coord_counts = 3 * np.asarray(counts, dtype=np.intp)
            id_recv = [atom_ids, counts, MPI.INT64_T]
            xyz_recv = [coords, coord_counts, MPI.DOUBLE]
        comm.Gatherv(np.ascontiguousarray(tags, dtype=np.int64), id_recv, root=0)
        comm.Gatherv(np.ascontiguousarray(x, dtype=np.float64), xyz_recv, root=0)

        message = None
        if root:
            try:
                if self.error:
                    raise RuntimeError(self.error)
                # The metric and older checkpoints index coordinates by atom ID - 1.
                atom_ids -= 1
                if (np.any(atom_ids < 0) or np.any(atom_ids >= natoms)
                        or not np.all(np.bincount(atom_ids, minlength=natoms) == 1)):
                    raise ValueError("PIRMSD requires consecutive, unique atom IDs")
                xyz = np.empty_like(coords)
                xyz[atom_ids] = coords
                cache = self.evaluate(timestep, xyz, orthogonal_box(lmp))
                summary = {k: v for k, v in cache.items() if k not in ("forces", "ratchet")}
                if cache["decision"]["apply"]:
                    owned_forces = np.asarray(cache["forces"], dtype=np.float64)[atom_ids]
                    force_send = [owned_forces, coord_counts, MPI.DOUBLE]
                message = (None, summary, self.evaluations)
            except Exception as exc:
                message = (str(exc) or type(exc).__name__, None, self.evaluations)
        error, summary, self.evaluations = comm.bcast(message, root=0)
        if error is not None:
            self.error = error
            lmp.force_timeout()
            lmp.fix_external_set_energy_global(self.fix_id, 0.)
            lmp.fix_external_set_virial_global(self.fix_id, [0.] * 6)
            return

        if not root:
            self.cache = summary
        if summary["decision"]["apply"]:
            comm.Scatterv(force_send, forces, root=0)
        lmp.fix_external_set_energy_global(self.fix_id, summary["energy"])
        lmp.fix_external_set_virial_global(self.fix_id, summary["virial"])
