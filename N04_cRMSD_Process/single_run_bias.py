"""Schedule cluster updates within a single LAMMPS Verlet run."""
from pathlib import Path
import uuid

import numpy as np
from ratchet_force import orthogonal_box


def run_bias_segment(pirmsd, lmp, directory, start_iter, niterations, md_steps,
                     output_iteration, branch_interval, rng, signature):
    if start_iter == niterations:
        return
    driver = pirmsd.force_driver
    comm = pirmsd.comm
    first = int(lmp.extract_global('ntimestep'))
    origin = first - start_iter * md_steps
    final = origin + niterations * md_steps
    stride = md_steps * max(output_iteration, 1)
    token = comm.bcast(uuid.uuid4().hex if pirmsd.rank == 0 else None, root=0)
    pattern = Path(directory).resolve() / f'checkpoint_{token}_*.restart'
    types = np.array(lmp.gather_atoms('type', 0, 1)).reshape(-1)
    # Native restarts are written by LAMMPS after the second Verlet half-step.
    # Pair that file with the still-current Python state before the next evaluation.
    cadence = str(stride)
    if origin % stride:
        lmp.command(f'variable pirmsd_next_restart equal "{origin}+(floor((step-({origin}))/{stride})+1)*{stride}"')
        cadence = 'v_pirmsd_next_restart'
    lmp.command(f'restart {cadence} "{pattern}"')
    pending = None
    last = None

    def commit(step):
        restart = Path(str(pattern).replace('*', str(step)))
        pirmsd.commit_restart(directory, restart, (step-origin)//md_steps, rng, signature)

    def callback(caller, timestep, nlocal, tags, x, forces):
        nonlocal last, pending
        try:
            if driver.error:
                raise RuntimeError(driver.error)
            if pending is not None and timestep > pending:
                commit(pending)
                pending = None
            elapsed = timestep - origin
            resumed_setup = timestep == first and start_iter > 0
            fresh = timestep != last and not resumed_setup
            update = fresh and timestep < final and elapsed % branch_interval == 0
            report = fresh and elapsed > 0 and elapsed % md_steps == 0
            if update or report:
                xyz = np.asarray(lmp.gather_atoms('x', 1, 3)).reshape(-1, 3)
                box = orthogonal_box(lmp)
            if update:
                err = None
                try:
                    pirmsd.run(elapsed // branch_interval, xyz, types, box)
                except Exception as exc:
                    err = repr(exc)
                errors = comm.allgather(err)
                if any(errors):
                    raise RuntimeError(next(e for e in errors if e))
            driver.callback(caller, timestep, nlocal, tags, x, forces)
            driver.check()
            if report:
                error = None
                if pirmsd.rank == 0:
                    try:
                        pirmsd.record_bias_iteration(elapsed // md_steps - 1, xyz, box)
                    except Exception as exc:
                        error = repr(exc)
                error = comm.bcast(error, root=0)
                if error:
                    raise RuntimeError(error)
                if elapsed % stride == 0:
                    pending = timestep
            last = timestep
        except Exception as exc:
            driver.error = str(exc) or type(exc).__name__
            forces[:] = 0.
            lmp.fix_external_set_energy_global(driver.fix_id, 0.)
            lmp.fix_external_set_virial_global(driver.fix_id, [0.] * 6)
            lmp.force_timeout()

    lmp.set_fix_external_callback(driver.fix_id, callback)
    run = f'run {final - first}'
    if start_iter:
        # Preserve the original time span for thermostats with ramped targets.
        run += f' start {origin} stop {final}'
    lmp.command(run)
    driver.check()
    if last != final or int(lmp.extract_global('ntimestep')) != final:
        raise RuntimeError('Bias integration stopped before the requested final timestep')
    if pending != final:
        restart = Path(str(pattern).replace('*', str(final)))
        lmp.command(f'write_restart "{restart}"')
    commit(final)
    lmp.command('restart 0')
