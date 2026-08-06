# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment: a point mass on an "infinitely stiff, massless cable", and whether a static solve works.

Two ways to model a pendulum bob hanging from a fixed pivot, and how each represents the idealized
rigid, massless cable. The bob starts horizontal (arm along +X), 90 degrees from the hanging
equilibrium, with gravity perpendicular to the cable.

1. Particle bob + stiff spring (``_build_spring_pendulum``). The bob is a particle; the cable is a
   massless spring (only particles carry mass). This WORKS under ``SolverVBD(static_solve=True)`` and
   settles straight below the pivot in ~1 step. "Infinitely stiff" is only *approximated* by a large
   spring stiffness — VBD has no rigid/inextensible distance constraint for particles (the DISTANCE
   joint is unsupported and particles take no joints) — so a residual stretch ~ m*g/ke remains and
   shrinks toward zero as ke grows. It is robust because (a) the per-particle VBD solve guards a
   singular local Hessian (``if |det(h)| > 1e-8`` else skip) instead of dividing through it, and
   (b) once the bob moves under gravity the cable tension supplies geometric stiffness.

2. Rigid bob + BALL joint (``_build_rigid_pendulum``). The cable is *not* a separate body: a massless
   rigid link of length L is the offset baked into the BALL joint's child frame, and the hard BALL
   joint is the inextensible, "infinitely stiff" constraint. Under ordinary *dynamics* this is a
   genuinely inextensible cable (measured length stays L to ~1e-4). A *separate* zero-mass rigid body
   cannot be used for the cable: ``mass == 0`` gives ``inv_mass == 0``, which the solver treats as
   static (frozen), so it could not swing.

   This representation is NOT compatible with ``static_solve=True``: a body constrained only by a BALL
   joint has three unconstrained rotational DOFs (all rotations about the pivot). The joint supplies no
   rotational stiffness, and ``static_solve`` removes the inertia term (``I/dt^2``) that is the only
   thing regularizing those DOFs under dynamics, so the 6x6 angular block is rank-deficient. The direct
   ``ldlt6_solve`` has no pivot guard (unlike the particle path), so it divides by ~0 pivots and yields
   NaN -- at *every* configuration, including the exact hanging equilibrium (there the spin-about-cable
   mode is still a zero-stiffness, zero-force null direction that breaks the factorization). This is a
   structural singularity of the inertia-free rigid solve, not an instability of being far from rest.

Run directly to print the comparison:

    python3 newton/tests/test_vbd_pendulum_static_solve.py
"""

import argparse

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.solver_vbd import SolverVBD

_PIVOT = wp.vec3(0.0, 0.0, 3.0)
_CABLE_LENGTH = 1.0
_BOB_MASS = 1.0


def _build_rigid_pendulum(device):
    """Rigid bob on a massless rigid cable (a BALL-joint frame offset) from a fixed pivot.

    Returns (model, bob_body_index). The bob starts horizontal (offset +X by L) so it must swing 90
    degrees down to its straight-below equilibrium.
    """
    builder = newton.ModelBuilder()

    # Fixed (kinematic) anchor body located exactly at the pivot.
    anchor = builder.add_link(xform=wp.transform(_PIVOT, wp.quat_identity()))
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    # Bob: a near-point mass. A small nonzero inertia keeps the angular DOF well-conditioned so the
    # rigid link can rotate about the pivot (a truly zero inertia would freeze the swing).
    bob_start = wp.vec3(_PIVOT[0] + _CABLE_LENGTH, _PIVOT[1], _PIVOT[2])
    i_com = 1.0e-3
    bob = builder.add_link(
        xform=wp.transform(bob_start, wp.quat_identity()),
        mass=_BOB_MASS,
        inertia=wp.mat33(i_com, 0.0, 0.0, 0.0, i_com, 0.0, 0.0, 0.0, i_com),
    )

    # BALL joint at the pivot: the child anchor is offset by -L along the body +X, so satisfying the
    # constraint (child anchor == pivot) places the bob COM a fixed distance L from the pivot -- i.e.
    # the offset *is* the rigid, massless cable of length L.
    joint = builder.add_joint_ball(
        parent=anchor,
        child=bob,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-_CABLE_LENGTH, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])

    builder.body_flags[anchor] = int(newton.BodyFlags.KINEMATIC)
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, bob


def _build_spring_pendulum(device, spring_ke):
    """Particle bob on a massless (stiff) spring from a pinned particle at the pivot.

    Returns (model, bob_particle_index). The spring rest length is the initial distance L.
    """
    builder = newton.ModelBuilder()
    # Pinned anchor particle (mass 0) at the pivot.
    anchor = builder.add_particle(pos=_PIVOT, vel=wp.vec3(0.0), mass=0.0)
    # Bob particle offset +X by L (starts horizontal).
    bob_start = wp.vec3(_PIVOT[0] + _CABLE_LENGTH, _PIVOT[1], _PIVOT[2])
    bob = builder.add_particle(pos=bob_start, vel=wp.vec3(0.0), mass=_BOB_MASS)
    # Massless spring == the cable; rest length auto-set to the initial distance L.
    builder.add_spring(anchor, bob, ke=spring_ke, kd=0.0, control=0.0)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, bob


def _run_rigid(model, bob, iterations=80, num_steps=600, dt=1.0 / 60.0, chebyshev_rho=0.95):
    solver = SolverVBD(model, iterations=iterations, static_solve=True, rigid_chebyshev_rho=chebyshev_rho)
    state_0, state_1, control = model.state(), model.state(), model.control()
    prev = state_0.body_q.numpy()[bob, :3].copy()
    motion, positions = [], [prev]
    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        p = state_0.body_q.numpy()[bob, :3].copy()
        motion.append(float(np.linalg.norm(p - prev)))
        prev = p
        positions.append(p)
    return np.array(motion), positions, float(np.max(np.abs(state_0.body_qd.numpy()[bob])))


def _run_spring(model, bob, iterations=80, num_steps=600, dt=1.0 / 60.0, chebyshev_rho=0.95):
    solver = SolverVBD(model, iterations=iterations, static_solve=True, particle_chebyshev_rho=chebyshev_rho)
    state_0, state_1, control = model.state(), model.state(), model.control()
    prev = state_0.particle_q.numpy()[bob].copy()
    motion, positions = [], [prev]
    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        p = state_0.particle_q.numpy()[bob].copy()
        motion.append(float(np.linalg.norm(p - prev)))
        prev = p
        positions.append(p)
    return np.array(motion), positions, float(np.max(np.abs(state_0.particle_qd.numpy()[bob])))


def _conv_step(motion, threshold):
    below = np.where(motion < threshold)[0]
    return int(below[0]) + 1 if len(below) else None


def _report(name, motion, positions, max_v):
    bob = positions[-1]
    pivot = np.array([_PIVOT[0], _PIVOT[1], _PIVOT[2]])
    length = float(np.linalg.norm(bob - pivot))
    # Equilibrium check: hanging straight below pivot (x,y == pivot, z below).
    horiz_err = float(np.hypot(bob[0] - pivot[0], bob[1] - pivot[1]))
    return (
        f"{name:22s}: conv@1e-3={_conv_step(motion, 1e-3)}  conv@1e-4={_conv_step(motion, 1e-4)}  "
        f"mot_end={motion[-1]:.2e}  bob=({bob[0]:+.3f},{bob[2]:+.3f})  "
        f"len={length:.4f} (L={_CABLE_LENGTH})  horiz_err={horiz_err:.1e}  max|v|={max_v:.0e}"
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    device = wp.get_preferred_device()
    with wp.ScopedDevice(device):
        model_r, bob_r = _build_rigid_pendulum(device)
        rigid = _run_rigid(model_r, bob_r, iterations=args.iterations, num_steps=args.steps)
        print("rigid pendulum (BALL joint = inextensible massless cable):")
        print("  " + _report("rigid", *rigid))

        print("\nparticle pendulum (stiff massless spring; stretch ~ m*g/ke):")
        for ke in (1.0e3, 1.0e4, 1.0e5, 1.0e6):
            model_s, bob_s = _build_spring_pendulum(device, spring_ke=ke)
            spring = _run_spring(model_s, bob_s, iterations=args.iterations, num_steps=args.steps)
            print("  " + _report(f"spring ke={ke:.0e}", *spring))


if __name__ == "__main__":
    _main()
