# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static-equilibrium test for the VBD solver with a cable (rod chain).

Hangs a cable that is pinned at one end under gravity and solves it with
``SolverVBD(static_solve=True)``. Because the momentum term is removed from the energy, each ``step``
is a quasi-static Newton solve, so the chain settles monotonically into a catenary-like drape instead
of swinging. ``rigid_chebyshev_rho`` accelerates the AVBD sweeps, which is what makes a long joint
chain (a slow global bending mode) reach equilibrium in a practical number of sweeps.

Run directly to also render an inspectable PNG:

    python3 newton/tests/test_vbd_cable_static_solve.py --output /tmp/vbd_cable_static_equilibrium.png
"""

import argparse
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.solver_vbd import SolverVBD

_NUM_ELEMENTS = 8
_SEGMENT_LENGTH = 0.2
_Z_HEIGHT = 3.0
_BEND_STIFFNESS = 50.0
_CHEBYSHEV_RHO = 0.95


def _make_straight_cable_along_x(num_elements, segment_length, z_height):
    """Straight cable along +X, centered at the origin, at the given Z height."""
    length = float(num_elements * segment_length)
    start = wp.vec3(-0.5 * length, 0.0, float(z_height))
    return newton.utils.create_straight_cable_points_and_quaternions(
        start=start,
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=length,
        num_segments=int(num_elements),
    )


def _build_cable_model(device):
    """Cable pinned at its left end, initially horizontal, gravity along -Z."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0

    points, edge_q = _make_straight_cable_along_x(_NUM_ELEMENTS, _SEGMENT_LENGTH, _Z_HEIGHT)

    rod_bodies, _ = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=0.02,
        bend_stiffness=_BEND_STIFFNESS,
        bend_damping=0.0,  # no rate damping: keep the static solve dt-independent
        label="test_cable_static",
        body_frame_origin="com",
    )

    builder.body_flags[rod_bodies[0]] = int(newton.BodyFlags.KINEMATIC)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, rod_bodies


def _run_solve(model, rod_bodies, static_solve, iterations=80, num_steps=150, dt=1.0 / 60.0, chebyshev_rho=0.0):
    """Step the VBD solver; return (positions_history, per_step_motion, final_body_velocity).

    ``positions_history`` is a list of ``[num_cable_bodies, 3]`` COM position arrays.
    ``per_step_motion`` is a list of max per-step COM displacements.
    ``final_body_velocity`` is the spatial velocity array ``[num_cable_bodies, 6]`` at the last step.

    With ``static_solve=True`` each step is a quasi-static Newton solve that relaxes toward
    equilibrium; with ``static_solve=False`` it is ordinary implicit-Euler dynamics. ``chebyshev_rho > 0``
    enables Chebyshev acceleration of the rigid AVBD sweeps.
    """
    solver = SolverVBD(
        model,
        iterations=iterations,
        static_solve=static_solve,
        rigid_chebyshev_rho=chebyshev_rho,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    body_ids = np.array(rod_bodies)
    positions = [state_0.body_q.numpy()[body_ids, :3].copy()]
    motion = []

    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        q = state_0.body_q.numpy()[body_ids, :3].copy()
        motion.append(float(np.max(np.linalg.norm(q - positions[-1], axis=1))))
        positions.append(q)

    final_vel = state_0.body_qd.numpy()[body_ids]
    return positions, motion, final_vel


def _run_static_solve(model, rod_bodies, **kwargs):
    return _run_solve(model, rod_bodies, static_solve=True, chebyshev_rho=_CHEBYSHEV_RHO, **kwargs)


def _error_curve(positions, q_ref):
    """Max nodal distance from each step's positions to the reference equilibrium."""
    return np.array([float(np.max(np.linalg.norm(p - q_ref, axis=1))) for p in positions])


def _render(rod_bodies, static, dynamic, output_path):
    """Render static equilibrium alongside a dynamic run for contrast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_s, motion_s, _ = static
    pos_d, motion_d, _ = dynamic

    q0 = pos_s[0]
    qN = pos_s[-1]

    free_z_s = [p[1:, 2].mean() for p in pos_s]
    free_z_d = [p[1:, 2].mean() for p in pos_d]

    fig = plt.figure(figsize=(15, 5))

    # Panel 1: cable shape at equilibrium.
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(q0[:, 0], q0[:, 2], "o-", c="0.75", ms=4, label="initial (horizontal)")
    ax.plot(qN[:, 0], qN[:, 2], "o-", c="tab:blue", ms=4, label="static equilibrium")
    ax.plot(qN[0, 0], qN[0, 2], "o", c="tab:red", ms=10, label="pinned anchor")
    ax.set_title("Static equilibrium (cable under gravity)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel 2: mean free-body height — static settles, dynamic oscillates.
    steps = range(len(free_z_s))
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, free_z_d, c="tab:orange", label="dynamic (momentum)")
    ax2.plot(steps, free_z_s, c="tab:blue", label="static (equilibrium)")
    ax2.axhline(free_z_s[-1], color="tab:blue", ls="--", lw=0.8, alpha=0.6)
    ax2.set_title("Mean height of free bodies")
    ax2.set_xlabel("step")
    ax2.set_ylabel("mean z [m]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: per-step motion — static decays, dynamic sustains.
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(range(1, len(motion_d) + 1), motion_d, c="tab:orange", marker=".", ms=3, label="dynamic")
    ax3.semilogy(range(1, len(motion_s) + 1), motion_s, c="tab:blue", marker=".", ms=3, label="static (Chebyshev)")
    ax3.set_title("Per-step body motion")
    ax3.set_xlabel("step")
    ax3.set_ylabel("max per-step COM motion [m]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


class TestVBDCableStaticSolve(unittest.TestCase):
    def test_static_equilibrium(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model, rod_bodies = _build_cable_model(device)
            positions, motion, final_vel = _run_static_solve(model, rod_bodies)

        q0 = positions[0]
        qN = positions[-1]

        # 1. The solve is converging to a static equilibrium: warm-started steps relax the same
        #    (bending + gravity) energy, so per-step motion decays and the shape stabilizes.
        motion = np.asarray(motion)
        self.assertLess(motion[-1], motion[0] * 0.3, "static solve is not converging")
        self.assertLess(motion[-1], 3e-3, "static solve has not settled to a stable shape")
        # Robust downward trend: the last quarter moves much less than the first quarter.
        quarter = len(motion) // 4
        self.assertLess(
            motion[-quarter:].mean(),
            motion[:quarter].mean() * 0.5,
            "per-step motion is not trending down",
        )

        # 2. Static solve reports zero velocity (momentum removed).
        self.assertLess(float(np.max(np.abs(final_vel))), 1e-6, "static solve produced nonzero velocity")

        # 3. The free part of the cable drooped below the anchor.
        anchor_z = float(q0[0, 2])
        self.assertLess(
            float(np.min(qN[1:, 2])),
            anchor_z - 0.1,
            "cable did not drape under gravity",
        )

        # 4. The pinned body stayed put.
        self.assertLess(
            float(np.linalg.norm(qN[0] - q0[0])),
            1e-6,
            "pinned body moved",
        )

    def test_contrast_with_dynamics(self):
        """The same undamped setup keeps swinging under dynamics but settles when static."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model_s, rods_s = _build_cable_model(device)
            _, motion_s, _ = _run_solve(model_s, rods_s, static_solve=True, chebyshev_rho=_CHEBYSHEV_RHO)
            model_d, rods_d = _build_cable_model(device)
            _, motion_d, _ = _run_solve(model_d, rods_d, static_solve=False)

        motion_s = np.asarray(motion_s)
        motion_d = np.asarray(motion_d)
        tail = len(motion_s) // 4

        # Static settles; undamped dynamics does not.
        self.assertLess(
            motion_s[-tail:].mean(),
            motion_d[-tail:].mean() * 0.3,
            "static did not settle relative to dynamics",
        )
        self.assertGreater(
            motion_d[-tail:].mean(),
            motion_d[:tail].mean() * 0.3,
            "undamped dynamics unexpectedly damped out",
        )

    def test_chebyshev_accelerates(self):
        """Chebyshev reaches the static equilibrium in far fewer sweeps than plain Gauss-Seidel."""
        rho = _CHEBYSHEV_RHO
        iters, steps = 40, 150  # equal 6000-sweep budget for the plain vs accelerated comparison
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            # Well-converged reference equilibrium (Chebyshev converges quickly).
            model_ref, rods_ref = _build_cable_model(device)
            ref, _, _ = _run_solve(
                model_ref, rods_ref, static_solve=True, iterations=80, num_steps=400, chebyshev_rho=rho
            )
            q_ref = ref[-1]

            model_p, rods_p = _build_cable_model(device)
            plain, _, _ = _run_solve(
                model_p, rods_p, static_solve=True, iterations=iters, num_steps=steps, chebyshev_rho=0.0
            )
            model_c, rods_c = _build_cable_model(device)
            cheb, _, _ = _run_solve(
                model_c, rods_c, static_solve=True, iterations=iters, num_steps=steps, chebyshev_rho=rho
            )

        err_plain = float(np.max(np.linalg.norm(plain[-1] - q_ref, axis=1)))
        err_cheb = float(np.max(np.linalg.norm(cheb[-1] - q_ref, axis=1)))

        self.assertTrue(np.all(np.isfinite(cheb[-1])), "Chebyshev run diverged")
        # At an equal sweep budget, acceleration is much closer to equilibrium.
        self.assertLess(err_cheb, err_plain * 0.6, "Chebyshev did not accelerate convergence")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="vbd_cable_static_equilibrium.png", help="Path for the output PNG")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    device = wp.get_preferred_device()
    with wp.ScopedDevice(device):
        model, rod_bodies = _build_cable_model(device)
        static = _run_solve(
            model,
            rod_bodies,
            static_solve=True,
            iterations=args.iterations,
            num_steps=args.steps,
            chebyshev_rho=_CHEBYSHEV_RHO,
        )
        model_d, rod_bodies_d = _build_cable_model(device)
        dynamic = _run_solve(
            model_d, rod_bodies_d, static_solve=False, iterations=args.iterations, num_steps=args.steps
        )

    _, motion_s, vel_s = static
    _, motion_d, _ = dynamic

    print(f"device: {device}")
    print(f"cable bodies: {len(rod_bodies)}")
    print(f"static  : per-step motion {motion_s[0]:.3e} -> {motion_s[-1]:.3e} m,  max|v| {np.max(np.abs(vel_s)):.3e}")
    print(f"dynamic : per-step motion {motion_d[0]:.3e} -> {motion_d[-1]:.3e} m  (sustained, undamped swinging)")

    path = _render(rod_bodies, static, dynamic, args.output)
    print(f"rendered: {path}")


if __name__ == "__main__":
    _main()
