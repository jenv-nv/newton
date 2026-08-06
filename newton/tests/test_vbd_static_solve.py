# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static-equilibrium test for the VBD solver.

Drapes a cloth grid that is pinned along one edge under gravity and solves it with
``SolverVBD(static_solve=True)``. Because the momentum term is removed from the energy,
each ``step`` is a quasi-static Newton solve, so the sheet settles monotonically into its
static equilibrium (a catenary-like drape) instead of swinging.

Run directly to also render an inspectable PNG:

    python3 newton/tests/test_vbd_static_solve.py --output /tmp/vbd_static_equilibrium.png
"""

import argparse
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.solver_vbd import SolverVBD


def _build_cloth_model(device):
    """Pinned cloth grid lying in the XY plane, gravity along -Z."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=16,
        dim_y=16,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        fix_left=True,  # pin one edge so the rest can drape
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,  # no rate damping: keep the static solve dt-independent
        edge_ke=1.0e0,
        edge_kd=0.0,
        particle_radius=0.01,
    )
    builder.color(include_bending=True)  # required by SolverVBD before finalize
    return builder.finalize(device=device)


def _run_solve(
    model, static_solve, iterations=60, num_steps=120, dt=1.0 / 60.0, chebyshev_rho=0.0, chebyshev_delay=5
):
    """Step the VBD solver; return (positions_history, per_step_motion, final_velocity).

    With ``static_solve=True`` each step is a quasi-static Newton solve that relaxes toward
    equilibrium; with ``static_solve=False`` it is ordinary implicit-Euler dynamics. ``chebyshev_rho > 0``
    enables Chebyshev acceleration of the VBD sweeps.
    """
    solver = SolverVBD(
        model,
        iterations=iterations,
        static_solve=static_solve,
        particle_chebyshev_rho=chebyshev_rho,
        particle_chebyshev_delay=chebyshev_delay,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    positions = [state_0.particle_q.numpy().copy()]
    motion = []
    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        q = state_0.particle_q.numpy().copy()
        motion.append(float(np.max(np.linalg.norm(q - positions[-1], axis=1))))
        positions.append(q)

    final_vel = state_0.particle_qd.numpy()
    return positions, motion, final_vel


def _run_static_solve(model, **kwargs):
    return _run_solve(model, static_solve=True, **kwargs)


def _free_mean_height(model, positions):
    """Mean height (up-axis, z) of the non-pinned particles at each step."""
    pinned = model.particle_inv_mass.numpy() == 0.0
    return np.array([p[~pinned, 2].mean() for p in positions])


def _error_curve(positions, q_ref):
    """Max nodal distance from each step's positions to the reference equilibrium."""
    return np.array([float(np.max(np.linalg.norm(p - q_ref, axis=1))) for p in positions])


def _render_chebyshev(model, q_ref, runs, iters, output_path):
    """Render error-to-equilibrium vs Gauss-Seidel sweeps for several Chebyshev rho values.

    ``runs`` is a list of ``(label, color, positions)`` tuples sharing the reference ``q_ref``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pinned = model.particle_inv_mass.numpy() == 0.0

    fig = plt.figure(figsize=(12, 5))

    # Convergence: error to equilibrium vs cumulative sweeps (step index * iters/step).
    ax = fig.add_subplot(1, 2, 1)
    for label, color, positions in runs:
        err = _error_curve(positions, q_ref)
        sweeps = np.arange(1, len(err)) * iters
        ax.semilogy(sweeps, err[1:], color=color, marker=".", ms=3, label=label)
    ax.set_title("Chebyshev acceleration of the VBD static solve")
    ax.set_xlabel("Gauss-Seidel sweeps")
    ax.set_ylabel("max error to equilibrium [m]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # The equilibrium all runs converge to (rendered from the reference). A pinned, near-bendless
    # sheet hangs straight down as a flat vertical curtain (x ~ 0), so use equal axis scaling to show
    # it truthfully rather than letting the degenerate x-axis auto-zoom into floating-point noise.
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(q_ref[~pinned, 0], q_ref[~pinned, 1], q_ref[~pinned, 2], s=8, c="tab:blue", label="equilibrium")
    ax2.scatter(q_ref[pinned, 0], q_ref[pinned, 1], q_ref[pinned, 2], s=20, c="tab:red", label="pinned edge")
    span = 0.5 * max(np.ptp(q_ref[:, 0]), np.ptp(q_ref[:, 1]), np.ptp(q_ref[:, 2]))
    centers = q_ref.min(axis=0) + 0.5 * np.ptp(q_ref, axis=0)
    ax2.set_xlim(centers[0] - span, centers[0] + span)
    ax2.set_ylim(centers[1] - span, centers[1] + span)
    ax2.set_zlim(centers[2] - span, centers[2] + span)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_title("Converged static equilibrium (vertical curtain)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def _render(model, static, dynamic, output_path):
    """Render static equilibrium alongside a dynamic (momentum-carrying) run for contrast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_s, motion_s, _ = static
    pos_d, motion_d, _ = dynamic

    q0 = pos_s[0]
    qN = pos_s[-1]
    pinned = model.particle_inv_mass.numpy() == 0.0

    h_s = _free_mean_height(model, pos_s)
    h_d = _free_mean_height(model, pos_d)

    fig = plt.figure(figsize=(16, 5))

    # Panel 1: static equilibrium shape.
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(q0[:, 0], q0[:, 1], q0[:, 2], s=8, c="0.75", label="initial (flat)")
    ax.scatter(qN[~pinned, 0], qN[~pinned, 1], qN[~pinned, 2], s=8, c="tab:blue", label="static equilibrium")
    ax.scatter(qN[pinned, 0], qN[pinned, 1], qN[pinned, 2], s=20, c="tab:red", label="pinned edge")
    ax.set_title("Static equilibrium (pinned cloth under gravity)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: free-edge mean height — static settles, dynamic oscillates about it.
    steps = range(len(h_s))
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, h_d, c="tab:orange", label="dynamic (momentum)")
    ax2.plot(steps, h_s, c="tab:blue", label="static (equilibrium)")
    ax2.axhline(h_s[-1], color="tab:blue", ls="--", lw=0.8, alpha=0.6)
    ax2.set_title("Mean height of free particles")
    ax2.set_xlabel("step")
    ax2.set_ylabel("mean z of free particles [m]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: per-step motion — static decays, dynamic sustains (undamped swinging).
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(range(1, len(motion_d) + 1), motion_d, c="tab:orange", marker=".", ms=3, label="dynamic")
    ax3.semilogy(range(1, len(motion_s) + 1), motion_s, c="tab:blue", marker=".", ms=3, label="static")
    ax3.set_title("Per-step vertex motion")
    ax3.set_xlabel("step")
    ax3.set_ylabel("max per-step vertex motion [m]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


class TestVBDStaticSolve(unittest.TestCase):
    def test_static_equilibrium(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model = _build_cloth_model(device)
            positions, motion, final_vel = _run_static_solve(model)

        pinned = model.particle_inv_mass.numpy() == 0.0
        q0 = positions[0]
        qN = positions[-1]

        # 1. The solve is converging to a static equilibrium: warm-started steps relax the same
        #    (elastic + gravity) energy, so per-step motion decays and the shape stabilizes.
        motion = np.asarray(motion)
        self.assertLess(motion[-1], motion[0] * 0.3, "static solve is not converging")
        self.assertLess(motion[-1], 1e-2, "static solve has not settled to a stable shape")
        # Robust downward trend: the last quarter moves much less than the first quarter.
        quarter = len(motion) // 4
        self.assertLess(
            motion[-quarter:].mean(),
            motion[:quarter].mean() * 0.5,
            "per-step motion is not trending down",
        )

        # 2. Static solve reports zero velocity (momentum removed).
        self.assertLess(float(np.max(np.abs(final_vel))), 1e-6, "static solve produced nonzero velocity")

        # 3. The free part of the sheet actually draped below the pinned edge.
        pinned_z = float(np.mean(q0[pinned, 2]))
        self.assertLess(
            float(np.min(qN[~pinned, 2])),
            pinned_z - 0.1,
            "cloth did not drape under gravity",
        )

        # 4. Pinned particles stayed put.
        self.assertLess(
            float(np.max(np.linalg.norm(qN[pinned] - q0[pinned], axis=1))),
            1e-6,
            "pinned particles moved",
        )

    def test_contrast_with_dynamics(self):
        """The same undamped setup keeps swinging under dynamics but settles when static."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            _, motion_s, _ = _run_solve(_build_cloth_model(device), static_solve=True)
            _, motion_d, _ = _run_solve(_build_cloth_model(device), static_solve=False)

        motion_s = np.asarray(motion_s)
        motion_d = np.asarray(motion_d)
        tail = len(motion_s) // 4

        # Static settles; undamped dynamics does not.
        self.assertLess(motion_s[-tail:].mean(), motion_d[-tail:].mean() * 0.3, "static did not settle relative to dynamics")
        self.assertGreater(
            motion_d[-tail:].mean(),
            motion_d[:tail].mean() * 0.3,
            "undamped dynamics unexpectedly damped out",
        )

    def test_chebyshev_accelerates(self):
        """Chebyshev reaches the static equilibrium in far fewer sweeps than plain Gauss-Seidel."""
        rho = 0.99
        iters, steps = 20, 150  # equal 3000-sweep budget for the plain vs accelerated comparison
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            # Well-converged reference equilibrium (Chebyshev converges quickly).
            ref, _, _ = _run_solve(
                _build_cloth_model(device), static_solve=True, iterations=30, num_steps=300, chebyshev_rho=rho
            )
            q_ref = ref[-1]
            plain, _, _ = _run_solve(
                _build_cloth_model(device), static_solve=True, iterations=iters, num_steps=steps, chebyshev_rho=0.0
            )
            cheb, _, _ = _run_solve(
                _build_cloth_model(device), static_solve=True, iterations=iters, num_steps=steps, chebyshev_rho=rho
            )

        err_plain = float(np.max(np.linalg.norm(plain[-1] - q_ref, axis=1)))
        err_cheb = float(np.max(np.linalg.norm(cheb[-1] - q_ref, axis=1)))

        self.assertTrue(np.all(np.isfinite(cheb[-1])), "Chebyshev run diverged")
        # At an equal sweep budget, acceleration is far closer to equilibrium.
        self.assertLess(err_cheb, 0.3, "Chebyshev did not approach equilibrium")
        self.assertLess(err_cheb, err_plain * 0.5, "Chebyshev did not accelerate convergence")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="vbd_static_equilibrium.png", help="Path for the static/dynamic PNG")
    parser.add_argument(
        "--chebyshev-output",
        default="vbd_chebyshev_convergence.png",
        help="Path for the Chebyshev convergence PNG",
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    device = wp.get_preferred_device()
    with wp.ScopedDevice(device):
        static = _run_solve(
            _build_cloth_model(device), static_solve=True, iterations=args.iterations, num_steps=args.steps
        )
        # Same setup, momentum retained: ordinary dynamics for contrast.
        dynamic = _run_solve(
            _build_cloth_model(device), static_solve=False, iterations=args.iterations, num_steps=args.steps
        )
        model = _build_cloth_model(device)  # for pinned mask / labels only

        # Chebyshev acceleration sweep: error to equilibrium vs sweeps for several rho.
        ref, _, _ = _run_solve(
            _build_cloth_model(device), static_solve=True, iterations=args.iterations, num_steps=args.steps, chebyshev_rho=0.99
        )
        q_ref = ref[-1]
        cheb_runs = []
        for rho, color in [(0.0, "tab:gray"), (0.9, "tab:orange"), (0.99, "tab:green")]:
            pos, _, _ = _run_solve(
                _build_cloth_model(device),
                static_solve=True,
                iterations=args.iterations,
                num_steps=args.steps,
                chebyshev_rho=rho,
            )
            label = f"rho={rho}" + (" (plain GS)" if rho == 0.0 else "")
            cheb_runs.append((label, color, pos))

    _, motion_s, vel_s = static
    _, motion_d, _ = dynamic
    print(f"device: {device}")
    print(f"particles: {model.particle_count}")
    print(f"static  : per-step motion {motion_s[0]:.3e} -> {motion_s[-1]:.3e} m,  max|v| {np.max(np.abs(vel_s)):.3e} m/s")
    print(f"dynamic : per-step motion {motion_d[0]:.3e} -> {motion_d[-1]:.3e} m  (sustained, undamped swinging)")
    for label, _, pos in cheb_runs:
        print(f"chebyshev {label:>16}: err to equilibrium {_error_curve(pos, q_ref)[-1]:.3e} m at {args.iterations * args.steps} sweeps")
    path = _render(model, static, dynamic, args.output)
    print(f"rendered: {path}")
    cheb_path = _render_chebyshev(model, q_ref, cheb_runs, args.iterations, args.chebyshev_output)
    print(f"rendered: {cheb_path}")


if __name__ == "__main__":
    _main()
