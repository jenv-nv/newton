# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aerodynamic-drag tests for ``SolverVBD`` on the rod (``add_rod``) path.

The solver gained two viscous-drag coefficients, ``drag_linear`` [N·s/m] and
``drag_angular`` [N·m·s/rad], which add a wrench opposing each body's *absolute*
(world-frame) velocity in the forward step:

    f_drag = (-drag_linear · v,  -drag_angular · ω).

Unlike the rod's ``bend_damping`` / ``twist_damping`` (which act on the
*relative* velocity across a joint, i.e. shape change), this dissipates *bulk*
motion of the whole rod — the regime air resistance lives in. These tests pin
down the properties a caller relies on:

  * zero drag is inert (the added code path is skipped, results unchanged);
  * linear drag monotonically decays a free rod's translation, faster for a
    larger coefficient;
  * angular drag decays a free body's spin;
  * drag dissipates a swinging pinned rod's kinetic energy;
  * a static solve (velocity removed from the energy) is unaffected by drag.

Run directly for a quick textual summary:

    python3 newton/tests/test_vbd_rod_drag.py
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.solver_vbd import SolverVBD

_NUM_ELEMENTS = 6
_SEGMENT_LENGTH = 0.2
_RADIUS = 0.02
_BEND_STIFFNESS = 50.0
_DT = 1.0 / 60.0


def _straight_rod(num_elements, segment_length):
    """Straight rod along +X, centered at the origin at z = 0."""
    length = float(num_elements * segment_length)
    return newton.utils.create_straight_cable_points_and_quaternions(
        start=wp.vec3(-0.5 * length, 0.0, 0.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=length,
        num_segments=int(num_elements),
    )


def _build_rod(device, pin=False, gravity=(0.0, 0.0, 0.0), bend_damping=0.0):
    """A rod chain built with ``add_rod``; optionally pin body 0 (kinematic)."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0

    points, edge_q = _straight_rod(_NUM_ELEMENTS, _SEGMENT_LENGTH)
    rod_bodies, _ = builder.add_rod(
        positions=points,
        quaternions=edge_q,
        radius=_RADIUS,
        bend_stiffness=_BEND_STIFFNESS,
        bend_damping=bend_damping,
        label="drag_rod",
        body_frame_origin="com",
    )
    if pin:
        builder.body_flags[rod_bodies[0]] = int(newton.BodyFlags.KINEMATIC)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity(gravity)
    return model, rod_bodies


def _build_free_body(device, inertia_scalar=1.0e-3, mass=0.2):
    """A single free-floating rigid body (isolates the angular-drag term)."""
    builder = newton.ModelBuilder()
    body = builder.add_body(
        mass=mass,
        inertia=wp.mat33(inertia_scalar, 0.0, 0.0, 0.0, inertia_scalar, 0.0, 0.0, 0.0, inertia_scalar),
        label="free_body",
    )
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model, [body]


def _run(model, body_ids, num_steps, drag_linear=0.0, drag_angular=0.0,
         static_solve=False, init_qd=None, iterations=30, chebyshev_rho=0.0, dt=_DT):
    """Step SolverVBD with the given drag; return per-step velocity/position history.

    Returns ``(lin_speed, ang_speed, positions)`` where ``lin_speed`` /
    ``ang_speed`` are ``[num_steps + 1]`` mean-over-bodies speed arrays and
    ``positions`` is a list of ``[nbodies, 3]`` COM positions.
    """
    solver = SolverVBD(
        model,
        iterations=iterations,
        static_solve=static_solve,
        rigid_chebyshev_rho=chebyshev_rho,
        drag_linear=drag_linear,
        drag_angular=drag_angular,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    ids = np.asarray(body_ids)

    if init_qd is not None:
        qd = state_0.body_qd.numpy()
        qd[ids] = np.asarray(init_qd, dtype=np.float32)
        state_0.body_qd = wp.array(qd, dtype=wp.spatial_vector, device=model.device)

    def _speeds(state):
        qd = state.body_qd.numpy()[ids]
        lin = float(np.mean(np.linalg.norm(qd[:, 0:3], axis=1)))  # [linear | angular] halves
        ang = float(np.mean(np.linalg.norm(qd[:, 3:6], axis=1)))
        return lin, ang

    lin0, ang0 = _speeds(state_0)
    lin_hist, ang_hist = [lin0], [ang0]
    pos_hist = [state_0.body_q.numpy()[ids, :3].copy()]

    for _ in range(num_steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        lin, ang = _speeds(state_0)
        lin_hist.append(lin)
        ang_hist.append(ang)
        pos_hist.append(state_0.body_q.numpy()[ids, :3].copy())

    return np.asarray(lin_hist), np.asarray(ang_hist), pos_hist


class TestVBDRodDrag(unittest.TestCase):
    def test_zero_drag_is_inert(self):
        """drag=0 skips the drag branch entirely, so it must not perturb the sim.

        Compares a run built with explicit ``drag_linear=drag_angular=0`` against
        one built without the arguments at all (defaults): identical trajectory.
        A nonzero-drag run of the same setup must differ — proving the knob is live.
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            init = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            model_a, ids_a = _build_rod(device)
            base_default = _run(model_a, ids_a, 30, init_qd=init)

            model_b, ids_b = _build_rod(device)
            base_zero = _run(model_b, ids_b, 30, drag_linear=0.0, drag_angular=0.0, init_qd=init)

            model_c, ids_c = _build_rod(device)
            dragged = _run(model_c, ids_c, 30, drag_linear=2.0, init_qd=init)

        # Explicit zero drag == default (no drag args): the branch is genuinely skipped.
        np.testing.assert_allclose(base_default[0], base_zero[0], rtol=1e-6, atol=1e-8,
                                   err_msg="drag=0 changed the result vs the default path")
        # Drag actually does something.
        self.assertLess(dragged[0][-1], base_zero[0][-1] * 0.9,
                        "nonzero drag did not change the trajectory")

    def test_linear_drag_decays_free_translation(self):
        """A free rod coasting at constant velocity (no gravity) is slowed by linear drag."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            v0 = 0.5
            init = [v0, 0.0, 0.0, 0.0, 0.0, 0.0]

            model_b, ids_b = _build_rod(device)
            lin_base, _, _ = _run(model_b, ids_b, 60, init_qd=init)

            model_d, ids_d = _build_rod(device)
            # Size the coefficient off the body mass so the per-step decay is pronounced
            # regardless of the rod's density: the implicit factor per step is
            # 1 / (1 + c·inv_m·dt).
            inv_m = float(np.mean(model_d.body_inv_mass.numpy()[np.asarray(ids_d)]))
            c_lin = 5.0 / inv_m
            lin_drag, _, _ = _run(model_d, ids_d, 60, drag_linear=c_lin, init_qd=init)

        # Baseline: no force, so it coasts — final speed ≈ initial.
        self.assertAlmostEqual(lin_base[-1], v0, delta=0.05 * v0,
                               msg="undamped free rod did not coast at constant speed")
        # Drag: strong decay, and monotonically non-increasing (never speeds up / reverses).
        self.assertLess(lin_drag[-1], 0.2 * v0, "linear drag did not decay translation")
        self.assertTrue(np.all(np.diff(lin_drag) <= 1e-6),
                        "drag-damped speed is not monotonically decreasing")
        # Implicit drag can never overshoot: the decay factor 1/(1+c·inv_m·dt) is in (0,1].
        self.assertGreaterEqual(lin_drag.min(), -1e-9, "drag drove speed negative (overshoot)")

    def test_per_body_drag_decays_bodies_independently(self):
        """set_body_drag gives each body its own coefficient in a single batched solve.

        Two free bodies with no gravity and the same initial speed, one with strong
        drag and one with none: after stepping the same solver, the dragged body must
        be much slower and the drag-free body essentially unchanged. This is the
        capability a batched population fit relies on (each parallel world its own drag).
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            for _ in range(2):
                builder.add_body(mass=0.2, inertia=wp.mat33(1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3))
            builder.color()
            model = builder.finalize(device=device)
            model.set_gravity((0.0, 0.0, 0.0))

            solver = SolverVBD(model, iterations=30)
            inv_m = float(model.body_inv_mass.numpy()[0])
            # Body 0: strong drag; body 1: none.
            solver.set_body_drag(linear=[5.0 / inv_m, 0.0])

            state_0, state_1, control = model.state(), model.state(), model.control()
            qd = state_0.body_qd.numpy()
            qd[:, 0] = 0.5  # both bodies moving at 0.5 m/s in +x
            state_0.body_qd = wp.array(qd, dtype=wp.spatial_vector, device=device)
            for _ in range(40):
                solver.step(state_0, state_1, control, None, _DT)
                state_0, state_1 = state_1, state_0
            v = state_0.body_qd.numpy()[:, 0]

        self.assertLess(float(v[0]), 0.2 * 0.5, "dragged body did not slow down")
        self.assertAlmostEqual(float(v[1]), 0.5, delta=0.05, msg="drag-free body was affected")

    def test_implicit_drag_stable_at_huge_coefficient(self):
        """A coefficient far past the explicit stability limit stays bounded and decays.

        Explicit drag has factor (1 - c·inv_m·dt): for c·inv_m·dt > 2 it diverges and
        pumps energy. The implicit factor 1/(1 + c·inv_m·dt) is in (0,1] for *any* c, so
        the same setup must instead decay smoothly to rest. This is the whole reason for
        the implicit formulation.
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            v0 = 0.5
            init = [v0, 0.0, 0.0, 0.0, 0.0, 0.0]
            model_d, ids_d = _build_rod(device)
            inv_m = float(np.mean(model_d.body_inv_mass.numpy()[np.asarray(ids_d)]))
            c_huge = 1000.0 / inv_m  # c·inv_m·dt ≈ 16.7 — deep in the explicit-blowup regime
            lin, _, _ = _run(model_d, ids_d, 40, drag_linear=c_huge, init_qd=init)

        self.assertTrue(np.all(np.isfinite(lin)), "implicit drag diverged (NaN/inf)")
        self.assertLessEqual(float(lin.max()), v0 + 1e-6, "implicit drag pumped energy")
        self.assertTrue(np.all(np.diff(lin) <= 1e-6), "implicit drag was not monotone")
        self.assertLess(lin[-1], 1e-2 * v0, "huge implicit drag did not bring the rod to rest")

    def test_stronger_drag_decays_faster(self):
        """Doubling the linear coefficient leaves less residual speed after the same time."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            init = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            model_1, ids_1 = _build_rod(device)
            inv_m = float(np.mean(model_1.body_inv_mass.numpy()[np.asarray(ids_1)]))
            c = 2.0 / inv_m
            weak, _, _ = _run(model_1, ids_1, 40, drag_linear=c, init_qd=init)

            model_2, ids_2 = _build_rod(device)
            strong, _, _ = _run(model_2, ids_2, 40, drag_linear=2.0 * c, init_qd=init)

        self.assertLess(strong[-1], weak[-1], "larger drag coefficient did not decay faster")

    def test_angular_drag_decays_spin(self):
        """A free body spun about a principal axis (no gravity) is slowed by angular drag."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            w0 = 4.0
            init = [0.0, 0.0, 0.0, 0.0, w0, 0.0]  # ω about +Y

            model_b, ids_b = _build_free_body(device)
            _, ang_base, _ = _run(model_b, ids_b, 40, init_qd=init)

            model_d, ids_d = _build_free_body(device)
            inv_I = float(model_d.body_inv_inertia.numpy()[ids_d[0]][1, 1])
            c_ang = 5.0 / inv_I
            _, ang_drag, _ = _run(model_d, ids_d, 40, drag_angular=c_ang, init_qd=init)

        # Undamped: spin is conserved. Damped: it decays hard and never reverses.
        self.assertAlmostEqual(ang_base[-1], w0, delta=0.1 * w0,
                               msg="undamped free body did not conserve spin")
        self.assertLess(ang_drag[-1], 0.2 * w0, "angular drag did not decay spin")
        self.assertTrue(np.all(np.diff(ang_drag) <= 1e-6),
                        "drag-damped spin is not monotonically decreasing")

    def _swing_path(self, drag_linear=0.0, drag_angular=0.0, steps=180):
        """Total path a horizontally-released pinned rod travels under gravity.

        The metric is the summed per-step COM displacement over the whole run.
        That integrates over every swing phase, so it is immune to where in an
        oscillation the run happens to end — draining energy strictly shortens
        the path. Also returns the undamped tail speed for a "still swinging"
        sanity check.
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model, ids = _build_rod(device, pin=True, gravity=(0.0, 0.0, -9.81))
            lin, _, pos = _run(model, ids, steps, drag_linear=drag_linear,
                               drag_angular=drag_angular, iterations=40)
        path = float(np.sum([np.mean(np.linalg.norm(pos[t] - pos[t - 1], axis=1))
                             for t in range(1, len(pos))]))
        tail = lin[-len(lin) // 4:].mean()
        return path, tail

    def _rod_drag_coeff(self, rate, angular):
        """A drag coefficient giving a per-step decay of ~``rate·dt`` for the rod's
        bodies — sized off inverse inertia if ``angular``, else off inverse mass, so
        the two channels decay at a comparable rate despite very different scales.
        (Implicit drag is stable for any coefficient; this just keeps the effect in a
        readable, physically-sensible range.)
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model, ids = _build_rod(device)
            ids = np.asarray(ids)
            if angular:
                inv = float(np.mean([np.trace(m) / 3.0 for m in model.body_inv_inertia.numpy()[ids]]))
            else:
                inv = float(np.mean(model.body_inv_mass.numpy()[ids]))
        return rate / inv

    def test_linear_drag_alone_dissipates_swing(self):
        """Linear drag by itself shortens the swing path of a pinned rod under gravity.

        This is the case drag exists for: bulk swinging that the rod's internal
        bend/twist damping cannot remove (it only sees relative joint motion).
        """
        rate = 0.1 / _DT  # ~10% decay per step
        path_base, tail_base = self._swing_path()
        path_drag, _ = self._swing_path(drag_linear=self._rod_drag_coeff(rate, angular=False))

        self.assertLess(path_drag, path_base * 0.85,
                        "linear drag alone did not shorten the swing path")
        self.assertGreater(tail_base, 1e-3, "undamped rod unexpectedly settled")

    def test_angular_drag_alone_decays_rod_spin(self):
        """Angular drag by itself decays a free rod spinning about its own axis.

        The rod lies along +X, so a spin ω about X gives every body ``v = ω×r = 0``
        (all COMs are on the axis) and zero *relative* twist between neighbours —
        a rigid, joint-unstrained rotation with no linear motion. That isolates the
        angular drag channel on the rod path: only ``drag_angular`` can slow it.
        """
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            w0 = 4.0
            init = [0.0, 0.0, 0.0, w0, 0.0, 0.0]  # spin about +X (the rod axis)

            model_b, ids_b = _build_rod(device)
            _, ang_base, _ = _run(model_b, ids_b, 60, init_qd=init)

            model_d, ids_d = _build_rod(device)
            inv_I = float(np.mean([np.trace(m) / 3.0
                                   for m in model_d.body_inv_inertia.numpy()[np.asarray(ids_d)]]))
            c_ang = 5.0 / inv_I
            _, ang_drag, _ = _run(model_d, ids_d, 60, drag_angular=c_ang, init_qd=init)

        # Undamped: the rigid spin is conserved. Damped: it decays hard, monotonically.
        self.assertAlmostEqual(ang_base[-1], w0, delta=0.1 * w0,
                               msg="undamped rod did not conserve its axial spin")
        self.assertLess(ang_drag[-1], 0.2 * w0, "angular drag did not decay the rod spin")
        # Twist-joint coupling puts tiny ripples on the mean spin, so don't require strict
        # per-step monotonicity here (the single-body test does). What must hold is that
        # drag never injects energy: the spin never climbs above where it started.
        self.assertLessEqual(float(ang_drag.max()), w0 + 1e-6,
                             "angular drag pumped the rod spin above its initial value")

    def test_static_solve_ignores_drag(self):
        """Under static_solve the velocity is zeroed, so drag can contribute nothing."""
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            model_b, ids_b = _build_rod(device, pin=True, gravity=(0.0, 0.0, -9.81))
            _, _, pos_base = _run(model_b, ids_b, 120, static_solve=True, chebyshev_rho=0.95, iterations=40)

            model_d, ids_d = _build_rod(device, pin=True, gravity=(0.0, 0.0, -9.81))
            _, _, pos_drag = _run(model_d, ids_d, 120, static_solve=True, chebyshev_rho=0.95,
                                  drag_linear=1.0e3, drag_angular=1.0e3, iterations=40)

        max_diff = float(np.max(np.linalg.norm(pos_base[-1] - pos_drag[-1], axis=1)))
        self.assertLess(max_diff, 1e-6, "drag altered the static-solve equilibrium")


if __name__ == "__main__":
    unittest.main(verbosity=2)
