#!/usr/bin/env python3
"""Shared cable-building helpers for ``droop_figure.py``.

Not a standalone script -- ``droop_figure.py`` imports ``_import_newton``, ``_make_solver``,
``build_cable``, and ``rod_nodes`` from here directly.
"""

from __future__ import annotations

import os
import sys

import numpy as np


def _import_newton(newton_path: str | None):
    """Import newton, optionally from a specific checkout (takes precedence on sys.path)."""
    if newton_path:
        sys.path.insert(0, os.path.abspath(newton_path))
    import warp as wp  # noqa: PLC0415
    import newton  # noqa: PLC0415

    return wp, newton


def _straight_points(newton, wp, start, direction, length, num_segments):
    """Rod centerline helper. ``create_straight_cable_*`` was renamed to ``rod_straight_*``
    (deprecated in Newton 1.6); prefer the new name and fall back for older checkouts."""
    fn = getattr(newton.utils, "rod_straight_points_and_quaternions", None)
    if fn is None:
        fn = newton.utils.create_straight_cable_points_and_quaternions
    return fn(start=start, direction=direction, length=length, num_segments=num_segments)


def build_cable(newton, wp, *, num_elements, segment_length, radius, bend_stiffness,
                bend_damping, stretch_stiffness, total_mass, device):
    """Cable clamped at one end, built HORIZONTAL along +X so it must relax under gravity."""
    length = num_elements * segment_length
    points, quats = _straight_points(
        newton, wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0), length, num_elements
    )

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
    rod_bodies, _ = builder.add_rod(
        positions=points,
        quaternions=quats,
        radius=radius,
        stretch_stiffness=stretch_stiffness,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=bend_stiffness,
        twist_damping=bend_damping,
        label="cable",
        body_frame_origin="com",
    )

    anchor = rod_bodies[0]
    builder.body_flags[anchor] = int(newton.BodyFlags.KINEMATIC)
    builder.body_mass[anchor] = 0.0
    builder.body_inv_mass[anchor] = 0.0
    builder.body_inertia[anchor] = wp.mat33(0.0)
    builder.body_inv_inertia[anchor] = wp.mat33(0.0)

    per_body = total_mass / num_elements
    for b in rod_bodies[1:]:
        builder.body_mass[b] = per_body
        builder.body_inv_mass[b] = 1.0 / per_body

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, rod_bodies


def _quat_rotate(q, v):
    """Rotate ``v`` by warp-ordered quaternion ``q`` = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def rod_nodes(body_q, rod_bodies, segment_length):
    """Centerline nodes (num_elements+1, 3) from the rod's per-segment body transforms.

    ``add_rod(body_frame_origin="com")`` puts each body at its segment's *midpoint*, so the
    body positions span only ``[l/2, L - l/2]`` -- half a segment short at each end. Deflection
    read straight off the last body is therefore the deflection at ``x = L - l/2``, not at the
    tip, which under-reports it by ``delta(L - l/2) / delta(L)`` (~6.7% for 10 segments).
    Reconstructing the nodes removes that bias and makes the measured span match beam theory's.

    The segment tangent is the body frame's local +z (verified against the as-built pose).
    """
    pos = body_q[rod_bodies, :3]
    quat = body_q[rod_bodies, 3:7]
    half = 0.5 * segment_length
    tangents = np.array([_quat_rotate(q, np.array([0.0, 0.0, 1.0])) for q in quat])
    nodes = np.empty((len(rod_bodies) + 1, 3))
    nodes[:-1] = pos - half * tangents
    nodes[-1] = pos[-1] + half * tangents[-1]
    return nodes


def _make_solver(newton, model, *, iterations, static_solve, compliant_alm):
    from newton._src.solvers.vbd.solver_vbd import SolverVBD  # noqa: PLC0415

    kwargs = dict(iterations=iterations, static_solve=static_solve, rigid_avbd_contact_alpha=0.0)
    # These are additions on the static-solve branch / newer main; stay importable without them.
    import inspect  # noqa: PLC0415

    params = inspect.signature(SolverVBD.__init__).parameters
    if "rigid_chebyshev_rho" in params:
        kwargs["rigid_chebyshev_rho"] = 0.0
    if "rigid_compliant_alm" in params and compliant_alm is not None:
        kwargs["rigid_compliant_alm"] = compliant_alm
    return SolverVBD(model, **kwargs)
