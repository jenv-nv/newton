# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Aerodynamic Drag Demo (VBD cable)
#
# A single cable is pinned at one end (kinematic root) and released
# horizontally, so its free end falls and swings under gravity. The VBD
# solver's implicit aerodynamic drag (drag_linear / drag_angular) opposes each
# body's ABSOLUTE velocity, so raising the coefficient damps the swing until
# the cable eases into its droop instead of oscillating about it.
#
# This is the knob-with-eyeballs companion to test_vbd_rod_drag.py: unlike the
# rod's bend/twist damping (which only sees RELATIVE joint motion), drag removes
# the bulk swinging energy air resistance would.
#
# UI (GL viewer):
#   - "Reset" button           : re-release the cable from horizontal.
#   - "Linear drag" slider      : drag_linear  [N s/m].
#   - "Angular drag" slider     : drag_angular [N m s/rad].
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_drag
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_drag --test --viewer null
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


class Example:
    """Pinned cable released horizontally; interactive aerodynamic drag."""

    NUM_ELEMENTS = 16
    SEGMENT_LENGTH = 0.10
    CABLE_RADIUS = 0.01
    BEND_STIFFNESS = 20.0
    BEND_DAMPING = 0.0  # no internal (relative) damping: isolate the aerodynamic drag

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cable_length = self.NUM_ELEMENTS * self.SEGMENT_LENGTH

        # Gravity along -Z: the horizontal cable's free end falls in -Z.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

        start = wp.vec3(0.0, 0.0, 0.0)
        points = newton.utils.cable_straight_points(
            start=start,
            direction=wp.vec3(1.0, 0.0, 0.0),  # horizontal, along +X
            length=self.cable_length,
            num_segments=self.NUM_ELEMENTS,
        )
        quats = newton.utils.rod_parallel_transport_quaternions(points)

        rod_bodies, _ = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.CABLE_RADIUS,
            stretch_stiffness=1.0e5,
            bend_stiffness=self.BEND_STIFFNESS,
            bend_damping=self.BEND_DAMPING,
            label="drag_cable",
            body_frame_origin="com",
        )
        # Zero mass + zero inertia pins the root kinematically in Newton's VBD.
        root = rod_bodies[0]
        builder.body_mass[root] = 0.0
        builder.body_inv_mass[root] = 0.0
        builder.body_inertia[root] = wp.mat33(0.0)
        builder.body_inv_inertia[root] = wp.mat33(0.0)

        self.tip_body = int(rod_bodies[-1])

        builder.color()
        self.model = builder.finalize()

        # Live drag coefficients, mutated by the UI and pushed onto the solver each
        # substep. Start at zero so the first thing you see is the undamped swing.
        self.drag_linear = 0.0
        self.drag_angular = 0.0
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.sim_iterations, rigid_compliant_alm=True,
            drag_linear=self.drag_linear, drag_angular=self.drag_angular,
        )

        # Slider ranges sized off the cable's mass/inertia so the coefficient maps to
        # an intuitive decay rate: the free-body per-second decay is c*inv (see
        # test_vbd_rod_drag.py). Full-scale ~ heavy damping.
        free = np.asarray(rod_bodies[1:])
        inv_m = float(np.mean(self.model.body_inv_mass.numpy()[free]))
        inv_I = float(np.mean([np.trace(m) / 3.0 for m in self.model.body_inv_inertia.numpy()[free]]))
        self.drag_linear_max = 30.0 / inv_m   # up to ~30 /s linear decay
        self.drag_angular_max = 30.0 / inv_I  # up to ~30 /s angular decay

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.anchor_z = float(node_xyz(self.state_0.body_q.numpy()[rod_bodies[0]], self.SEGMENT_LENGTH)[2])

        self._reset_requested = False

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.5 * self.cable_length, -2.6, 0.0),
            target=wp.vec3(0.5 * self.cable_length, 0.0, -0.5 * self.cable_length),
            fov=32.0,
        )

        # No CUDA-graph capture: the drag coefficients are plain scalars read at each
        # step(), so a captured graph would freeze them at capture-time values and the
        # sliders would do nothing. The cable is tiny, so ungraphed stepping is fine.
        self.graph = None

    # ----- UI --------------------------------------------------------------
    def gui(self, ui):
        # Auto-registered by newton.examples.run when the viewer has a GUI (a null
        # viewer used for --test has no gui(), so this is simply never called there).
        ui.text("Aerodynamic drag")
        if ui.button("Reset (re-release from horizontal)"):
            self._reset_requested = True
        _changed, self.drag_linear = ui.slider_float(
            "Linear drag [N s/m]", self.drag_linear, 0.0, self.drag_linear_max)
        _changed, self.drag_angular = ui.slider_float(
            "Angular drag [N m s/rad]", self.drag_angular, 0.0, self.drag_angular_max)
        tip_z = float(node_xyz(self.state_0.body_q.numpy()[self.tip_body], self.SEGMENT_LENGTH)[2])
        ui.text(f"tip drop: {self.anchor_z - tip_z:+.3f} m")

    def _reset(self):
        self.state_0 = self.model.state()  # fresh states restore the horizontal rest pose
        self.state_1 = self.model.state()
        self.sim_time = 0.0

    # ----- sim loop --------------------------------------------------------
    def _simulate_substeps(self):
        # Push the current (possibly just-moved) slider values onto the solver.
        self.solver.drag_linear = self.drag_linear
        self.solver.drag_angular = self.drag_angular
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self._reset_requested:
            self._reset_requested = False
            self._reset()
        self._simulate_substeps()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        tip_z = float(node_xyz(self.state_0.body_q.numpy()[self.tip_body], self.SEGMENT_LENGTH)[2])
        assert np.isfinite(tip_z), "tip position is non-finite"
        # Released horizontal under gravity, the free end must have dropped below the anchor.
        assert tip_z < self.anchor_z - 0.1 * self.cable_length, (
            f"free end did not fall: tip_z={tip_z:.3f}, anchor_z={self.anchor_z:.3f}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
