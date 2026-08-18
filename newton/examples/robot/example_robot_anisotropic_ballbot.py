# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-balancing ballbot driven by three anisotropic-friction cylinders."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

_WHEEL_RADIUS = 0.12
_WHEEL_HALF_WIDTH = 0.027
_MU_HIGH = 2.0
_MU_LOW = 0.1


@wp.kernel
def set_ballbot_torques(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    body: int,
    ball: int,
    target_position: wp.array[wp.vec3],
    command: wp.array[wp.vec3],
    disturbance: wp.array[wp.vec3],
    drive_dofs: wp.array[int],
    drive_directions: wp.array[wp.vec3],
    gains: wp.array[float],
):
    """Apply full-state feedback through the three wheel joints."""
    rotation = wp.transform_get_rotation(body_q[body])
    up = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, 1.0))
    angular_velocity = wp.quat_rotate_inv(rotation, wp.spatial_bottom(body_qd[body]))
    body_f[body] += wp.spatial_vector(disturbance[0], wp.vec3())

    body_position = wp.transform_get_translation(body_q[body])
    ball_position = wp.transform_get_translation(body_q[ball])
    relative_position = body_position - ball_position
    horizontal_offset = wp.length(wp.vec3(relative_position[0], relative_position[1], 0.0))
    if up[2] < 0.85 or relative_position[2] < 0.25 or horizontal_offset > 0.25:
        for wheel in range(3):
            joint_f[drive_dofs[wheel]] = 0.0
        return

    body_velocity = wp.spatial_top(body_qd[body])
    position_error = body_position - target_position[0]
    position_local = wp.quat_rotate_inv(rotation, wp.vec3(position_error[0], position_error[1], 0.0))
    velocity_local = wp.quat_rotate_inv(rotation, wp.vec3(body_velocity[0], body_velocity[1], 0.0))
    target_velocity_world = command[0]
    target_velocity = wp.quat_rotate_inv(rotation, wp.vec3(target_velocity_world[0], target_velocity_world[1], 0.0))
    lean = wp.vec3(-up[0], -up[1], 0.0)
    lean_rate = wp.vec3(angular_velocity[1], -angular_velocity[0], 0.0)
    velocity_error = velocity_local - wp.vec3(target_velocity[0], target_velocity[1], 0.0)
    planar_command = gains[0] * position_local + gains[1] * velocity_error
    planar_command += gains[2] * lean + gains[3] * lean_rate
    yaw_command = gains[5] * (angular_velocity[2] - target_velocity_world[2])

    for wheel in range(3):
        dof = drive_dofs[wheel]
        wheel_speed = joint_qd[dof]
        overspeed_damping = wp.sign(wheel_speed) * 2.0 * wp.max(wp.abs(wheel_speed) - 12.0, 0.0)
        torque = (
            (2.0 / 3.0) * wp.dot(planar_command, drive_directions[wheel])
            + yaw_command / 3.0
            - gains[4] * wheel_speed
            - overspeed_damping
        )
        joint_f[dof] = wp.clamp(torque, -12.0, 12.0)


@wp.kernel
def align_wheel_contact_frames(
    contact_geom: wp.array[wp.vec2i],
    contact_worldid: wp.array[int],
    contact_count: wp.array[int],
    geom_xmat: wp.array2d[wp.mat33],
    wheel_geom_mask: wp.array[int],
    contact_frame: wp.array[wp.mat33],
):
    """Rebuild each ball-wheel contact basis around the wheel axis.

    MuJoCo seeds the contact tangent basis from a fixed world axis, so which
    tangent receives which coefficient of ``mujoco:pair_friction`` depends on the
    ballbot's heading rather than on the wheel it belongs to. The ball only ever
    touches the curved side of a wheel, so the wheel axis lies in the contact
    tangent plane and is exactly the direction that must slide freely; rebuilding
    the basis around it puts ``_MU_LOW`` on the axis and ``_MU_HIGH`` on the
    rolling direction at every heading.
    """
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    geoms = contact_geom[contact]
    wheel = int(-1)
    if wheel_geom_mask[geoms[0]] == 1:
        wheel = geoms[0]
    elif wheel_geom_mask[geoms[1]] == 1:
        wheel = geoms[1]
    if wheel < 0:
        return

    normal = contact_frame[contact][0]
    xmat = geom_xmat[contact_worldid[contact], wheel]
    axis = wp.vec3(xmat[0, 2], xmat[1, 2], xmat[2, 2])

    slide = axis - normal * wp.dot(normal, axis)
    # Degenerate only for rim or end-cap contact, where MuJoCo's basis still applies.
    if wp.length(slide) < 1.0e-6:
        return
    slide = wp.normalize(slide)
    roll = wp.cross(normal, slide)

    # fmt: off
    contact_frame[contact] = wp.mat33(
        normal[0], normal[1], normal[2],
        slide[0], slide[1], slide[2],
        roll[0], roll[1], roll[2],
    )
    # fmt: on


def add_anisotropic_wheel(
    builder: newton.ModelBuilder,
    parent: int,
    ball_shape: int,
    position: wp.vec3,
    orientation: wp.quat,
    label: str,
    vec5f: type,
) -> int:
    """Add a driven cylinder with anisotropic sphere-contact friction.

    Args:
        builder: Model builder receiving the wheel.
        parent: Parent ballbot body.
        ball_shape: Shape index of the driven sphere.
        position: Wheel center in the parent frame [m].
        orientation: Rotation from the wheel frame to the parent frame.
        label: Label prefix for the wheel body, joint, and shape.
        vec5f: Warp vector type used by MuJoCo pair friction.

    Returns:
        The driven cylinder joint.
    """
    wheel = builder.add_link(label=label)
    wheel_shape = builder.add_shape_cylinder(
        wheel,
        radius=_WHEEL_RADIUS,
        half_height=_WHEEL_HALF_WIDTH,
        cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=1.0, gap=0.01),
        color=(0.12, 0.55, 0.95),
        label=label,
    )
    for marker_side in (-1.0, 1.0):
        builder.add_shape_sphere(
            wheel,
            xform=wp.transform(
                p=wp.vec3(0.65 * _WHEEL_RADIUS, 0.0, marker_side * (_WHEEL_HALF_WIDTH + 0.004)),
                q=wp.quat_identity(),
            ),
            radius=0.018,
            as_site=True,
            color=(0.0, 0.0, 0.0),
            label=f"{label}_marker_{'positive' if marker_side > 0.0 else 'negative'}",
        )

    drive_joint = builder.add_joint_revolute(
        parent=parent,
        child=wheel,
        parent_xform=wp.transform(p=position, q=orientation),
        child_xform=wp.transform(),
        axis=newton.Axis.Z,
        target_kd=0.0,
        damping=0.03,
        armature=0.002,
        effort_limit=12.0,
        velocity_limit=20.0,
        actuator_mode=newton.JointTargetMode.EFFORT,
        label=f"{label}_drive",
    )
    builder.add_custom_values(
        **{
            "mujoco:pair_world": -1,
            "mujoco:pair_geom1": ball_shape,
            "mujoco:pair_geom2": wheel_shape,
            "mujoco:pair_condim": 3,
            "mujoco:pair_friction": vec5f(_MU_LOW, _MU_HIGH, 0.0, 0.0, 0.0),
        }
    )
    return drive_joint


class Example:
    """Balance a cylindrical body on a sphere using anisotropic wheels."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 0.01
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        vec5f = wp.types.vector(length=5, dtype=wp.float32)

        ballbot_x = 0.0
        ball_radius = 0.30
        ball = builder.add_link(
            xform=wp.transform(p=wp.vec3(ballbot_x, 0.0, ball_radius), q=wp.quat_identity()),
            label="ballbot_ball",
        )
        ball_shape = builder.add_shape_sphere(
            ball,
            radius=ball_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=35.0, mu=1.8, gap=0.01),
            color=(0.08, 0.09, 0.11),
            label="ballbot_ball",
        )
        builder.add_articulation(
            [builder.add_joint_free(ball, label="ballbot_ball_free")],
            label="ballbot_ball",
        )

        ballbot_body_height = 1.10
        initial_tilt_axis = wp.normalize(wp.vec3(1.0, 1.0, 0.0))
        initial_tilt = wp.quat_from_axis_angle(initial_tilt_axis, math.radians(1.0))
        ballbot_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(ballbot_x, 0.0, ballbot_body_height), q=initial_tilt),
            label="ballbot_body",
        )
        ballbot_body_shape = builder.add_shape_cylinder(
            ballbot_body,
            radius=0.10,
            half_height=0.45,
            cfg=newton.ModelBuilder.ShapeConfig(density=350.0, mu=0.8, gap=0.01),
            color=(0.88, 0.30, 0.08),
            label="ballbot_body",
        )
        builder.add_shape_collision_filter_pair(ball_shape, ballbot_body_shape)
        body_free_joint = builder.add_joint_free(ballbot_body, label="ballbot_body_free")
        body_qd_start = builder.joint_qd_start[body_free_joint]
        builder.joint_qd[body_qd_start] = 0.05
        builder.joint_qd[body_qd_start + 1] = -0.05

        ballbot_joints = [body_free_joint]
        drive_joints = []
        drive_directions = []
        wheel_center_distance = ball_radius + _WHEEL_RADIUS
        wheel_radial_offset = wheel_center_distance / math.sqrt(2.0)
        wheel_height = ball_radius + wheel_radial_offset + 0.01
        for wheel_index in range(3):
            angle = 2.0 * math.pi * wheel_index / 3.0
            radial = wp.vec3(math.cos(angle), math.sin(angle), 0.0)
            tangent = wp.vec3(-math.sin(angle), math.cos(angle), 0.0)
            wheel_position = radial * wheel_radial_offset + wp.vec3(
                0.0,
                0.0,
                wheel_height - ballbot_body_height,
            )
            drive_axis = (-radial + wp.vec3(0.0, 0.0, 1.0)) / math.sqrt(2.0)
            wheel_orientation = newton.math.quat_between_vectors_robust(
                wp.vec3(0.0, 0.0, 1.0),
                drive_axis,
            )
            drive_joint = add_anisotropic_wheel(
                builder,
                ballbot_body,
                ball_shape,
                wheel_position,
                wheel_orientation,
                f"ballbot_wheel_{wheel_index}",
                vec5f,
            )
            drive_joints.append(drive_joint)
            drive_directions.append(tangent)
            ballbot_joints.append(drive_joint)

        builder.add_articulation(ballbot_joints, label="ballbot")
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.5, gap=0.01))

        self.model = builder.finalize()
        self.ballbot_ball = ball
        self.ballbot_body = ballbot_body
        self.ballbot_target_position = wp.array(
            [wp.vec3(ballbot_x, 0.0, ballbot_body_height)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.ballbot_target_radius = wp.array([0.05], dtype=float, device=self.model.device)
        self.ballbot_target_color = wp.array(
            [wp.vec3(0.1, 0.9, 0.2)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.ballbot_command = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.ballbot_disturbance = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.ballbot_circle_center = wp.vec3(ballbot_x - 1.0, 0.0, ballbot_body_height)
        self.ballbot_circle_radius = 1.0
        self.ballbot_circle_speed = 0.40
        self.ballbot_velocity_feedforward = 0.75
        self.ballbot_target_velocity = np.zeros(2, dtype=np.float32)

        joint_qd_start = self.model.joint_qd_start.numpy()
        self.drive_dof_indices = np.array(
            [joint_qd_start[joint] for joint in drive_joints],
            dtype=np.int32,
        )
        self.drive_dofs = wp.array(self.drive_dof_indices, dtype=int, device=self.model.device)
        self.drive_directions = wp.array(drive_directions, dtype=wp.vec3, device=self.model.device)
        # Full-state position, velocity, lean, and lean-rate gains; wheel and yaw damping.
        # The lean terms are deliberately soft so the recovery from each disturbance
        # stays visible; stiffening them past roughly -150/-90 flattens it out.
        self.controller_gains = wp.array(
            [-20.0, -40.0, -90.0, -65.0, 1.25, -15.0],
            dtype=float,
            device=self.model.device,
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=True,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=10,
            ls_iterations=20,
            njmax=2048,
            nconmax=1024,
        )

        # The anisotropic pair friction only reaches the intended directions once
        # the contact basis is rebuilt per wheel; see align_wheel_contact_frames().
        wheel_geom_mask = np.zeros(self.solver.mj_model.ngeom, dtype=np.int32)
        for pair in range(self.solver.mj_model.npair):
            wheel_geom_mask[int(self.solver.mj_model.pair_geom2[pair])] = 1
        self.wheel_geom_mask = wp.array(wheel_geom_mask, dtype=int, device=self.model.device)
        self.solver.mjw_model.callback.contactfilter = self.align_contact_frames

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.min_up = 1.0
        self.min_height = math.inf
        self.max_ball_offset = 0.0
        self.max_drive_speed = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.4, -3.2, 2.0), pitch=-24.0, yaw=128.0)
        self.device = wp.get_device()
        self.capture()

    def align_contact_frames(self, mjw_model, mjw_data):
        """Re-align ball-wheel contact bases once narrowphase has written them."""
        wp.launch(
            align_wheel_contact_frames,
            dim=mjw_data.contact.frame.shape[0],
            inputs=[
                mjw_data.contact.geom,
                mjw_data.contact.worldid,
                mjw_data.nacon,
                mjw_data.geom_xmat,
                self.wheel_geom_mask,
                mjw_data.contact.frame,
            ],
        )

    def capture(self):
        """Capture one simulation frame."""
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                set_ballbot_torques,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.state_0.joint_qd,
                    self.control.joint_f,
                    self.ballbot_body,
                    self.ballbot_ball,
                    self.ballbot_target_position,
                    self.ballbot_command,
                    self.ballbot_disturbance,
                    self.drive_dofs,
                    self.drive_directions,
                    self.controller_gains,
                ],
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Update the target trajectory and advance one frame."""
        circle_time = max(self.sim_time - 2.0, 0.0)
        circle_angle = self.ballbot_circle_speed * circle_time / self.ballbot_circle_radius
        circle_cos = math.cos(circle_angle)
        circle_sin = math.sin(circle_angle)
        self.ballbot_target_position.fill_(
            self.ballbot_circle_center
            + wp.vec3(self.ballbot_circle_radius * circle_cos, self.ballbot_circle_radius * circle_sin, 0.0)
        )
        if self.sim_time < 2.0:
            self.ballbot_target_velocity.fill(0.0)
        else:
            self.ballbot_target_velocity[:] = (
                -self.ballbot_circle_speed * circle_sin,
                self.ballbot_circle_speed * circle_cos,
            )
        self.ballbot_command.fill_(
            wp.vec3(
                self.ballbot_velocity_feedforward * float(self.ballbot_target_velocity[0]),
                self.ballbot_velocity_feedforward * float(self.ballbot_target_velocity[1]),
                0.0,
            )
        )

        disturbance_phase = self.sim_time % 8.0
        if self.sim_time >= 8.0 and disturbance_phase < 0.15:
            disturbance_index = int(self.sim_time / 8.0) % 4
            disturbance_directions = (
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
                wp.vec3(-1.0, 0.0, 0.0),
                wp.vec3(0.0, -1.0, 0.0),
            )
            self.ballbot_disturbance.fill_(12.0 * disturbance_directions[disturbance_index])
        else:
            self.ballbot_disturbance.fill_(wp.vec3())

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Track balance and wheel motion."""
        body_q = self.state_0.body_q.numpy()[self.ballbot_body]
        body_qd = self.state_0.body_qd.numpy()[self.ballbot_body]
        ball_q = self.state_0.body_q.numpy()[self.ballbot_ball]
        assert np.all(np.isfinite(body_q)), f"non-finite ballbot pose at t={self.sim_time:.3f} s"
        assert np.all(np.isfinite(body_qd)), f"non-finite ballbot velocity at t={self.sim_time:.3f} s"

        rotation = wp.quat(*body_q[3:7])
        up = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        self.min_up = min(self.min_up, float(up[2]))
        self.min_height = min(self.min_height, float(body_q[2]))
        self.max_ball_offset = max(
            self.max_ball_offset,
            float(np.linalg.norm(body_q[:2] - ball_q[:2])),
        )
        joint_qd = self.state_0.joint_qd.numpy()
        self.max_drive_speed = max(
            self.max_drive_speed,
            float(np.max(np.abs(joint_qd[self.drive_dof_indices]))),
        )

    def test_final(self):
        """Verify finite, supported motion and active wheel drives."""
        assert self.min_up > 0.75, f"ballbot lost balance: minimum up={self.min_up:.3f}"
        assert self.min_height > 0.50, f"ballbot body lost support: minimum z={self.min_height:.3f} m"
        assert self.max_ball_offset < 0.30, f"ballbot lost the ball: maximum offset={self.max_ball_offset:.3f} m"
        assert self.max_drive_speed > 0.1, "anisotropic wheel drives did not engage"

    def render(self):
        """Render the ballbot and its target."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "ballbot_target",
            self.ballbot_target_position,
            self.ballbot_target_radius,
            self.ballbot_target_color,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
