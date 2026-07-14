#!/usr/bin/env python3
"""
Cable System Identification — Standalone Demo
==============================================

Recovers the physical properties of a flexible cable (rest bend angle and bend
stiffness) by matching simulated cable renders against reference images.

Algorithm
─────────
1. Load reference images from goal/, HLS-segment them to extract the orange
   cable silhouette → reference masks.
2. CMA-ES proposes batches of N candidate (bend_angle, stiffness) pairs.
3. Each batch is evaluated in one batched World (N cables simulated and
   ray-traced simultaneously via SensorTiledCamera tiling).
4. The rendered sim frames (same resolution as goal images) are HLS-segmented
   and compared to the reference masks via Dice loss.
5. Repeat until convergence and report the recovered parameters.

Dependencies
────────────
    numpy  cv2  matplotlib  cma  warp  newton

Usage
─────
    python cable_sysid_demo.py
"""

import datetime
import json
import math
import time
from pathlib import Path

import cma
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warp as wp

import newton
import newton.sensors as sensors
from newton.viewer import ViewerNull


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Camera position [m]
SENSOR_POS = [-0.96, -0.776, 0.594]

# Cable root attachment point [m]
CABLE_START = [0.06, -0.78, 0.65]

# Render resolution — must match the goal images.
RENDER_W, RENDER_H = 1920, 1080

# HLS bounds for segmenting the orange cable in the reference (goal) images.
GOAL_HLS_LO = [10,  90,  14]   # (H, L, S) lower
GOAL_HLS_HI = [30, 154, 255]   # (H, L, S) upper

# HLS bounds for segmenting the cable in Newton's ray-traced frames.
# The cable colour (0.9, 0.2, 0.1) appears as a dark orange-red
# ≈ (H=3, L=45, S=228) in OpenCV HLS; L > 15 excludes the black background.
SIM_HLS_LO = [0,  15, 100]
SIM_HLS_HI = [10, 100, 255]

# ── CMA-ES ───────────────────────────────────────────────────────────────────

POPSIZE = 32                      # candidates evaluated in parallel per generation
SIGMA0  = 0.2                     # initial step size
MAXITER = 50                      # max generations
X0      = [0.0, np.log10(5.0)]   # initial guess: straight cable, 5 N·m/rad

# ── Simulation ────────────────────────────────────────────────────────────────

SETTLE_FRAMES   = 2000   # frames run before scoring to let the cable reach equilibrium
NUM_EVAL_FRAMES = 1      # sim frames scored per evaluation (1 = single equilibrium pose)


# ─────────────────────────────────────────────────────────────────────────────
# Newton world
# ─────────────────────────────────────────────────────────────────────────────

def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world transform whose -Z axis points at ``target`` (OpenGL convention)."""
    eye = np.asarray(eye, dtype=np.float32)
    fwd = np.asarray(target, dtype=np.float32) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float32))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    rot = np.column_stack((right, cam_up, -fwd))
    return wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(rot.flatten())))


def _build_cable_chain(start, segment_length, angles,
                       initial_direction=wp.vec3(0.0, 0.0, -1.0)):
    """Per-joint (alpha, beta) bend angles → node positions and orientations."""
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d0  = np.array([initial_direction[0], initial_direction[1], initial_direction[2]],
                   dtype=np.float64)
    d0 /= np.linalg.norm(d0)
    cross = np.cross(ref, d0)
    dot   = float(np.dot(ref, d0))
    clen  = float(np.linalg.norm(cross))
    if clen < 1e-8:
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0 if dot > 0 else math.pi)
    else:
        q = wp.quat_from_axis_angle(wp.vec3(*(cross / clen).tolist()), math.atan2(clen, dot))

    points, quats = [start], []
    for alpha, beta in angles:
        quats.append(q)
        d = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
        points.append(points[-1] + d * segment_length)
        q = wp.mul(q, wp.mul(
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(alpha)),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(beta)),
        ))
    return points, quats


class World:
    """Newton simulation for one or more cable candidates evaluated in parallel.

    All N cables are built into a single Newton model (one world per cable),
    then simulated and ray-traced in a single pass, amortising GPU overhead
    across the CMA-ES population.
    """

    fps          = 60
    sim_substeps = 10
    num_elements = 10    # capsule segments per cable

    def __init__(self, cable_start, angles_list, stiffness_list, sensor_pos,
                 sim_iterations=5, settle_frames=0):
        self.n = len(angles_list)

        self.frame_dt = 1.0 / self.fps
        self.sim_dt   = self.frame_dt / self.sim_substeps

        segment_length = 0.05
        cable_radius   = 0.005
        cable_color    = (0.9, 0.2, 0.1)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.0e0

        self.cable_bodies_list = []

        for i, (angles, stiffness) in enumerate(zip(angles_list, stiffness_list)):
            builder.begin_world(label=f"cable_{i}")
            pts, quats = _build_cable_chain(wp.vec3(*cable_start), segment_length, angles)
            rod_bodies, _ = builder.add_rod(
                positions=pts, quaternions=quats, radius=cable_radius,
                stretch_stiffness=1.0e6, bend_stiffness=stiffness,
                bend_damping=1.0e-1, label="cable",
            )
            for b in rod_bodies:
                for s in builder.body_shapes[b]:
                    builder.shape_color[s] = cable_color

            # Root body is kinematic (zero mass → fixed attachment point)
            root = rod_bodies[0]
            builder.body_mass[root]        = 0.0
            builder.body_inv_mass[root]    = 0.0
            builder.body_inertia[root]     = wp.mat33(0.0)
            builder.body_inv_inertia[root] = wp.mat33(0.0)

            cable_mass = 0.0175
            for b in rod_bodies[1:]:
                builder.body_mass[b]     = cable_mass
                builder.body_inv_mass[b] = 1.0 / cable_mass

            self.cable_bodies_list.append(rod_bodies)
            builder.end_world()

        builder.color()
        self.model = builder.finalize()

        # Camera: RENDER_W×RENDER_H, looking along +X from sensor_pos.
        target = np.array(sensor_pos, np.float32) + np.array([1.0, 0.0, 0.0])
        pose   = look_at(sensor_pos, target)
        self.sensor            = sensors.SensorTiledCamera(self.model)
        self.rays              = self.sensor.utils.compute_pinhole_camera_rays(
                                     RENDER_W, RENDER_H, math.radians(45))
        self.color_image       = self.sensor.utils.create_color_image_output(
                                     RENDER_W, RENDER_H)
        self.camera_transforms = wp.array([[pose] * self.n], dtype=wp.transformf,
                                          device=self.model.device)

        self.solver   = newton.solvers.SolverVBD(self.model, iterations=sim_iterations,
                                                  rigid_avbd_contact_alpha=0.0)
        self.state_0  = self.model.state()
        self.state_1  = self.model.state()
        self.control  = self.model.control()
        self.contacts = self.model.contacts()
        self.sim_frame_idx = wp.array([0], dtype=wp.int32, device=self.model.device)

        # Start straight; bend energy relaxes the cable to its rest shape during settling
        straight_pts, straight_q = _build_cable_chain(
            wp.vec3(*cable_start), segment_length, [(0.0, 0.0)] * self.num_elements)
        bq = self.state_0.body_q.numpy()
        for bodies in self.cable_bodies_list:
            for i, b in enumerate(bodies):
                p, q = straight_pts[i], straight_q[i]
                bq[b] = (p[0], p[1], p[2], q[0], q[1], q[2], q[3])
        self.state_0.body_q.assign(bq)
        self.state_1.body_q.assign(bq)
        self.solver.body_q_prev.assign(bq)
        self.state_0.body_qd.zero_()
        self.state_1.body_qd.zero_()

        viewer = ViewerNull()
        viewer.set_model(self.model)
        self._viewer = viewer

        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._substeps(viewer)
            self.graph = capture.graph
        else:
            self.graph = None

        if settle_frames:
            for _ in range(settle_frames):
                self._step(0)

    def _substeps(self, viewer=None):
        v = viewer or self._viewer
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            v.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _step(self, frame_idx):
        self.sim_frame_idx.fill_(frame_idx)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._substeps()

    def _render(self):
        """Ray-trace all N cables; return list of RGBA uint8 arrays (H, W, 4)."""
        self.model.bvh_refit_shapes(self.state_1)
        self.model.bvh_refit_particles(self.state_1)
        self.sensor.update(
            self.state_1, self.camera_transforms, self.rays,
            color_image=self.color_image,
            clear_data=sensors.SensorTiledCamera.ClearData(
                clear_color=0xFF000000, clear_albedo=0xFF000000),
        )
        color_rgba = self.sensor.utils.to_rgba_from_color(self.color_image)
        arr = color_rgba.numpy()
        return [np.ascontiguousarray(arr[i]) for i in range(self.n)]

    def run_sequence(self, num_frames, ref_masks, sim_hls_lo, sim_hls_hi,
                     settle_frames=0):
        """Step ``num_frames``, render at reference-aligned frames, return per-world Dice losses."""
        if settle_frames:
            for _ in range(settle_frames):
                self._step(0)

        n_ref = len(ref_masks)
        losses = [0.0] * self.n

        sim_to_ref = {}
        for j in range(n_ref):
            f = round(j / max(1, n_ref - 1) * (num_frames - 1))
            sim_to_ref.setdefault(f, []).append(j)

        for f in range(num_frames):
            self._step(f)
            if f in sim_to_ref:
                frames = self._render()
                for j in sim_to_ref[f]:
                    for i in range(self.n):
                        sim_mask = segment_hsl(frames[i], sim_hls_lo, sim_hls_hi)
                        losses[i] += dice_loss(sim_mask, ref_masks[j])

        return [l / n_ref for l in losses]


# ─────────────────────────────────────────────────────────────────────────────
# Loss and segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_hsl(img, lower, upper):
    """Binary HLS threshold of an RGB(A) uint8 image.  Returns (H, W) uint8 mask."""
    hls = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2HLS)
    return cv2.inRange(hls, np.array(lower, np.uint8), np.array(upper, np.uint8))


def dice_loss(mask_a, mask_b, eps=1.0):
    """Sørensen–Dice loss: 0 = perfect overlap, ≈1 = no overlap."""
    a = (mask_a > 0).astype(np.float32).ravel()
    b = (mask_b > 0).astype(np.float32).ravel()
    return float(1.0 - (2.0 * np.dot(a, b) + eps) / (a.sum() + b.sum() + eps))


# ─────────────────────────────────────────────────────────────────────────────
# Goal image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_goal_frames(goal_dir: Path):
    """Load PNG files from ``goal_dir`` as RGB uint8 arrays, sorted by name."""
    paths = sorted(goal_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {goal_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"Could not read {p}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = frames[0].shape[:2]
    print(f"  Loaded {len(frames)} frame(s) from {goal_dir}  ({w}×{h})")
    if (w, h) != (RENDER_W, RENDER_H):
        raise ValueError(
            f"Goal images are {w}×{h} but RENDER_W×RENDER_H = {RENDER_W}×{RENDER_H}. "
            f"Update RENDER_W/RENDER_H at the top of this file to match.")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES optimiser
# ─────────────────────────────────────────────────────────────────────────────

def optimize(ref_masks):
    """CMA-ES over [bend_angle, log₁₀(stiffness)]; return (angle, stiffness, history)."""
    n = World.num_elements

    def to_angles(x):
        return [(float(x[0]), 0.0)] * n

    def to_stiffness(x):
        return 10.0 ** float(x[1])

    history = []
    es = cma.CMAEvolutionStrategy(X0, SIGMA0, {
        'popsize':  POPSIZE,
        'maxiter':  MAXITER,
        'verbose':  -9,
        'tolx':     1e-6,
        'tolfun':   1e-9,
    })

    print(f"\n  {'Iter':>4}  {'Best loss':>10}  {'Angle (rad)':>12}  {'Stiffness':>12}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*12}")

    while not es.stop():
        solutions      = es.ask()
        angles_list    = [to_angles(x)    for x in solutions]
        stiffness_list = [to_stiffness(x) for x in solutions]

        world = World(CABLE_START, angles_list, stiffness_list, SENSOR_POS)
        fitnesses = world.run_sequence(
            NUM_EVAL_FRAMES, ref_masks, SIM_HLS_LO, SIM_HLS_HI,
            settle_frames=SETTLE_FRAMES,
        )
        es.tell(solutions, fitnesses)

        bx = es.result.xbest
        entry = dict(
            iter=len(history),
            best_loss=float(es.result.fbest),
            best_angle=float(bx[0]),
            best_stiffness=float(to_stiffness(bx)),
            mean_loss=float(np.mean(fitnesses)),
        )
        history.append(entry)

        if len(history) % 5 == 0 or len(history) == 1:
            print(f"  {entry['iter']:>4}  {entry['best_loss']:>10.6f}  "
                  f"{entry['best_angle']:>+12.4f}  {entry['best_stiffness']:>12.2f}")

    bx = es.result.xbest
    return float(bx[0]), float(to_stiffness(bx)), history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

BG, DARK, GRID = '#1a1a2e', '#0d0d1a', '#333333'


def _style(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5)


def save_figure(goal_rgb, ref_mask, rec_frame, rec_mask, history, out_dir):
    """5-panel figure: goal photo | goal mask | sim result | overlap | loss curve."""
    fig = plt.figure(figsize=(18, 4.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.3)

    def show(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=9)
        ax.axis('off')

    def coloured(mask, rgb):
        out = np.zeros((*mask.shape, 3), np.uint8)
        out[mask > 0] = rgb
        return out

    show(fig.add_subplot(gs[0]), goal_rgb,
         'Reference image\n(goal/)')
    show(fig.add_subplot(gs[1]), coloured(ref_mask, [255, 160, 50]),
         'Reference mask\n(HLS segmented)')
    show(fig.add_subplot(gs[2]), rec_frame[..., :3],
         'Sim render\n(recovered params)')

    ov = np.zeros((*ref_mask.shape, 3), np.uint8)
    ov[ref_mask > 0]                       = [255, 160,  50]
    ov[(ref_mask > 0) & (rec_mask > 0)]    = [255, 255, 255]
    ov[(ref_mask == 0) & (rec_mask > 0)]   = [ 50, 210, 255]
    show(fig.add_subplot(gs[3]), ov, 'Overlap\n(white = match)')

    ax = fig.add_subplot(gs[4])
    ax.plot([h['iter'] for h in history], [h['best_loss'] for h in history],
            color='#4fc3f7', linewidth=2)
    ax.set_xlabel('Iteration', color='white', fontsize=8)
    ax.set_ylabel('Dice loss',  color='white', fontsize=8)
    ax.set_title('CMA-ES convergence', color='white', fontsize=9)
    _style(ax)

    fig.savefig(out_dir / 'results.png', dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved results.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    demo_dir = Path(__file__).parent
    out_dir  = demo_dir / 'output'
    out_dir.mkdir(exist_ok=True)

    print("=" * 62)
    print("Cable System Identification — Standalone Demo")
    print("=" * 62)
    print(f"sensor_pos   = {SENSOR_POS}")
    print(f"cable_start  = {CABLE_START}")
    print(f"render res   = {RENDER_W}×{RENDER_H}")
    print(f"initial guess: α = {X0[0]:.3f} rad,  k = {10.0**X0[1]:.1f} N·m/rad")
    print()

    # ── 1. Load reference ─────────────────────────────────────────────────────
    print("[1/3] Loading reference images …")
    goal_frames = load_goal_frames(demo_dir / 'goal')
    ref_masks = [segment_hsl(f, GOAL_HLS_LO, GOAL_HLS_HI) for f in goal_frames]
    set_px = ref_masks[0].sum() // 255
    print(f"  Cable pixels in first mask: {set_px}  "
          f"({100 * set_px / (RENDER_W * RENDER_H):.3f}% of frame)\n")

    # ── 2. CMA-ES ─────────────────────────────────────────────────────────────
    print(f"[2/3] CMA-ES optimisation  (popsize={POPSIZE}, maxiter≤{MAXITER}) …")
    t0 = time.perf_counter()
    best_angle, best_stiffness, history = optimize(ref_masks)
    elapsed = time.perf_counter() - t0
    print()

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print("[3/3] Results")

    angles = [(best_angle, 0.0)] * World.num_elements
    world  = World(CABLE_START, [angles], [best_stiffness], SENSOR_POS,
                   settle_frames=SETTLE_FRAMES)
    world._step(0)
    rec_frame = world._render()[0]
    rec_mask  = segment_hsl(rec_frame, SIM_HLS_LO, SIM_HLS_HI)
    final_loss = dice_loss(rec_mask, ref_masks[0])

    print(f"\n  Recovered:  α = {best_angle:+.4f} rad,  "
          f"k = {best_stiffness:.2f} N·m/rad")
    print(f"  Final Dice loss : {final_loss:.6f}")
    print(f"  Iterations      : {len(history)}")
    print(f"  Wall time       : {elapsed:.1f} s")
    print()

    result = {
        'timestamp':       datetime.datetime.now().isoformat(timespec='seconds'),
        'recovered':       {'bend_angle_rad': best_angle, 'bend_stiffness_Nm': best_stiffness},
        'final_dice_loss': final_loss,
        'n_iterations':    len(history),
        'elapsed_s':       elapsed,
    }
    with open(out_dir / 'result.json', 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved result.json")

    cv2.imwrite(str(out_dir / 'mask_goal.png'), ref_masks[0])
    cv2.imwrite(str(out_dir / 'mask_sim.png'),  rec_mask)
    print(f"  Saved mask_goal.png, mask_sim.png")

    save_figure(goal_frames[0], ref_masks[0], rec_frame, rec_mask, history, out_dir)
    print(f"\nAll outputs in {out_dir}/")


if __name__ == '__main__':
    main()
