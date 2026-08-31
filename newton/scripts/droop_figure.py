#!/usr/bin/env python3
"""Figure for the droop experiment: static_solve vs dynamic settle at two stretch stiffnesses.

Reuses the cable builder and drivers from ``beam_theory_check.py`` so the physics is
identical to the numeric sweep; this script only adds full-centerline capture and plotting.

The figure shows that, everything else held equal, increasing the stretch stiffness
increases the droop.

    python3 droop_figure.py --bend-stiffness 0.10 --bend-damping 0.01 --total-mass 0.022 --frames 6000 --iterations 20 --settle-tol 0.0005 --out droop_it20.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beam_theory_check import _import_newton, _make_solver, build_cable, rod_nodes  # noqa: E402

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light mode).
C_DYNAMIC = "#2a78d6"
C_STATIC = "#eb6834"
C_MUTED = "#8a8a85"
C_TEXT = "#0b0b0b"
C_SECONDARY = "#52514e"


# Both drivers return centerline *nodes*, not body COMs. The bodies sit at segment midpoints,
# so plotting them draws a polyline inset by half a segment at each end -- which reads as the
# cable being attached further right than it actually is, when in fact both span 0 to L.
def shape_dynamic(newton, model, rod_bodies, *, iterations, substeps, frames, frame_dt,
                  segment_length, settle_window):
    """Returns ``(nodes, swing)``.

    ``swing`` is the peak-to-peak movement [m] of any node over the last ``settle_window``
    frames -- an upper bound on any residual oscillation amplitude. Sampling *every* frame
    matters: checkpoints spaced k frames apart alias a residual swing whose period divides k
    to near-zero, reporting a false pass precisely when the cable is still ringing. Consecutive
    frames can only be fooled by a period shorter than 2 frames.

    This answers "has the motion stopped", which is a different question from "is the solver
    converged": §12 of ``static_solve_findings.md`` warns that an increment test cannot detect
    an unconverged *stiff* direction, because a large residual there produces a tiny
    displacement. That failure mode is covered separately by iteration doubling. What this
    catches is sampling mid-transient -- a real risk here, since at
    ``stretch=1e6`` the droop overshoots to 309 mm at frame 25 and only settles by frame ~400.
    """
    solver = _make_solver(newton, model, iterations=iterations, static_solve=False,
                          compliant_alm=True)
    s0, s1, ctl = model.state(), model.state(), model.control()
    sub_dt = frame_dt / substeps
    window_start = max(1, frames - settle_window + 1)
    samples = []
    for f in range(1, frames + 1):
        for _ in range(substeps):
            solver.step(s0, s1, ctl, None, sub_dt)
            s0, s1 = s1, s0
        if f >= window_start:
            samples.append(rod_nodes(s0.body_q.numpy(), rod_bodies, segment_length))
    nodes = samples[-1]
    stack = np.stack(samples)
    swing = float(np.max(stack.max(axis=0) - stack.min(axis=0))) if len(samples) > 1 else 0.0
    return nodes, swing


def shape_static(newton, model, rod_bodies, *, iterations, segment_length):
    solver = _make_solver(newton, model, iterations=iterations, static_solve=True,
                          compliant_alm=True)
    s0, s1, ctl = model.state(), model.state(), model.control()
    solver.step(s0, s1, ctl, None, 1.0 / 60.0)
    return rod_nodes(s1.body_q.numpy(), rod_bodies, segment_length)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--newton-path", default=None)
    p.add_argument("--num-elements", type=int, default=10)
    p.add_argument("--segment-length", type=float, default=0.05)
    p.add_argument("--radius", type=float, default=0.005)
    p.add_argument("--total-mass", type=float, default=0.175)
    p.add_argument("--bend-stiffness", type=float, default=10.0)
    p.add_argument("--bend-damping", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--frames", type=int, default=800)
    p.add_argument("--frame-dt", type=float, default=1.0 / 60.0)
    p.add_argument("--static-iterations", type=int, default=6000)
    p.add_argument("--stretch-stiffness", type=float, nargs="+", default=[1.0e4, 1.0e6])
    p.add_argument("--settle-window", type=int, default=100,
                   help="frames at the end of the settle over which to measure the residual "
                        "peak-to-peak swing")
    p.add_argument("--settle-tol", type=float, default=1.0e-4,
                   help="max acceptable peak-to-peak swing [m]; above this the run is flagged "
                        "as not settled (default 0.1 mm)")
    p.add_argument("--no-static", action="store_true",
                   help="omit the static_solve curve (and skip running it, which is most of "
                        "the cost at high --static-iterations)")
    p.add_argument("--out", default="droop_experiment.png")
    args = p.parse_args()

    wp, newton = _import_newton(args.newton_path)
    wp.init()
    device = wp.get_preferred_device()

    L = args.num_elements * args.segment_length

    results = []
    unsettled = []
    for ks in args.stretch_stiffness:
        kw = dict(num_elements=args.num_elements, segment_length=args.segment_length,
                  radius=args.radius, bend_stiffness=args.bend_stiffness,
                  bend_damping=args.bend_damping, stretch_stiffness=ks,
                  total_mass=args.total_mass, device=device)
        with wp.ScopedDevice(device):
            m, b = build_cable(newton, wp, **kw)
            dyn, swing = shape_dynamic(newton, m, b, iterations=args.iterations,
                                       substeps=args.substeps, frames=args.frames,
                                       frame_dt=args.frame_dt,
                                       segment_length=args.segment_length,
                                       settle_window=args.settle_window)
            sta = None
            if not args.no_static:
                m, b = build_cable(newton, wp, **kw)
                sta = shape_static(newton, m, b, iterations=args.static_iterations,
                                   segment_length=args.segment_length)
        settled = swing <= args.settle_tol
        results.append((ks, dyn, sta, swing, settled))
        msg = f"stretch={ks:.0e}  dynamic droop={1.0 - dyn[-1, 2]:.4f} m"
        if sta is not None:
            msg += f"  static droop={1.0 - sta[-1, 2]:.4f} m"
        msg += (f"  [swing over last {args.settle_window} frames: {swing * 1000:.4f} mm"
                f"{'' if settled else '  *** NOT SETTLED ***'}]")
        print(msg)
        if not settled:
            unsettled.append((ks, swing))

    if unsettled:
        print(f"\n*** {len(unsettled)} of {len(results)} run(s) had not settled after "
              f"{args.frames} frames (tol {args.settle_tol * 1000:.3f} mm):")
        for ks, swing in unsettled:
            print(f"      stretch={ks:.0e}: {swing * 1000:.4f} mm residual swing")
        print("    Raise --frames. Settle time depends strongly on stretch stiffness and")
        print("    bend damping (see static_solve_findings.md §13e); a value that suffices")
        print("    at 1e4 can be ~16x too short at 1e6.")

    plot(results, args, L)


def plot(results, args, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")

    lo = min(min([d[:, 2].min()] + ([] if s is None else [s[:, 2].min()]))
             for _, d, s, _, _ in results)

    for ax, (ks, dyn, sta, swing, settled) in zip(axes, results):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color=C_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=1)
        ax.text(L, 1.0 + 0.006, "as built (horizontal)", color=C_SECONDARY,
                fontsize=8.5, ha="right", va="bottom")

        ax.plot(dyn[:, 0], dyn[:, 2], color=C_DYNAMIC, lw=2.0, marker="o", ms=4.5,
                mec="#fcfcfb", mew=1.0, zorder=4, label=f"dynamic settle ({args.iterations} it x {args.substeps} substeps)")
        if sta is not None:
            ax.plot(sta[:, 0], sta[:, 2], color=C_STATIC, lw=2.0, marker="o", ms=4.5,
                    mec="#fcfcfb", mew=1.0, zorder=3,
                    label=f"static_solve ({args.static_iterations} it)")

        # When the two solvers agree their tip labels land on top of each
        # other, so stagger them vertically; when they disagree, leave them on their marks.
        close = sta is not None and abs(dyn[-1, 2] - sta[-1, 2]) < 0.015
        curves = [(dyn, C_DYNAMIC, +1)] + ([] if sta is None else [(sta, C_STATIC, -1)])
        for shape, color, sign in curves:
            droop = 1.0 - shape[-1, 2]
            txt = f"{droop * 1000:.1f} mm"
            ax.annotate(txt,
                        xy=(shape[-1, 0], shape[-1, 2]),
                        xytext=(6, sign * 9 if close else 0),
                        textcoords="offset points", color=color, fontsize=10,
                        va="center", ha="left", fontweight="bold")

        title = f"stretch_stiffness = {ks:.0e}"
        if not settled:
            # Mark it on the figure, not just the console -- a plot of an unsettled cable is
            # misleading on its own and these get pasted into notes without the log. The text
            # carries the meaning; colour is not the only signal.
            title += f"   [NOT SETTLED: {swing * 1000:.2f} mm swing]"
        ax.set_title(title, color=C_TEXT, fontsize=13, pad=10, loc="left")
        # Horizontal coordinate, not arc length: the tip foreshortens as the cable droops.
        ax.set_xlabel("horizontal distance x  [m]", color=C_SECONDARY, fontsize=10)
        ax.set_xlim(-0.02, L + 0.16)
        ax.set_ylim(lo - 0.03, 1.04)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d5d4cf")
        ax.tick_params(colors=C_SECONDARY, labelsize=9)

    axes[0].set_ylabel("height  [m]", color=C_SECONDARY, fontsize=10)
    # A single remaining series identifies itself from its tip label; a legend box would just
    # be noise.
    if len(axes[0].get_legend_handles_labels()[0]) > 1:
        axes[0].legend(loc="lower left", frameon=False, fontsize=9.5,
                       labelcolor=C_SECONDARY)

    fig.suptitle("Cable drop under gravity", color=C_TEXT, fontsize=14,
                 x=0.008, ha="left", y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
