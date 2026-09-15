#!/usr/bin/env python3
"""Plot dispersion curves, port role of plot_aniso_TE2.m."""

from __future__ import annotations

import argparse

from aniso_safe.plotting import plot_slowness, plot_velocity


def _parse_floats(text):
    if text is None:
        return None
    return [float(v) for v in text.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser(description="Plot ANISO SAFE dispersion curves")
    parser.add_argument("--dir", required=True, help="Directory containing Results-*.npz or Results-*.mat")
    parser.add_argument("--out", default="dispersion_slowness.png", help="Output PNG for slowness")
    parser.add_argument("--velocity-out", default=None, help="Optional output PNG for phase velocity")
    parser.add_argument("--slowness-limits", nargs=2, type=float, default=None, help="Slowness range in us/ft")
    parser.add_argument("--freqs", default=None, help="Space/comma separated frequency list in kHz")
    args = parser.parse_args()

    freqs = _parse_floats(args.freqs)
    print("Plot_Aniso_TE2 Program has been started!")
    out = plot_slowness(args.dir, args.out, slowness_limits=args.slowness_limits, freqs=freqs)
    print(f"\tSaved: {out}")
    if args.velocity_out:
        out_v = plot_velocity(args.dir, args.velocity_out, freqs=freqs)
        print(f"\tSaved: {out_v}")


if __name__ == "__main__":
    main()
