#!/usr/bin/env python3
"""General field-map plotting entry, port role of plot_aniso.m.

Field maps require FEMatrices eigenvector interpolation kernels. The spectrum
part is routed to the same loader used by plot_aniso_TE2.py.
"""

from __future__ import annotations

import argparse

from aniso_safe.plotting import plot_slowness


def main():
    parser = argparse.ArgumentParser(description="Plot ANISO SAFE maps and slowness")
    parser.add_argument("--dir", required=True, help="Directory containing result files")
    parser.add_argument("--slowness", choices=["y", "n"], default="y")
    parser.add_argument("--out", default="plot_aniso_slowness.png")
    parser.add_argument("--slowness-limits", nargs=2, type=float, default=None)
    args = parser.parse_args()
    print("Plot_Aniso Program has been started!")
    if args.slowness == "y":
        out = plot_slowness(args.dir, args.out, slowness_limits=args.slowness_limits)
        print(f"\tSaved: {out}")
    print("\tField maps are pending the FEMatrices interpolation kernels.")


if __name__ == "__main__":
    main()
