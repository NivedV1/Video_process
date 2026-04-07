#!/usr/bin/env python3
"""
Estimate optical trap stiffness from particle displacement data using equipartition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from particle_tracking_core import estimate_stiffness, load_dat_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate trap stiffness from tracked particle fluctuations."
    )
    parser.add_argument("dat_file", type=Path, help="Path to the particle .dat file.")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="Physical size of one pixel in micrometers (default: 1.0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Temperature in Celsius (default: 25.0 C).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, headers = load_dat_table(args.dat_file)
    results = estimate_stiffness(
        table=table,
        headers=headers,
        pixel_size_um=args.pixel_size,
        temperature_k=args.temperature + 273.15,
    )

    print(f"dat_file = {args.dat_file}")
    print(f"temperature_C = {args.temperature:.2f}")
    print(f"pixel_size_um = {args.pixel_size:.6f}")
    print()
    print("column\tvariance_px2\tvariance_um2\tk_N_per_m\tk_pN_per_um\tk_pN_per_nm")
    for row in results:
        print(
            f"{row['column']}\t{row['variance_px2']:.8f}\t{row['variance_um2']:.8f}\t"
            f"{row['k_n_per_m']:.6e}\t{row['k_pn_per_um']:.6f}\t{row['k_pn_per_nm']:.6f}"
        )


if __name__ == "__main__":
    main()
