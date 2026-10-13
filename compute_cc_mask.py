#!/usr/bin/env python3
'Compute structure-to-map masked correlation.'
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from protassem.core.scoring import calculate_cc_mask


def compute(structure, density_mrc, resolution, contour=0.0):
    return calculate_cc_mask(density_mrc, structure, resolution, contour)


def main():
    p = argparse.ArgumentParser(
        description='Compute structure-to-map CC_mask',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure", help='Structure file (.pdb or .cif)')
    p.add_argument("density_mrc", help='Density map file (.mrc)')
    p.add_argument("resolution", type=float, help='Resolution (Å)')
    p.add_argument("contour", type=float, nargs="?", default=0.0,
                   help='Density threshold (default: 0.0)')
    a = p.parse_args()

    for f in (a.structure, a.density_mrc):
        if not os.path.exists(f):
            print(f"Error: file does not exist - {f}")
            return 1

    cc = compute(a.structure, a.density_mrc, a.resolution, a.contour)
    print(f"CC_mask = {cc:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
