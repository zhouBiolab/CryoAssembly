# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class AssemblyOptConfig:
    chain_similar_relax: float = 0.03
    domain_similar_relax: float = 0.025
    clash_overlap_thr: float = 0.10


HIGH_CC_TIER: float = 0.55
HIGH_CC_RELAX: float = 0.05


def homolog_threshold(base, relax, accepted_ccs, *, low_band=True):
    'Select the fitting threshold for a homologous group.'
    if not accepted_ccs:
        return base
    a = min(accepted_ccs)
    if a >= HIGH_CC_TIER:
        return HIGH_CC_TIER - HIGH_CC_RELAX
    if not low_band:
        return base - relax
    if a >= base + relax:
        return base
    return a - relax


def ca_count(pdb):
    n = 0
    try:
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() in ("CA", "P"):
                    n += 1
    except Exception:
        pass
    return n


def ca_overlap(pdb_a, pdb_b):
    from protassem.core import check_clash
    return check_clash.calculate_overlap_ratio_numpy(pdb_a, pdb_b, clash_distance=3.0)


def select_round_end_candidate(candidates, ca_margin):
    if not candidates:
        return None
    best_cc = max(c["cc"] for c in candidates)
    near = [c for c in candidates if best_cc - c["cc"] <= ca_margin]
    return max(near, key=lambda c: ca_count(c["pdb"]))
