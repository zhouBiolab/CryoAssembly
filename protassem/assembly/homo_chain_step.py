import os
import logging

from protassem.assembly.homo_chain_refine import run_homo_chain_refine

log = logging.getLogger("protassem.homo_chain")


def maybe_homo_refine(orch, input_complex):
    """Returns the homo-chain-refined complex path, or None if skipped/unchanged."""
    if not getattr(orch, "homo_chain_refine", False):
        return None
    if not input_complex or not os.path.exists(input_complex):
        log.info("Step 5 (homo-chain-refine) skipped: no input complex")
        return None

    log.info("=" * 60)
    log.info("Step 5: homologous-chain refine")
    log.info("=" * 60)

    domain_ranges = {}
    for cid, val in getattr(orch, "domain_adjacency", {}).items():
        ranges = val[0] if isinstance(val, (tuple, list)) else None
        if ranges:
            domain_ranges[cid] = ranges

    out = os.path.join(str(orch.final_dir), "homo_chain_refined_complex.cif")
    workdir = os.path.join(str(orch.work_dir), "homo_chain")
    try:
        result = run_homo_chain_refine(
            input_complex, orch.original_density_mrc, orch.resolution, orch.contour,
            out, workdir, domain_ranges=domain_ranges,
            nproc=getattr(orch, "num_processes", 8), log=log.info)
    except Exception as e:
        log.warning("Step 5 (homo-chain-refine) failed: %s", e)
        return None

    if result and os.path.exists(result):
        log.info("Step 5 complete. Primary: %s", result)
        log.info("Backup (pre-homo): %s", input_complex)
        return result
    log.info("Step 5 produced no change")
    return None
