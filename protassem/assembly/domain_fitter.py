"""Domain fitting logic for the unified assembly queue."""

import logging

from protassem.fitting.pipeline import run_fitting
from protassem.assembly.assembly_opt import homolog_threshold

log = logging.getLogger(__name__)


def effective_domain_threshold(orch, drec, threshold):
    eff = threshold
    if orch._is_homolog_accepted_domain(drec):
        gid = drec.get("group_id")
        accepted_ccs = orch.accepted_domain_group_ccs.get(gid, []) if gid is not None else []
        eff = max(orch.min_domain_threshold,
                  homolog_threshold(threshold, orch.domain_similar_relax,
                                    accepted_ccs, low_band=False))
    return eff


def fit_domain_once(orch, drec, threshold, round_num):
    cid = drec["chain_id"]
    dnum = drec["domain_num"]
    dkey = "%s_d%d" % (cid, dnum)
    orch._ensure_work_files()
    orch._domain_iter += 1
    fit_dir = (orch.work_dir / "domain_fit" / ("round%02d" % round_num)
               / ("%s_d%d" % (cid, dnum)))
    eff = effective_domain_threshold(orch, drec, threshold)
    log.info('  [Domain fitting] %s | round %d | thr=%.3f%s', dkey, round_num, threshold,
             "" if eff == threshold else " (eff=%.3f)" % eff)
    fit_result = run_fitting(
        orch.current_target_txt, drec["txt_file"],
        orch.current_density_mrc, orch.resolution, orch.contour,
        str(fit_dir), mode="domain", chain_pdb=drec["pdb_file"],
        early_stop_threshold=eff,
        original_density_mrc=orch.original_density_mrc,
        num_processes=orch.num_processes, batch_size=orch.batch_size,
        mask_radius_factor=orch.mask_radius_factor,
        min_point_distance_factor=orch.min_point_distance_factor,
        context=orch.context)
    if not fit_result["success"] or not fit_result.get("final_pdb"):
        drec["status"] = "rejected"
        orch.failed_domain_pdbs.append(drec["pdb_file"])
        log.info('    %s has no fitting result -> rejected', dkey)
        return ("failed", 0.0, None)
    cc = fit_result["cc_mask"]
    fp = fit_result["final_pdb"]
    if cc >= eff:
        log.info('    %s meets threshold cc=%.4f >= %.4f -> accept and mask', dkey, cc, eff)
        orch._accept_domain(drec, fp, cc)
        orch._mask_region(fp)
        return ("accepted", cc, fp)
    attempts = drec.setdefault("_attempts", [])
    attempts.append((cc, fp))
    min_att = getattr(orch, "plateau_min_attempts", 3)
    eps = getattr(orch, "plateau_eps", 0.02)
    if len(attempts) >= min_att:
        recent = [a for a, _ in attempts[-min_att:]]
        if max(recent) - min(recent) < eps:
            chosen = next(((a, f) for a, f in sorted(attempts, key=lambda t: t[0], reverse=True)
                           if not orch._clashes_with_accepted(f)), None)
            if chosen is not None:
                bcc, bfp = chosen
                log.info('    %s plateau for %d rounds (Δ=%.4f<%.3f) -> accept historical best cc=%.4f (current %.4f, no excessive clash)',
                         dkey, min_att, max(recent) - min(recent), eps, bcc, cc)
                orch._accept_domain(drec, bfp, bcc)
                orch._mask_region(bfp)
                return ("accepted", bcc, bfp)
            log.info('    %s reached a plateau but all historical poses clash with accepted structures -> retain as candidate', dkey)
            return ("candidate", cc, fp)
    log.info('    %s below threshold cc=%.4f < %.4f -> round-end candidate (%d rounds tried)', dkey, cc, eff, len(attempts))
    return ("candidate", cc, fp)
