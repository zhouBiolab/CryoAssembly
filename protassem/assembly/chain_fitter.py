"""Chain/complex fitting logic for the unified assembly queue."""

import os
import shutil
import logging
from collections import defaultdict

from protassem.core.structure import (
    pdb_to_cif, align_by_resid,
)
from protassem.core.scoring import calculate_cc_mask
from protassem.core.similarity import calculate_tm_score
from protassem.fitting.pipeline import run_fitting
from protassem.fitting.local_optimizer import local_optimize
from protassem.assembly.assembly_opt import homolog_threshold
from protassem.assembly.pose_guard import InitialPoseGuard

MAX_CHAIN_RETRY = 2

log = logging.getLogger(__name__)


class ChainFitState:
    """Tracks similarity groups and accepted chains during chain fitting."""

    def __init__(self):
        self.similar_groups = defaultdict(list)
        self.accepted_groups = defaultdict(list)
        self.accepted_chain_pdbs = []
        self.group_fail_count = defaultdict(int)
        self.accepted_group_ids = set()
        self.pose_guard = InitialPoseGuard()
        self.round_num = 0


def fit_chain_item(orch, rec, state):
    """Fit a single chain/complex.

    Returns list of domain records to add to queue, or None.
    """
    cid = rec["chain_id"]
    pdb = rec["pdb_file"]
    is_complex = rec.get("is_complex", False)

    group_id = rec.get("group_id")
    if group_id is None:
        group_id = orch._find_similar_group(pdb, state.similar_groups)
        if group_id is None:
            group_id = cid
    state.similar_groups[group_id].append(rec)

    hgid = group_id
    if (is_complex and hgid not in state.accepted_group_ids
            and state.group_fail_count.get(hgid, 0) >= 2):
        rec["status"] = "skipped_similar_failed"
        domains = [d for d in orch.domain_records.get(cid, [])
                   if d["status"] == "available"]
        if len(domains) >= 1:
            if cid not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(cid)
            log.info('Chain %s: homologous group(%s) failed chain fitting twice -> skip chain fitting and queue domains',
                     cid, hgid)
            return domains
        rec["status"] = "rejected_no_domain"
        return None

    if not orch._target_has_points():
        return None

    base_threshold = orch.complex_threshold if is_complex else orch.chain_threshold
    accepted_ccs = [r.get("cc_mask") for r in state.accepted_groups.get(group_id, [])
                    if r.get("cc_mask") is not None]
    threshold = homolog_threshold(base_threshold, orch.chain_similar_relax, accepted_ccs)
    pose_scope = None
    if not is_complex:
        if state.pose_guard.find_failed(orch, state.round_num, 'chain', group_id,
                                        rec, threshold):
            domains = [d for d in orch.domain_records.get(cid, [])
                       if d['status'] == 'available']
            rec['status'] = 'needs_domain_assembly' if domains else 'rejected_no_domain'
            rec['skip_reason'] = 'same_initial_pose'
            if domains and cid not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(cid)
            return domains or None
        try:
            pose_scope = state.pose_guard.scope(orch, state.round_num)
        except OSError:
            pass

    orch._ensure_work_files()
    orch._chain_iter += 1
    log.info("Chain fitting %d: %s%s", orch._chain_iter, cid,
             " (complex)" if is_complex else "")
    temp_chain_dir = orch.work_dir / "temp_chain" / ("temp_chain_%d" % orch._chain_iter)
    if temp_chain_dir.exists():
        shutil.rmtree(temp_chain_dir)
    os.makedirs(temp_chain_dir)
    shutil.copy2(rec["pdb_file"], temp_chain_dir)
    shutil.copy2(rec["txt_file"], temp_chain_dir)

    if accepted_ccs:
        log.info('Chain %s: homologous group has %d accepted members (min cc=%.4f) -> eff_thr=%.3f',
                 cid, len(accepted_ccs), min(accepted_ccs), threshold)

    fit_dir = orch.work_dir / "chain_fit" / ("chain_fit_%d" % orch._chain_iter)
    fit_result = run_fitting(
        orch.current_target_txt, str(temp_chain_dir),
        orch.current_density_mrc, orch.resolution, orch.contour,
        str(fit_dir), mode="chain",
        early_stop_threshold=threshold,
        original_density_mrc=orch.original_density_mrc,
        num_processes=orch.num_processes, batch_size=orch.batch_size,
        mask_radius_factor=orch.mask_radius_factor,
        min_point_distance_factor=orch.min_point_distance_factor,
        context=orch.context)

    if not fit_result["success"] or not fit_result.get("final_pdb"):
        rec["status"] = "failed"
        orch.failed_chain_pdbs.append(pdb)
        state.group_fail_count[hgid] += 1
        domains = [d for d in orch.domain_records.get(cid, [])
                   if d["status"] == "available"]
        if len(domains) >= 1:
            orch.needs_domain_assembly.append(cid)
            return domains
        rec["status"] = "rejected_no_domain"
        log.info("Chain %s: fitting failed, no domains available", cid)
        return None

    final_pdb = fit_result["final_pdb"]
    cc = fit_result["cc_mask"]
    rec["cc_mask"] = cc
    rec["fitted_pdb"] = final_pdb

    orch._save_attempt("chain_%s_iter%d" % (cid, orch._chain_iter), rec, cc, final_pdb)

    domains = orch.domain_records.get(cid, [])
    has_domains = len(domains) >= 1

    log.info('Chain %s fitted (cc=%.4f, chain_threshold=%.3f for reporting only)', cid, cc, threshold)

    success = False
    if len(domains) > 1 and orch.domain_opt:
        success = try_domains_via_chain_pose(orch, rec, final_pdb)

    if has_domains:
        if success:
            rec["status"] = "accepted_via_domains"
            state.accepted_groups[group_id].append(rec)
            state.accepted_chain_pdbs.append(pdb)
            state.accepted_group_ids.add(hgid)
        else:
            rec["status"] = "needs_domain_assembly"
            orch.failed_chain_pdbs.append(pdb)
            state.group_fail_count[hgid] += 1
            if pose_scope is not None:
                state.pose_guard.record_failed(orch, state.round_num, 'chain',
                                               group_id, rec, threshold, cc, pose_scope)
        orch.needs_domain_assembly.append(cid)
        return [d for d in domains if d["status"] == "available"]
    else:
        rec["status"] = "rejected_no_domain"
        orch.failed_chain_pdbs.append(pdb)
        log.info("Chain %s: no domains available, skipping", cid)
        return None


def try_domains_via_chain_pose(orch, chain_rec, fitted_chain_pdb):
    """When chain fails, transform its domains by chain pose and optimize."""
    cid = chain_rec["chain_id"]
    domains = orch.domain_records.get(cid, [])
    if not domains:
        return False

    transform_dir = orch.work_dir / ("chain_domain_transform_%s" % cid)
    os.makedirs(transform_dir, exist_ok=True)
    success_count = 0

    for drec in domains:
        if drec["status"] in ("accepted", "rejected"):
            continue
        dnum = drec["domain_num"]
        transformed = str(transform_dir / ("transformed_d%d.pdb" % dnum))
        if not align_by_resid(fitted_chain_pdb, drec["pdb_file"], transformed):
            continue

        init_cc = calculate_cc_mask(orch.current_density_mrc, transformed,
                                    orch.resolution, orch.contour)
        if init_cc <= 0.10:
            continue

        opt_pdb = str(transform_dir / ("optimized_d%d.pdb" % dnum))
        ok, opt_path, opt_cc = local_optimize(
            transformed, orch.current_density_mrc, opt_pdb,
            orch.resolution, orch.contour,
            initial_cc=init_cc, context=orch.context)
        if ok and opt_path:
            final_pdb, final_cc = opt_path, opt_cc
        else:
            final_pdb, final_cc = transformed, init_cc

        dthr = orch.initial_domain_threshold
        if orch._is_homolog_accepted_domain(drec):
            dthr = max(orch.min_domain_threshold,
                       orch.initial_domain_threshold - orch.domain_similar_relax)
        log.info("  chain_pose %s d%d: init=%.4f optimized=%.4f thr=%.4f -> %s",
                 cid, dnum, init_cc, opt_cc, dthr,
                 'Accepted' if final_cc >= dthr else 'Below threshold; switching to domain fitting')
        if final_cc >= dthr:
            orch._domain_iter += 1
            orch._accept_domain(drec, final_pdb, final_cc)
            orch._mask_region(final_pdb)
            success_count += 1
        elif final_cc > 0:
            drec["chain_pose_cc"] = final_cc
            drec["chain_pose_pdb"] = final_pdb

    return success_count > 0
