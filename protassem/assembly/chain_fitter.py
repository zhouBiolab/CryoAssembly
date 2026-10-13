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

MAX_CHAIN_RETRY = 2   # 同源组尚无成功者时，一条链最多保留重试的次数

log = logging.getLogger(__name__)


class ChainFitState:
    """Tracks similarity groups and accepted chains during chain fitting."""

    def __init__(self):
        self.similar_groups = defaultdict(list)
        self.accepted_groups = defaultdict(list)
        self.accepted_chain_pdbs = []
        self.group_fail_count = defaultdict(int)   # 链同源组 -> 链拟合失败次数
        self.accepted_group_ids = set()            # 已接受过链/部分域的同源组


def fit_chain_item(orch, rec, state):
    """Fit a single chain/complex.

    Returns list of domain records to add to queue, or None.
    """
    cid = rec["chain_id"]
    pdb = rec["pdb_file"]
    is_complex = rec.get("is_complex", False)

    # 相似分组：优先用预计算 group_id（与 _precompute_similarity 一致），缺失才回退运行时 USalign。
    group_id = rec.get("group_id")
    if group_id is None:
        group_id = orch._find_similar_group(pdb, state.similar_groups)
        if group_id is None:
            group_id = cid
    state.similar_groups[group_id].append(rec)

    hgid = group_id
    # 同源链组 2 次保底：本组还没接受过任何链时最多试 2 条；2 条都没接受
    # (链不达标且链域交织也没收到域) -> 跳过该组其余相似链，域进池(域拟合阶段再跑)。
    if (hgid not in state.accepted_group_ids
            and state.group_fail_count.get(hgid, 0) >= 2):
        rec["status"] = "skipped_similar_failed"
        domains = [d for d in orch.domain_records.get(cid, [])
                   if d["status"] == "available"]
        if len(domains) >= 1:
            if cid not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(cid)
            log.info("Chain %s: 同源组(%s)链拟合已失败 2 次 -> 跳过链拟合，域进池(域拟合再跑)",
                     cid, hgid)
            return domains
        rec["status"] = "rejected_no_domain"
        return None

    if not orch._target_has_points():
        return None

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

    base_threshold = orch.complex_threshold if is_complex else orch.chain_threshold

    # 同源组已有接受 -> 按组内【最小】cc 分档算有效阈值（assembly_opt.homolog_threshold）。
    # 早停阈值必须与它一致：高 CC 档有效阈值是 0.50，若仍按旧的 0.45 早停，
    # 候选流会在 0.45 被掐断，链拿着不到 0.50 的结果去比 0.50 -> 必然被拒。
    accepted_ccs = [r.get("cc_mask") for r in state.accepted_groups.get(group_id, [])
                    if r.get("cc_mask") is not None]
    threshold = homolog_threshold(base_threshold, orch.chain_similar_relax, accepted_ccs)
    if accepted_ccs:
        log.info("Chain %s: 同源组已接受 %d 条(min cc=%.4f) -> eff_thr=%.3f",
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

    # 2026-09-30（只接受达标结构域）：链不再是接受单位。
    # 原来的 cc >= threshold 整链接受分支已删除 —— 无论链 cc 多少，都只把这条链
    # 当作"域姿态的来源"：逐域局部优化，只接受 cc 达标的域。
    # chain_threshold 退化为纯记录值（仍由 assembly_summary.txt 打印）。
    log.info("Chain %s fitted (cc=%.4f, chain_threshold=%.3f 仅记录)", cid, cc, threshold)

    success = False
    # 单域链不走链姿态：它的"链姿态"就是域自己的姿态，再套一次是重复劳动，
    # 直接让域进队列、和其他结构域一起排序拟合即可。
    if len(domains) > 1 and orch.domain_opt:
        success = try_domains_via_chain_pose(orch, rec, final_pdb)

        # 2026-09-30 撤除"保留重试、域暂不放"分支。
        # 它有两个致命副作用：
        #   1) 同源组里所有链都进这个分支时，_release_stalled_retries 的"还有兄弟在试"
        #      判定恒为真 -> 互等死锁，谁的域都放不出去；
        #   2) 这个分支的 return None 在 domain_opt 开关之前，导致
        #      try_domains_via_chain_pose（链姿态逐域局部优化）从未执行，
        #      两个实测任务里它的产出目录 chain_domain_transform_* 一个都没有。
        # 而且原始设计本就不需要它：链被拒后走 domain_opt 逐域局部优化，
        # 达标的域被接受 -> rec 记为 accepted_via_domains（相当于链也达标），
        # 同源组随即有了成功者，相似的链会按 base - relax 的放宽阈值继续拟合。
        # 逐域优化没成功时，下面的 else 分支把域放进 needs_domain_assembly，
        # 转由域拟合承接 —— 这就是原设计自带的回退机制。
    if has_domains:
        if success:
            # 有域达标 <=> 这条链能拟合：同源组记一笔，相似的链照常先做链拟合
            rec["status"] = "accepted_via_domains"
            state.accepted_groups[group_id].append(rec)
            state.accepted_chain_pdbs.append(pdb)
            state.accepted_group_ids.add(hgid)
        else:
            # 一个域都没达标（或单域链）-> 这条链不适合，域交给域拟合
            rec["status"] = "needs_domain_assembly"
            orch.failed_chain_pdbs.append(pdb)
            state.group_fail_count[hgid] += 1
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
        # 2026-09-30：逐域 cc 必须落日志，否则事后无从判断某个域为何被收/被退
        log.info("  chain_pose %s d%d: init=%.4f optimized=%.4f thr=%.4f -> %s",
                 cid, dnum, init_cc, opt_cc, dthr,
                 "接受" if final_cc >= dthr else "未达标转域拟合")
        if final_cc >= dthr:
            orch._domain_iter += 1
            orch._accept_domain(drec, final_pdb, final_cc)
            orch._mask_region(final_pdb)
            success_count += 1
        elif final_cc > 0:
            drec["chain_pose_cc"] = final_cc
            drec["chain_pose_pdb"] = final_pdb

    return success_count > 0
