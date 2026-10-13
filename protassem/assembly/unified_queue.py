"""Unified GLOBAL-ROUND assembly.

链和域统一按回旋半径排序、全局分轮：
- 一轮 = 把当前所有待处理项(pending 链 + available 域)按回旋半径走一遍。
- cc >= 当前阈值（同源相似已接受 -> 阈值 - domain_similar_relax）-> 立即接受 + 掩码。
- 同源组（还没接受过）：本轮最多试 2 个（怕首次没拟好），其余留到下一轮。
- 本轮没人达标 -> 在没达标候选里选 cc 最高（cc 相差 <= CA_TIEBREAK_MARGIN 时偏向 CA 多的一个）+ 降阈值。
- 阈值 0.45 -> 0.35（每轮 -0.015 前3 / -0.02）。
"""
import logging

from protassem.core.similarity import calculate_tm_score
from protassem.assembly.assembly_opt import select_round_end_candidate

log = logging.getLogger(__name__)

CA_TIEBREAK_MARGIN = 0.03   # 轮末无人达标：cc 相差 <= 此值的候选里偏向 CA 多的


def _accept_count(orch):
    return len(orch.accepted_chains) + len(orch.accepted_domain_pdbs)


def _group_has_accepted(orch, rec):
    """这条链的同源组里是否已经有整链接受（含"链姿态降域接受"）。"""
    gid = rec.get("group_id", rec["chain_id"])
    for other in orch.chain_records:
        if other is rec:
            continue
        if other.get("group_id", other["chain_id"]) != gid:
            continue
        if other["status"] in ("accepted_as_chain", "accepted_via_domains"):
            return True
    return False


def _release_stalled_retries(orch):
    """同源组既没成功者、也没有兄弟链还在试 -> 重试无望，放域、退回域拟合。

    没有这一步，等待重试的链会永远卡住，而它的域又一直被压着不放。
    """
    released = 0
    for rec in orch.chain_records:
        if rec["status"] != "chain_retry_pending":
            continue
        gid = rec.get("group_id", rec["chain_id"])
        # 2026-09-30 修复互等死锁：
        # 原判定把处于 chain_retry_pending 的兄弟也算作"还在试"，于是当同一同源组里
        # 所有链都待重试时，每条链都以为别人在推进，谁都不肯先放行 -> 全体互等。
        # 而"待重试"的链只有在组内已出现成功者时才有机会推进，可组内要有成功者又得先有链
        # 重试成功 —— 该条件永远不成立。因此这里只把仍处 pending（尚未尝试过）的兄弟
        # 视为"还能推进"；全是待重试时，一起放行回域装配。
        sibling_pending = any(
            other is not rec
            and other.get("group_id", other["chain_id"]) == gid
            and other["status"] == "pending"
            for other in orch.chain_records)
        if _group_has_accepted(orch, rec) or sibling_pending:
            continue
        cid = rec["chain_id"]
        rec["status"] = "needs_domain_assembly"
        if cid not in orch.needs_domain_assembly:
            orch.needs_domain_assembly.append(cid)
        released += 1
        log.info("链 %s: 同源组(%s)无成功者且无兄弟链在试 -> 放弃重试，域进池", cid, gid)
    return released


def _active_items(orch, recent_pdbs=None):
    """当前待处理项：pending 链 + available 域。

    **阶段严格分离（2026-09-30）**：先跑完全部链拟合，全部结束后才进入域拟合。

    原实现是"链与域混在一个队列里，按优先级逐轮调度"：任一链拟合完成，它的域
    立刻进池；而且"与上一轮接受域相似的域"会被排到所有待拟合链之前。结果就是
    链拟合与域拟合交织推进 —— 域名可能在**同源链尚未拟合**的情况下就消耗掉拟合
    预算，与"先链拟合，再逐个结构域拟合"的流程语义不符。

    现在：只要还有 pending 链，本函数只返回链；等链全部走完才放域进池。
    这样每个域都是在**全部链姿态都已确定**之后才被拟合，同源放宽阈值/掩码也
    都已按最终状态生效。
    """
    items = []
    pending_chain_ids = set()
    for rec in orch.chain_records:
        if rec["status"] == "pending":
            items.append(("chain", rec))
            pending_chain_ids.add(rec["chain_id"])
    # 等重试的链：同源组已经出过结果才重新排入（这时才吃得到放宽/分档阈值）。
    # 一并计入 pending_chain_ids，把它的域挡住，避免域拟合先吃掉。
    for rec in orch.chain_records:
        if rec["status"] == "chain_retry_pending" and _group_has_accepted(orch, rec):
            items.append(("chain", rec))
            pending_chain_ids.add(rec["chain_id"])

    # ---- 阶段一：还有链没拟合 -> 本轮只跑链，域一律不进池 ----
    if pending_chain_ids:
        items.sort(key=lambda x: -x[1]["gyration_radius"])
        return items

    # ---- 阶段二：全部链都已拟合过 -> 才放域进池 ----
    # 到这里 pending_chain_ids 必为空，父链条件自然满足。
    for doms in orch.domain_records.values():
        for d in doms:
            if d["status"] == "available":
                items.append(("domain", d))

    recent_pdbs = recent_pdbs or []
    _gof = getattr(orch, "domain_group_of", None) or {}
    recent_gids = set()
    for rp in recent_pdbs:
        g = _gof.get(rp)
        if g is not None:
            recent_gids.add(g)

    def _sim_to_recent(rec):
        gid = rec.get("group_id")
        if gid is not None and recent_gids:
            return 1 if gid in recent_gids else 0
        for rp in recent_pdbs:
            try:
                if calculate_tm_score(rec["pdb_file"], rp) >= orch.similarity_threshold:
                    return 1
            except Exception:
                pass
        return 0

    def keyf(x):
        kind, rec = x
        sim = _sim_to_recent(rec) if (kind == "domain" and recent_pdbs) else 0
        return (-sim, -rec["gyration_radius"])

    items.sort(key=keyf)
    return items


def run_unified_assembly(orch):
    from protassem.assembly.chain_fitter import fit_chain_item, ChainFitState
    from protassem.assembly.domain_fitter import fit_domain_once

    log.info("=" * 60)
    log.info("Unified GLOBAL-ROUND assembly (domain thr %.3f -> %.3f)",
             orch.initial_domain_threshold, orch.min_domain_threshold)
    log.info("=" * 60)

    chain_state = ChainFitState()
    # 预筛选已接受的链 -> 登记到相似组
    for rec in orch.chain_records:
        if rec["status"] == "accepted_as_chain":
            gid = rec.get("group_id", rec["chain_id"])
            chain_state.similar_groups[gid].append(rec)
            chain_state.accepted_groups[gid].append(rec)
            chain_state.accepted_chain_pdbs.append(rec["pdb_file"])

    # 单域链：直接当域处理（不单独链拟合、不域优化）
    for rec in orch.chain_records:
        if (rec["status"] == "pending" and not rec.get("is_complex")
                and len(orch.domain_records.get(rec["chain_id"], [])) == 1):
            rec["status"] = "routed_to_domain"
            if rec["chain_id"] not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(rec["chain_id"])
            for d in orch.domain_records[rec["chain_id"]]:
                d["status"] = "available"
            log.info("链 %s 单域 -> 直接域拟合", rec["chain_id"])

    threshold = orch.initial_domain_threshold
    decreases = 0
    round_num = 0
    last_accepted_pdbs = []   # 上一轮接受的域 pdb（轮间优先：其相似域下轮前排）

    while orch._target_has_points():
        _release_stalled_retries(orch)
        items = _active_items(orch, last_accepted_pdbs)
        if not items:
            break
        round_num += 1
        dom_before = len(orch.accepted_domain_pdbs)
        labels = []
        for t, r in items:
            if t == "chain":
                labels.append("C:%s" % r["chain_id"])
            else:
                labels.append("D:%s_d%d" % (r["chain_id"], r["domain_num"]))
        log.info("-" * 50)
        log.info("[Round %d] domain_thr=%.3f | %d 项: %s",
                 round_num, threshold, len(items), labels)

        any_accepted = False
        candidates = []          # 本轮没达标的域 drec（带 _rcc/_rpdb）
        group_reps = []          # 本轮"未接受同源组"的代表 pdb
        group_count = {}         # rep -> 本轮已试个数（2-tries）
        skipped_count = 0        # 本轮因 2-tries 被跳过(仍 available, 待下一轮)
        accepted_gids = set(orch.accepted_domain_groups)
        processed = 0            # 本轮实际处理的项数（链拟合/域拟合）

        for kind, rec in items:
            if not orch._target_has_points():
                log.info("[Round %d] 目标点云耗尽，停止", round_num)
                break

            if kind == "chain":
                if rec["status"] not in ("pending", "chain_retry_pending"):
                    continue
                before = _accept_count(orch)
                fit_chain_item(orch, rec, chain_state)
                processed += 1
                if _accept_count(orch) > before:
                    any_accepted = True
                    log.info("[Round %d] 链 %s 达标接受 -> 本轮结束, 下一轮",
                             round_num, rec["chain_id"])
                    break
                # 失败链已分域(available)，下一轮 _active_items 会捡到（按回旋半径交织）
                continue

            # ---------- domain ----------
            drec = rec
            if drec["status"] != "available":
                continue
            pdb = drec["pdb_file"]
            dkey = "%s_d%d" % (drec["chain_id"], drec["domain_num"])

            # 同源"未接受组"：本轮最多试 2 个（怕首次没拟好），其余留下一轮
            gid = drec.get("group_id")
            if gid is not None:
                if gid not in accepted_gids:
                    if group_count.get(gid, 0) >= 2:
                        log.info("  %s: 同源组(%s)本轮已试 2 个，跳过(留下一轮)", dkey, gid)
                        skipped_count += 1
                        continue
                    group_count[gid] = group_count.get(gid, 0) + 1
            elif not orch._is_similar_to_any(pdb, orch.accepted_domain_src_pdbs):
                rep = None
                for rp in group_reps:
                    if calculate_tm_score(pdb, rp) >= orch.similarity_threshold:
                        rep = rp
                        break
                if rep is not None and group_count.get(rep, 0) >= 2:
                    log.info("  %s: 同源组本轮已试 2 个，跳过(留下一轮)", dkey)
                    skipped_count += 1
                    continue
                if rep is None:
                    rep = pdb
                    group_reps.append(rep)
                    group_count[rep] = 0
                group_count[rep] += 1

            status, cc, fp = fit_domain_once(orch, drec, threshold, round_num)
            processed += 1
            if status == "accepted":
                any_accepted = True
                log.info("[Round %d] 域 %s 达标接受 -> 本轮结束, 下一轮",
                         round_num, dkey)
                break
            elif status == "candidate":
                drec["_rcc"] = cc
                drec["_rpdb"] = fp
                candidates.append(drec)

        # ---------- 轮末 ----------
        if any_accepted:
            log.info("[Round %d] 有达标，进入下一轮（阈值不变 %.3f）", round_num, threshold)
        elif candidates:
            cands = [{"cc": d["_rcc"], "pdb": d["_rpdb"], "drec": d,
                      "key": "%s_d%d" % (d["chain_id"], d["domain_num"])}
                     for d in candidates]
            chosen = select_round_end_candidate(cands, CA_TIEBREAK_MARGIN)
            log.info("[Round %d] 没人达标 -> 轮末选最高(cc≈则偏 CA 多): %s cc=%.4f",
                     round_num, chosen["key"], chosen["cc"])
            orch._accept_domain(chosen["drec"], chosen["pdb"], chosen["cc"])
            orch._mask_region(chosen["pdb"])
            old = threshold
            dec = 0.015 if decreases < 3 else 0.02
            threshold = max(orch.min_domain_threshold, threshold - dec)
            decreases += 1
            log.info("[Round %d] 降阈值 %.3f -> %.3f", round_num, old, threshold)
        elif skipped_count > 0:
            log.info("[Round %d] 无达标无候选，但有 %d 个被跳过(2-tries) -> 下一轮重试",
                     round_num, skipped_count)
            # 不 break、不降阈值；下一轮 group_count 重置后会拟这些被跳过的
        elif processed > 0:
            log.info("[Round %d] 本轮处理 %d 项均失败/被拒(无接受无候选) -> 进下一轮"
                     "(失败链的域已入池)", round_num, processed)
            # 不 break：失败链已分域、被拒域已剔除，while 会重新评估池
        else:
            log.info("[Round %d] 无可处理项，结束", round_num)
            break

        for d in candidates:
            d.pop("_rcc", None)
            d.pop("_rpdb", None)

        last_accepted_pdbs = orch.accepted_domain_src_pdbs[dom_before:]
        if last_accepted_pdbs:
            log.info("[Round %d] 本轮接受 %d 域 -> 下一轮其相似域优先前排",
                     round_num, len(last_accepted_pdbs))

    log.info("=" * 60)
    log.info("Unified assembly done: %d 轮, accepted=%d 组件",
             round_num, _accept_count(orch))
