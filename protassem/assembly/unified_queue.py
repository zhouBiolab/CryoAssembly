import logging

from protassem.core.similarity import calculate_tm_score
from protassem.assembly.assembly_opt import select_round_end_candidate

log = logging.getLogger(__name__)

CA_TIEBREAK_MARGIN = 0.03


def _accept_count(orch):
    return len(orch.accepted_chains) + len(orch.accepted_domain_pdbs)


def _group_has_accepted(orch, rec):
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
    released = 0
    for rec in orch.chain_records:
        if rec["status"] != "chain_retry_pending":
            continue
        gid = rec.get("group_id", rec["chain_id"])
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
        log.info('Chain %s: homologous group(%s) has no successful member or active sibling -> stop retries and queue domains', cid, gid)
    return released


def _active_items(orch, recent_pdbs=None):
    items = []
    pending_chain_ids = set()
    for rec in orch.chain_records:
        if rec["status"] == "pending":
            items.append(("chain", rec))
            pending_chain_ids.add(rec["chain_id"])
    for rec in orch.chain_records:
        if rec["status"] == "chain_retry_pending" and _group_has_accepted(orch, rec):
            items.append(("chain", rec))
            pending_chain_ids.add(rec["chain_id"])

    if pending_chain_ids:
        items.sort(key=lambda x: -x[1]["gyration_radius"])
        return items

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
    for rec in orch.chain_records:
        if rec["status"] == "accepted_as_chain":
            gid = rec.get("group_id", rec["chain_id"])
            chain_state.similar_groups[gid].append(rec)
            chain_state.accepted_groups[gid].append(rec)
            chain_state.accepted_chain_pdbs.append(rec["pdb_file"])

    for rec in orch.chain_records:
        if (rec["status"] == "pending" and not rec.get("is_complex")
                and len(orch.domain_records.get(rec["chain_id"], [])) == 1):
            rec["status"] = "routed_to_domain"
            if rec["chain_id"] not in orch.needs_domain_assembly:
                orch.needs_domain_assembly.append(rec["chain_id"])
            for d in orch.domain_records[rec["chain_id"]]:
                d["status"] = "available"
            log.info('Chain %s has one domain -> fit domain directly', rec["chain_id"])

    threshold = orch.initial_domain_threshold
    decreases = 0
    round_num = 0
    last_accepted_pdbs = []

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
        log.info('[Round %d] domain_thr=%.3f | %d items: %s',
                 round_num, threshold, len(items), labels)

        any_accepted = False
        candidates = []
        group_reps = []
        group_count = {}
        skipped_count = 0
        accepted_gids = set(orch.accepted_domain_groups)
        processed = 0

        for kind, rec in items:
            if not orch._target_has_points():
                log.info('[Round %d] target point cloud exhausted, stopping', round_num)
                break

            if kind == "chain":
                if rec["status"] not in ("pending", "chain_retry_pending"):
                    continue
                before = _accept_count(orch)
                fit_chain_item(orch, rec, chain_state)
                processed += 1
                if _accept_count(orch) > before:
                    any_accepted = True
                    log.info('[Round %d] chain %s accepted -> end round and advance',
                             round_num, rec["chain_id"])
                    break
                continue

            # ---------- domain ----------
            drec = rec
            if drec["status"] != "available":
                continue
            pdb = drec["pdb_file"]
            dkey = "%s_d%d" % (drec["chain_id"], drec["domain_num"])

            gid = drec.get("group_id")
            if gid is not None:
                if gid not in accepted_gids:
                    if group_count.get(gid, 0) >= 2:
                        log.info('  %s: homologous group(%s) already tried 2 members this round, defer to next round', dkey, gid)
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
                    log.info('  %s: homologous group already tried 2 members this round, defer to next round', dkey)
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
                log.info('[Round %d] domain %s accepted -> end round and advance',
                         round_num, dkey)
                break
            elif status == "candidate":
                drec["_rcc"] = cc
                drec["_rpdb"] = fp
                candidates.append(drec)

        if any_accepted:
            log.info('[Round %d] threshold reached, advance without changing threshold %.3f', round_num, threshold)
        elif candidates:
            cands = [{"cc": d["_rcc"], "pdb": d["_rpdb"], "drec": d,
                      "key": "%s_d%d" % (d["chain_id"], d["domain_num"])}
                     for d in candidates]
            chosen = select_round_end_candidate(cands, CA_TIEBREAK_MARGIN)
            log.info('[Round %d] threshold not reached -> select best round-end candidate (prefer more CA atoms for similar CC): %s cc=%.4f',
                     round_num, chosen["key"], chosen["cc"])
            orch._accept_domain(chosen["drec"], chosen["pdb"], chosen["cc"])
            orch._mask_region(chosen["pdb"])
            old = threshold
            dec = 0.015 if decreases < 3 else 0.02
            threshold = max(orch.min_domain_threshold, threshold - dec)
            decreases += 1
            log.info('[Round %d] lower threshold %.3f -> %.3f', round_num, old, threshold)
        elif skipped_count > 0:
            log.info('[Round %d] no accepted item or candidate, but %d skipped by 2-tries -> retry next round',
                     round_num, skipped_count)
        elif processed > 0:
            log.info('[Round %d] all %d items failed/rejected (no acceptance or candidate) -> next round with queued domains', round_num, processed)
        else:
            log.info('[Round %d] no processable items, stopping', round_num)
            break

        for d in candidates:
            d.pop("_rcc", None)
            d.pop("_rpdb", None)

        last_accepted_pdbs = orch.accepted_domain_src_pdbs[dom_before:]
        if last_accepted_pdbs:
            log.info('[Round %d] accepted %d domains -> prioritize similar domains next round',
                     round_num, len(last_accepted_pdbs))

    log.info("=" * 60)
    log.info('Unified assembly done: %d rounds, accepted=%d components',
             round_num, _accept_count(orch))
