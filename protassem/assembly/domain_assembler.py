
import os
import logging
import shutil

from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model, Chain

from protassem.core.scoring import calculate_cc_mask
from protassem.core.structure import pdb_to_cif, write_structure_with_chain_map

log = logging.getLogger(__name__)


def chain_map_of(orch, chain_id):
    record = orch.chain_record(chain_id)
    return (record.get("chain_map") or {}) if record else {}


def _restore_chain_ids(orch, chain_id, cif_path, source_cid=None):
    chain_map = chain_map_of(orch, chain_id)
    if not chain_map:
        if str(cif_path).lower().endswith(".cif"):
            return cif_path
        out_path = str(orch.work_dir
                       / ("final_%s.cif" % os.path.basename(cif_path).rsplit(".", 1)[0]))
        pdb_to_cif(cif_path, out_path, chain_id=None)
        return out_path
    if source_cid is not None:
        real_cid = chain_map.get(source_cid)
        if real_cid is None:
            log.warning("Chain %s: source chain %s not in chain_map %s; "
                        "keeping the id as-is", chain_id, source_cid, chain_map)
            return cif_path
        mapping = {source_cid: real_cid}
    else:
        mapping = chain_map
    out_path = str(orch.work_dir / ("final_%s.cif" % os.path.basename(cif_path).rsplit(".", 1)[0]))
    write_structure_with_chain_map(cif_path, mapping, out_path)
    return out_path


def assemble_domain_chains(orch):
    for cid, recs in orch.domain_records.items():
        if any(r.get("status") == "accepted" for r in recs):
            if cid not in orch.needs_domain_assembly:
                log.warning('Chain %s: accepted domains found outside assembly list; adding chain', cid)
                orch.needs_domain_assembly.append(cid)

    if not orch.needs_domain_assembly:
        _acc = sum(1 for recs in orch.domain_records.values()
                   for r in recs if r.get("status") == "accepted")
        if _acc:
            log.warning('No chain entered domain assembly, but %d domains were accepted and will not enter the complex', _acc)
        else:
            log.info('No chain requires domain assembly')
        return

    for cid in dict.fromkeys(orch.needs_domain_assembly):
        chain_rec = next((c for c in orch.chain_records
                          if c["chain_id"] == cid), None)
        if not chain_rec:
            continue

        fitted = [d for d in orch.domain_records.get(cid, [])
                  if d["status"] == "accepted"]
        if not fitted:
            log.info("Chain %s: no domain passed domain-min-cc; chain dropped",
                     cid)
            continue

        if len(fitted) == 1:
            _handle_single_domain(orch, cid, chain_rec, fitted[0])
            continue

        is_complex = chain_rec.get("is_complex", False)
        ranges, _ = orch.domain_adjacency.get(cid, ({}, {}))
        assembled_cif = merge_domains(orch, cid, fitted, ranges,
                                      is_complex=is_complex)
        if not assembled_cif:
            log.info("Chain %s: domain merge failed; chain dropped", cid)
            continue

        domain_cc = calculate_cc_mask(orch.original_density_mrc, assembled_cif,
                                      orch.resolution, orch.contour)
        log.info("Chain %s: assembled from %d domains (cc=%.4f)",
                 cid, len(fitted), domain_cc)
        final_cif = _restore_chain_ids(orch, cid, assembled_cif)
        filtered_cif = _filtered_domain_cif(orch, cid, fitted, ranges,
                                            assembled_cif,
                                            is_complex=is_complex)
        if filtered_cif is None:
            filtered_final = None
        elif os.path.abspath(filtered_cif) == os.path.abspath(assembled_cif):
            filtered_final = final_cif
        else:
            filtered_final = _restore_chain_ids(orch, cid, filtered_cif)
        _save_domain_chain(orch, cid, final_cif, domain_cc, fitted,
                           filtered_final)


def merge_domains(orch, chain_id, fitted_domains, domain_ranges,
                  is_complex=False, out_cif=None):
    out_cif = out_cif or str(orch.work_dir / ("assembled_%s.cif" % chain_id))
    try:
        segments = []
        for drec in fitted_domains:
            dnum = drec["domain_num"]
            if dnum in domain_ranges:
                for s, e in domain_ranges[dnum]:
                    segments.append({"start": s, "end": e, "drec": drec})
        segments.sort(key=lambda x: x["start"])

        struct = Structure.Structure("assembled")
        model_obj = Model.Model(0)

        if is_complex:
            chains_map = {}
            serial = 1
            for seg in segments:
                cif = seg["drec"].get("fitted_cif")
                if not cif or not os.path.exists(cif):
                    continue
                src_cid = seg["drec"].get("source_chain_id", chain_id)
                if src_cid not in chains_map:
                    chains_map[src_cid] = Chain.Chain(src_cid)
                chain_obj = chains_map[src_cid]
                parser = MMCIFParser(QUIET=True)
                ds = parser.get_structure("d", cif)
                for m in ds:
                    for ch in m:
                        for res in ch:
                            if seg["start"] <= res.get_id()[1] <= seg["end"]:
                                nr = res.copy()
                                for atom in nr:
                                    atom.serial_number = serial
                                    serial += 1
                                chain_obj.add(nr)
            for ch in chains_map.values():
                model_obj.add(ch)
        else:
            chain_obj = Chain.Chain(chain_id)
            serial = 1
            for seg in segments:
                cif = seg["drec"].get("fitted_cif")
                if not cif or not os.path.exists(cif):
                    continue
                parser = MMCIFParser(QUIET=True)
                ds = parser.get_structure("d", cif)
                for m in ds:
                    for ch in m:
                        for res in ch:
                            if seg["start"] <= res.get_id()[1] <= seg["end"]:
                                nr = res.copy()
                                for atom in nr:
                                    atom.serial_number = serial
                                    serial += 1
                                chain_obj.add(nr)

            if len(list(chain_obj)) == 0:
                log.warning("merge_domains: empty merge for chain %s "
                            "(ranges=%s), skipping", chain_id, domain_ranges)
                return None
            model_obj.add(chain_obj)

        struct.add(model_obj)
        io = MMCIFIO()
        io.set_structure(struct)
        io.save(out_cif)
        return out_cif
    except Exception as e:
        log.error("merge_domains error: %s", e)
        return None


def _filtered_domain_cif(orch, chain_id, fitted_domains, ranges, full_cif,
                         is_complex=False):
    thr = orch.complex_min_cc
    kept = [d for d in fitted_domains if d.get("cc_mask", 0.0) >= thr]
    if len(kept) == len(fitted_domains):
        return full_cif
    if not kept:
        return None
    out = str(orch.work_dir / ("assembled_%s_filtered.cif" % chain_id))
    return merge_domains(orch, chain_id, kept, ranges, out_cif=out,
                         is_complex=is_complex)


def _handle_single_domain(orch, chain_id, chain_rec, domain_rec):
    d_file = domain_rec.get("fitted_cif") or domain_rec.get("fitted_pdb")
    if not d_file or not os.path.exists(d_file):
        log.info("Chain %s: single domain file missing; chain dropped",
                 chain_id)
        return
    d_cc = calculate_cc_mask(orch.original_density_mrc, d_file,
                             orch.resolution, orch.contour)
    source = domain_rec.get("fitted_pdb") or d_file
    final_file = _restore_chain_ids(orch, chain_id, source,
                                    source_cid=domain_rec.get("source_chain_id"))
    filtered = (final_file if domain_rec.get("cc_mask", d_cc) >= orch.complex_min_cc
                else None)
    _save_domain_chain(orch, chain_id, final_file, d_cc, [domain_rec], filtered)


def _save_domain_chain(orch, chain_id, cif_file, cc, fitted_domains,
                       filtered_cif=None):
    orch._domain_chain_order += 1
    base = "domain_chain_%s_%02d" % (chain_id, orch._domain_chain_order)
    dst = str(orch.final_dir / "domain_chains" / (base + ".cif"))
    shutil.copy2(cif_file, dst)

    if filtered_cif is None:
        filtered_dst = None
    elif os.path.abspath(filtered_cif) == os.path.abspath(cif_file):
        filtered_dst = dst
    else:
        filtered_dst = str(orch.final_dir / "domain_chains"
                           / (base + "_filtered.cif"))
        shutil.copy2(filtered_cif, filtered_dst)

    orch.accepted_chains.append({
        "type": "domain_chain", "chain_id": chain_id,
        "fitting_order": orch._domain_chain_order,
        "cc_mask": cc, "fitted_cif": dst, "fitted_cif_filtered": filtered_dst,
        "domains": [d["domain_num"] for d in fitted_domains],
        "domain_details": [{"num": d["domain_num"],
                            "cc_mask": d.get("cc_mask")}
                           for d in fitted_domains],
    })
