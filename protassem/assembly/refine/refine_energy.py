#!/usr/bin/env python3
'Score connectivity between domain segments.'

import os
import subprocess
import re
import glob
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence


def _display_connection_type(value):
    return {
        "紧密连接": "tight connection",
        "松散连接": "loose connection",
        "无效连接": "invalid connection",
    }.get(value, value)


@dataclass
class Atom:
    atom_id: int
    atom_name: str
    res_name: str
    chain_id: str
    res_id: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    element: str = ""
    charge: str = ""
    record_type: str = "ATOM"

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Residue:
    res_id: int
    res_name: str
    chain_id: str
    atoms: List[Atom]

    @property
    def ca_coord(self) -> Optional[np.ndarray]:
        for atom in self.atoms:
            if atom.atom_name.strip() == "CA":
                return atom.coord
        return None

    @property
    def has_ca(self) -> bool:
        return self.ca_coord is not None


@dataclass
class DomainSegment:
    start_res: int
    end_res: int
    start_ca: np.ndarray
    end_ca: np.ndarray
    start_res_name: str
    end_res_name: str


def read_structure_file(structure_file: str) -> Tuple[List[Atom], List[Residue]]:
    if not os.path.exists(structure_file):
        raise FileNotFoundError(f"Structure file does not exist: {structure_file}")
    
    file_ext = os.path.splitext(structure_file)[1].lower()
    
    if file_ext == '.pdb':
        return _read_pdb_file(structure_file)
    elif file_ext == '.cif':
        return _read_cif_file(structure_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}; only .pdb and .cif are supported")


def _read_pdb_file(pdb_file: str) -> Tuple[List[Atom], List[Residue]]:
    all_atoms = []
    residue_dict = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom = _parse_atom_line(line)
                if atom:
                    all_atoms.append(atom)
                    if atom.res_id not in residue_dict:
                        residue_dict[atom.res_id] = Residue(
                            res_id=atom.res_id,
                            res_name=atom.res_name,
                            chain_id=atom.chain_id,
                            atoms=[]
                        )
                    residue_dict[atom.res_id].atoms.append(atom)

    residues = list(residue_dict.values())
    residues.sort(key=lambda r: r.res_id)

    return all_atoms, residues


def _read_cif_file(cif_file: str) -> Tuple[List[Atom], List[Residue]]:
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        raise ImportError('BioPython is required for CIF input: pip install biopython')
    
    all_atoms = []
    residue_dict = {}
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', cif_file)
    
    atom_id = 1
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                    
                res_id = residue.id[1]
                res_name = residue.resname
                
                for atom_obj in residue:
                    atom_name = atom_obj.name
                    coord = atom_obj.coord
                    occupancy = atom_obj.occupancy
                    b_factor = atom_obj.bfactor
                    element = atom_obj.element
                    
                    atom = Atom(
                        atom_id=atom_id,
                        atom_name=atom_name,
                        res_name=res_name,
                        chain_id=chain_id,
                        res_id=res_id,
                        x=coord[0],
                        y=coord[1],
                        z=coord[2],
                        occupancy=occupancy,
                        b_factor=b_factor,
                        element=element,
                        charge="",
                        record_type="ATOM"
                    )
                    
                    all_atoms.append(atom)
                    
                    if res_id not in residue_dict:
                        residue_dict[res_id] = Residue(
                            res_id=res_id,
                            res_name=res_name,
                            chain_id=chain_id,
                            atoms=[]
                        )
                    residue_dict[res_id].atoms.append(atom)
                    
                    atom_id += 1
    
    residues = list(residue_dict.values())
    residues.sort(key=lambda r: r.res_id)
    
    return all_atoms, residues


def read_pdb_structure(pdb_file: str) -> Tuple[List[Atom], List[Residue]]:
    return read_structure_file(pdb_file)


def _parse_atom_line(line: str) -> Optional[Atom]:
    try:
        record_type = line[0:6].strip()
        atom_id = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21:22].strip() if len(line) > 21 else 'A'
        res_id = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = float(line[54:60].strip()) if len(line) > 54 and line[54:60].strip() else 1.0
        b_factor = float(line[60:66].strip()) if len(line) > 60 and line[60:66].strip() else 0.0
        element = line[76:78].strip() if len(line) > 76 else ""
        charge = line[78:80].strip() if len(line) > 78 else ""

        return Atom(
            atom_id=atom_id, atom_name=atom_name, res_name=res_name, chain_id=chain_id,
            res_id=res_id, x=x, y=y, z=z, occupancy=occupancy, b_factor=b_factor,
            element=element, charge=charge, record_type=record_type
        )
    except:
        return None


def get_domain_segments(structure_file: str) -> List[DomainSegment]:
    'Read ordered domain segments and terminal CA coordinates.'
    _, residues = read_structure_file(structure_file)

    ca_residues = [r for r in residues if r.has_ca]
    if not ca_residues:
        print(f"Warning: {structure_file} contains no CA atoms")
        return []

    ca_residues.sort(key=lambda r: r.res_id)

    segments = []
    segment_residues = [ca_residues[0]]

    for i in range(1, len(ca_residues)):
        if ca_residues[i].res_id == segment_residues[-1].res_id + 1:
            segment_residues.append(ca_residues[i])
        else:
            if segment_residues:
                segments.append(DomainSegment(
                    start_res=segment_residues[0].res_id,
                    end_res=segment_residues[-1].res_id,
                    start_ca=segment_residues[0].ca_coord,
                    end_ca=segment_residues[-1].ca_coord,
                    start_res_name=segment_residues[0].res_name,
                    end_res_name=segment_residues[-1].res_name
                ))
            segment_residues = [ca_residues[i]]

    if segment_residues:
        segments.append(DomainSegment(
            start_res=segment_residues[0].res_id,
            end_res=segment_residues[-1].res_id,
            start_ca=segment_residues[0].ca_coord,
            end_ca=segment_residues[-1].ca_coord,
            start_res_name=segment_residues[0].res_name,
            end_res_name=segment_residues[-1].res_name
        ))

    return segments



def calculate_segment_connection_energy(
        seg1: DomainSegment,
        seg2: DomainSegment,
        ideal_bond_distance: float = 3.8
) -> float:
    distance = np.linalg.norm(seg1.end_ca - seg2.start_ca)
    energy = (distance - ideal_bond_distance) ** 2

    print(f"    Segment connection: residue{seg1.end_res}({seg1.end_res_name}) -> residue{seg2.start_res}({seg2.start_res_name})")
    print(f"    CA-CA distance: {distance:.3f} Å, energy: {energy:.6f}")

    return energy


def _find_structure_files(directory: str, pattern: str = None) -> List[str]:
    structure_files = []
    
    pdb_files = glob.glob(os.path.join(directory, "*.pdb"))
    structure_files.extend(pdb_files)
    
    cif_files = glob.glob(os.path.join(directory, "*.cif"))
    structure_files.extend(cif_files)
    
    if pattern:
        structure_files = [f for f in structure_files if re.search(pattern, os.path.basename(f))]
    
    return structure_files


def calculate_chain_connection_energy_from_dir(
        structure_dir: str,
        ideal_bond_distance: float = 3.8,
        pattern: str = None
) -> float:
    if not os.path.exists(structure_dir):
        raise FileNotFoundError(f"Directory does not exist: {structure_dir}")

    if not os.path.isdir(structure_dir):
        raise NotADirectoryError(f"Path is not a directory: {structure_dir}")

    structure_files = _find_structure_files(structure_dir, pattern)

    if not structure_files:
        print(f"Warning: directory {structure_dir} contains no structure files (PDB or CIF)")
        return 0.0

    structure_files.sort()

    print(f"In directory {structure_dir} contains {len(structure_files)} structure files:")
    for i, structure_file in enumerate(structure_files, 1):
        print(f"  {i}. {os.path.basename(structure_file)}")
    print()

    if len(structure_files) <= 1:
        print('Warning: fewer than 2 structure files; no connection energy to compute')
        return 0.0

    return calculate_chain_connection_energy(structure_files, ideal_bond_distance)


def calculate_chain_connection_energy(
        structure_files: List[str],
        ideal_bond_distance: float = 3.8
) -> float:
    if isinstance(structure_files, str):
        if os.path.isdir(structure_files):
            return calculate_chain_connection_energy_from_dir(structure_files, ideal_bond_distance)
        else:
            structure_files = [structure_files]

    if len(structure_files) <= 1:
        return 0.0

    print("=" * 80)
    print('Compute whole-chain connection energy')
    print("=" * 80)

    all_segments = []

    for domain_idx, structure_file in enumerate(structure_files):
        segments = get_domain_segments(structure_file)
        if not segments:
            print(f"Warning: skipping file {structure_file} (no valid segments)")
            continue

        for seg in segments:
            all_segments.append({
                'domain_idx': domain_idx,
                'domain_file': structure_file,
                'segment': seg
            })

    if len(all_segments) < 2:
        print('Warning: fewer than 2 available segments')
        return 0.0

    all_segments.sort(key=lambda x: x['segment'].start_res)

    print(f"\nTotal: {len(all_segments)} segments (from {len(structure_files)} domains):")
    for i, seg_info in enumerate(all_segments):
        seg = seg_info['segment']
        domain_file = os.path.basename(seg_info['domain_file'])
        print(f"  Segment{(i + 1)}: residue {seg.start_res:4d}-{seg.end_res:4d} (from {domain_file})")

    print(f"\nComputing connection energies for adjacent segments:")
    total_energy = 0.0
    connection_count = 0

    for i in range(len(all_segments) - 1):
        seg1_info = all_segments[i]
        seg2_info = all_segments[i + 1]

        seg1 = seg1_info['segment']
        seg2 = seg2_info['segment']

        gap = seg2.start_res - seg1.end_res

        if gap <= 10:
            connection_count += 1
            print(f"\nConnection {connection_count}:")
            print(f"  From: {os.path.basename(seg1_info['domain_file'])} Segment {seg1.start_res}-{seg1.end_res}")
            print(f"  To: {os.path.basename(seg2_info['domain_file'])} Segment {seg2.start_res}-{seg2.end_res}")
            print(f"  Gap: {gap} residues")

            energy = calculate_segment_connection_energy(seg1, seg2, ideal_bond_distance)
            total_energy += energy
        else:
            print(f"\nSkipping connection (gap={gap} too large):")
            print(f"  Segment {seg1.start_res}-{seg1.end_res} -> {seg2.start_res}-{seg2.end_res}")

    print(f"\n" + "=" * 80)
    print(f"Computed {connection_count} connections")
    print(f"Total chain connection energy: {total_energy:.6f}")
    print("=" * 80)

    return total_energy



def calculate_chain_connection_score(
        structure_files: List[str],
        ideal_distance: float = 3.8,
        tight_tolerance: float = 1.5,
        loose_min: float = 5.3,
        loose_max: float = 20.0
) -> Tuple[float, Dict]:
    if isinstance(structure_files, str):
        if os.path.isdir(structure_files):
            structure_dir = structure_files
            structure_files = _find_structure_files(structure_dir)
            structure_files.sort()
        else:
            structure_files = [structure_files]

    if len(structure_files) <= 1:
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    print("=" * 80)
    print('Compute distance-based whole-chain connection score')
    print("=" * 80)
    print(f"Scoring rules:")
    print(f"  - Tight connection [{(ideal_distance - tight_tolerance):.1f}, {(ideal_distance + tight_tolerance):.1f}] Å: +1.0 points")
    print(f"  - Loose connection ({loose_min:.1f}, {loose_max:.1f}] Å: +0.1 points")
    print(f"  - Other distances: +0 points")
    print()

    all_segments = []

    for domain_idx, structure_file in enumerate(structure_files):
        segments = get_domain_segments(structure_file)
        if not segments:
            print(f"Warning: skipping file {structure_file} (no valid segments)")
            continue

        for seg in segments:
            all_segments.append({
                'domain_idx': domain_idx,
                'domain_file': structure_file,
                'segment': seg
            })

    if len(all_segments) < 2:
        print('Warning: fewer than 2 available segments')
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    all_segments.sort(key=lambda x: x['segment'].start_res)

    print(f"Total: {len(all_segments)} segments (from {len(structure_files)} domains):")
    for i, seg_info in enumerate(all_segments):
        seg = seg_info['segment']
        domain_file = os.path.basename(seg_info['domain_file'])
        print(f"  Segment{(i + 1)}: residue {seg.start_res:4d}-{seg.end_res:4d} (from {domain_file})")

    print(f"\nComputing connection scores for adjacent segments:")
    total_score = 0.0
    connection_count = 0
    tight_count = 0
    loose_count = 0
    no_score_count = 0

    connection_details = []

    for i in range(len(all_segments) - 1):
        seg1_info = all_segments[i]
        seg2_info = all_segments[i + 1]

        seg1 = seg1_info['segment']
        seg2 = seg2_info['segment']

        gap = seg2.start_res - seg1.end_res

        if gap <= loose_max:
            connection_count += 1

            distance = np.linalg.norm(seg1.end_ca - seg2.start_ca)

            tight_min = ideal_distance - tight_tolerance
            tight_max = ideal_distance + tight_tolerance

            if tight_min <= distance <= tight_max:
                score = 1.0
                connection_type = "紧密连接"
                tight_count += 1
            elif loose_min < distance <= loose_max:
                score = 0.1
                connection_type = "松散连接"
                loose_count += 1
            else:
                score = 0.0
                connection_type = "无效连接"
                no_score_count += 1

            total_score += score

            print(f"\nConnection {connection_count}:")
            print(f"  From: {os.path.basename(seg1_info['domain_file'])} Segment {seg1.start_res}-{seg1.end_res}")
            print(f"  To: {os.path.basename(seg2_info['domain_file'])} Segment {seg2.start_res}-{seg2.end_res}")
            print(f"  Gap: {gap} residues")
            print(f"  CA-CA distance: {distance:.3f} Å")
            print(f"  Type: {_display_connection_type(connection_type)}, score: {score:.1f}")

            connection_details.append({
                'from_file': os.path.basename(seg1_info['domain_file']),
                'from_res': f"{seg1.start_res}-{seg1.end_res}",
                'to_file': os.path.basename(seg2_info['domain_file']),
                'to_res': f"{seg2.start_res}-{seg2.end_res}",
                'gap': gap,
                'distance': distance,
                'type': connection_type,
                'score': score
            })
        else:
            print(f"\nSkipping connection (gap={gap} too large):")
            print(f"  Segment {seg1.start_res}-{seg1.end_res} -> {seg2.start_res}-{seg2.end_res}")

    print(f"\n" + "=" * 80)
    print(f"Statistics:")
    print(f"  Computed {connection_count} connections")
    print(f"  Tight connection ({tight_min:.1f}-{tight_max:.1f} Å): {tight_count} items")
    print(f"  Loose connection ({loose_min:.1f}-{loose_max:.1f} Å): {loose_count} items")
    print(f"  Invalid connections: {no_score_count} items")
    print(f"Total chain score: {total_score:.2f}")
    print("=" * 80)

    details = {
        'tight_connections': tight_count,
        'loose_connections': loose_count,
        'no_score_connections': no_score_count,
        'total_connections': connection_count,
        'connection_details': connection_details
    }

    return total_score, details


def calculate_chain_connection_score_from_dir(
        structure_dir: str,
        ideal_distance: float = 3.8,
        tight_tolerance: float = 1.5,
        loose_min: float = 5.3,
        loose_max: float = 12.0,
        pattern: str = None
) -> Tuple[float, Dict]:
    if not os.path.exists(structure_dir):
        raise FileNotFoundError(f"Directory does not exist: {structure_dir}")

    if not os.path.isdir(structure_dir):
        raise NotADirectoryError(f"Path is not a directory: {structure_dir}")

    structure_files = _find_structure_files(structure_dir, pattern)

    if not structure_files:
        print(f"Warning: directory {structure_dir} contains no structure files (PDB or CIF)")
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    structure_files.sort()

    print(f"In directory {structure_dir} contains {len(structure_files)} structure files:")
    for i, structure_file in enumerate(structure_files, 1):
        print(f"  {i}. {os.path.basename(structure_file)}")
    print()

    if len(structure_files) <= 1:
        print('Warning: fewer than 2 structure files; no connection score to compute')
        return 0.0, {
            'tight_connections': 0,
            'loose_connections': 0,
            'no_score_connections': 0,
            'total_connections': 0
        }

    return calculate_chain_connection_score(
        structure_files,
        ideal_distance,
        tight_tolerance,
        loose_min,
        loose_max
    )



def _is_valid_structure_file(filename: str) -> bool:
    if not (filename.endswith('.pdb') or filename.endswith('.cif')):
        return False
    
    basename = filename.rsplit('.', 1)[0]
    
    if re.search(r'_d_\d+$', basename):
        return True
    if re.search(r'_\d+$', basename):
        return False
    return True


def _find_pdb_files(folder_path: str) -> List[str]:
    structure_files = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if _is_valid_structure_file(file):
                structure_files.append(os.path.join(folder_path, file))
    return structure_files


def _extract_tm_scores(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    rmsd_match = re.search(r"RMSD=\s*([\d.]+)", output)
    tm1_match = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1", output)
    tm2_match = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2", output)

    if rmsd_match and tm1_match and tm2_match:
        return float(rmsd_match.group(1)), float(tm1_match.group(1)), float(tm2_match.group(1))
    return None, None, None


def compare_all_pdbs_in_dir(
        root_dir: str,
        usalign_path: str = "USalign",
        threshold: float = 0.5
):
    print("=" * 100)
    print(f"🔍 Scanning directory: {root_dir}")
    print("=" * 100)

    all_structure_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".pdb") or file.endswith(".cif"):
                all_structure_files.append(os.path.join(dirpath, file))

    all_structure_files.sort()
    n_files = len(all_structure_files)
    if n_files < 2:
        print(f"❌ Insufficient files (found {n_files} structure files)")
        return

    print(f"✅ Found {n_files} structure files (PDB/CIF), starting pairwise comparison...\n")

    total_comparisons = n_files * (n_files - 1) // 2
    counter = 0
    passed_pairs = []

    for i in range(n_files):
        for j in range(i + 1, n_files):
            counter += 1
            f1 = all_structure_files[i]
            f2 = all_structure_files[j]
            name1 = os.path.basename(f1)
            name2 = os.path.basename(f2)

            print(f"[{counter}/{total_comparisons}] Comparison: {name1} vs {name2}")

            cmd = [usalign_path, f1, f2, "-TMscore", "7", "-ter", "0"]
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
                rmsd, tm1, tm2 = _extract_tm_scores(output)
                if tm1 is None or tm2 is None:
                    print('⚠️ Cannot parse TM-score\n')
                    continue

                max_tm = max(tm1, tm2)
                if max_tm >= threshold:
                    passed_pairs.append((name1, name2, tm1, tm2, rmsd, max_tm))
                    print(f"   ✅ High-similarity match: TM1={tm1:.3f}  TM2={tm2:.3f}  RMSD={rmsd:.3f}  Max={max_tm:.3f}\n")
                else:
                    print(f"   TM1={tm1:.3f}  TM2={tm2:.3f}  RMSD={rmsd:.3f}  Max={max_tm:.3f}\n")

            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout: {name1} vs {name2}\n")
            except subprocess.CalledProcessError as e:
                print(f"❌ Alignment error: {e}\n")
            except Exception as e:
                print(f"❗ Exception: {e}\n")

    print("=" * 100)
    print(f"✅ Completed; compared {counter} structure pairs.")
    print(f"🎯 Threshold met (TM ≥ {threshold}) file pairs: {len(passed_pairs)}")
    print("=" * 100)

    if passed_pairs:
        print('\n📄 Highly similar file pairs:')
        for (n1, n2, tm1, tm2, rmsd, max_tm) in passed_pairs:
            print(f" - {n1} vs {n2} | TM1={tm1:.3f} TM2={tm2:.3f} RMSD={rmsd:.3f} Max={max_tm:.3f}")
    else:
        print('⚪ No file pair exceeds the threshold.')

    print("=" * 100)


def _calculate_tm_score(structure1: str, structure2: str, usalign_path: str) -> Tuple[
    Optional[float], Optional[float], Optional[float]]:
    cmd = [usalign_path, structure1, structure2, "-TMscore", "7", "-ter", "0"]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return _extract_tm_scores(output)
    except subprocess.CalledProcessError:
        return None, None, None


def cross_folder_tm_score_analysis(
        root_dir: str,
        threshold: float = 0.5,
        usalign_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "core", "USalign")
) -> List[Dict]:
    folder_info_map = {}

    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return []

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            structure_files = _find_pdb_files(folder_path)
            if structure_files:
                folder_info_map[folder_name] = structure_files

    if len(folder_info_map) < 2:
        print('At least 2 directories with valid structure files are required')
        return []

    above_threshold_results = []

    for folder1_name, folder2_name in combinations(folder_info_map.keys(), 2):
        structure_files1 = folder_info_map[folder1_name]
        structure_files2 = folder_info_map[folder2_name]

        for structure1 in structure_files1:
            for structure2 in structure_files2:
                rmsd, tm1, tm2 = _calculate_tm_score(structure1, structure2, usalign_path)

                if rmsd is not None and tm1 is not None and tm2 is not None:
                    max_tm_score = min(tm1, tm2)

                    if max_tm_score >= threshold:
                        above_threshold_results.append({
                            'folder1': folder1_name,
                            'folder2': folder2_name,
                            'pdb1': os.path.basename(structure1),
                            'pdb2': os.path.basename(structure2),
                            'max_tm_score': max_tm_score,
                            'rmsd': rmsd,
                            'tm_score_1': tm1,
                            'tm_score_2': tm2
                        })

    if above_threshold_results:
        print(f"\nFound {len(above_threshold_results)} pairs above threshold({threshold}) structures")
        above_threshold_results.sort(key=lambda x: x['max_tm_score'], reverse=True)
        for result in above_threshold_results:
            print(
                f"  {result['folder1']}/{result['pdb1']} <-> {result['folder2']}/{result['pdb2']}: TM={result['max_tm_score']:.3f}")
    else:
        print(f"No pair found above threshold({threshold}) structure pairs")

    return above_threshold_results



def example_usage(case_dir):

    print("\n" + "=" * 80)
    print('Example 1: connection energy (PDB and CIF supported)')
    print("-" * 80)

    test_dir = case_dir
    if os.path.exists(test_dir):
        total_energy = calculate_chain_connection_energy(test_dir)
        print(f"\nTotal energy: {total_energy:.6f}")
    else:
        print('Example directory does not exist, skipping...')

    print("\n" + "=" * 80)
    print('Example 2: distance-based connection score (PDB and CIF supported)')
    print("-" * 80)

    if os.path.exists(test_dir):
        score, details = calculate_chain_connection_score_from_dir(test_dir)
        print(f"\nTotal score: {score:.2f}")
        print(f"Tight connections: {details['tight_connections']}")
        print(f"Loose connections: {details['loose_connections']}")
        print(f"Invalid connections: {details['no_score_connections']}")
        print(f"Total connections: {details['total_connections']}")
    else:
        print('Example directory does not exist, skipping...')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Connection energy and score example')
    parser.add_argument("case_dir", help='Directory containing fitted domains')
    example_usage(parser.parse_args().case_dir)
