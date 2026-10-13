
import sys
import os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO, Superimposer
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def get_parser_for_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pdb':
        return PDBParser(QUIET=True), 'pdb'
    elif ext == '.cif':
        return MMCIFParser(QUIET=True), 'cif'
    else:
        raise ValueError(f"Unsupported file format: {ext}; only .pdb and .cif are supported")


def get_io_for_format(output_file):
    ext = os.path.splitext(output_file)[1].lower()
    
    if ext == '.pdb':
        return PDBIO()
    elif ext == '.cif':
        return MMCIFIO()
    else:
        return PDBIO()


def get_sequence_and_ca(structure):
    sequence = ""
    ca_info = {}
    position = 0
    
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    aa = AA_MAP[res.get_resname()]
                    sequence += aa
                    resid = res.get_id()[1]
                    ca_info[position] = (resid, res["CA"])
                    position += 1
    
    return sequence, ca_info


def find_best_alignment(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-0.5):
    alignments = pairwise2.align.globalms(
        seq1, seq2,
        match_score, mismatch_penalty, gap_penalty, gap_penalty,
        one_alignment_only=True
    )
    
    if not alignments:
        raise ValueError('No valid sequence alignment found!')
    
    return alignments[0]


def get_aligned_residues(alignment, ca_info1, ca_info2):
    aligned_seq1, aligned_seq2, score, begin, end = alignment
    
    atoms1, atoms2 = [], []
    pos1, pos2 = 0, 0
    matched_pairs = []
    
    for i in range(len(aligned_seq1)):
        char1, char2 = aligned_seq1[i], aligned_seq2[i]
        
        if char1 != '-' and char2 != '-':
            if char1 == char2:
                if pos1 in ca_info1 and pos2 in ca_info2:
                    resid1, atom1 = ca_info1[pos1]
                    resid2, atom2 = ca_info2[pos2]
                    atoms1.append(atom1)
                    atoms2.append(atom2)
                    matched_pairs.append((resid1, resid2, char1))
        
        if char1 != '-':
            pos1 += 1
        if char2 != '-':
            pos2 += 1
    
    return atoms1, atoms2, matched_pairs


def calculate_and_align_with_sequence(structure_file1, structure_file2, output_file="aligned.pdb", 
                                    match_score=2, mismatch_penalty=-1, gap_penalty=-0.5):
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)
    
    print(f"Reading file 1: {structure_file1} (format: {format1.upper()})")
    print(f"Reading file 2: {structure_file2} (format: {format2.upper()})")

    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)

    seq1, ca_info1 = get_sequence_and_ca(structure1)
    seq2, ca_info2 = get_sequence_and_ca(structure2)
    
    print(f"Structure 1 sequence length: {len(seq1)}")
    print(f"Structure 2 sequence length: {len(seq2)}")
    print(f"Structure 1 sequence: {seq1}")
    print(f"Structure 2 sequence: {seq2}")
    print()

    print('Aligning sequences...')
    alignment = find_best_alignment(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
    
    print('Alignment results:')
    print(format_alignment(*alignment))
    
    atoms1, atoms2, matched_pairs = get_aligned_residues(alignment, ca_info1, ca_info2)
    
    if len(atoms1) == 0:
        raise ValueError('No matching residue pairs; cannot align!')
    
    print(f"Found {len(atoms1)} matched residue pairs:")
    for i, (resid1, resid2, aa) in enumerate(matched_pairs[:10]):
        print(f"  {resid1} ({aa}) <-> {resid2} ({aa})")
    if len(matched_pairs) > 10:
        print(f"  ... and {(len(matched_pairs) - 10)} matched pairs")
    print()

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    print(f"Sequence alignment uses {len(atoms1)} residues for superposition")
    print(f"Sequence similarity score: {alignment[2]:.1f}")
    print(f"RMSD = {sup.rms:.3f} Å")

    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)
    
    output_format = os.path.splitext(output_file)[1].upper()
    print(f"Aligned structure saved to: {output_file} (format: {output_format})")


def calculate_and_align_simple(structure_file1, structure_file2, output_file="aligned.pdb"):
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)
    
    print('Using simple residue-number matching...')
    print(f"Reading file 1: {structure_file1} (format: {format1.upper()})")
    print(f"Reading file 2: {structure_file2} (format: {format2.upper()})")
    
    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)
    
    atoms1, atoms2 = [], []
    
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for res1, res2 in zip(chain1, chain2):
                if res1.get_resname() in AA_MAP and "CA" in res1:
                    if res2.get_resname() in AA_MAP and "CA" in res2:
                        atoms1.append(res1["CA"])
                        atoms2.append(res2["CA"])
    
    if len(atoms1) == 0:
        raise ValueError('No matching residue pairs; cannot align!')
    
    print(f"Found {len(atoms1)} residue pairs")
    
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())
    
    print(f"Using {len(atoms1)} residues for superposition")
    print(f"RMSD = {sup.rms:.3f} Å")
    
    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)
    
    output_format = os.path.splitext(output_file)[1].upper()
    print(f"Aligned structure saved to: {output_file} (format: {output_format})")


def calculate_and_align_by_resid(structure_file1, structure_file2, output_file="aligned.pdb"):
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)

    print(f"Using residue-number matching (--resid)")
    print(f"Reading file 1 (reference): {structure_file1} (format: {format1.upper()})")
    print(f"Reading file 2 (moving): {structure_file2} (format: {format2.upper()})")

    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)

    ca_dict1 = {}
    for model in structure1:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    key = (chain.get_id(), res.get_id()[1])
                    ca_dict1[key] = res["CA"]

    ca_dict2 = {}
    for model in structure2:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    key = (chain.get_id(), res.get_id()[1])
                    ca_dict2[key] = res["CA"]

    print(f"Reference structure CA atoms: {len(ca_dict1)}")
    print(f"Moving structure CA atoms: {len(ca_dict2)}")

    resseq_set1 = {resseq for (_, resseq) in ca_dict1}
    resseq_set2 = {resseq for (_, resseq) in ca_dict2}
    common_resseqs = sorted(resseq_set1 & resseq_set2)

    if not common_resseqs:
        raise ValueError('No common residue numbers; cannot align!')

    atoms1, atoms2 = [], []
    resseq_to_atom1 = {}
    for (cid, resseq), atom in ca_dict1.items():
        if resseq not in resseq_to_atom1:
            resseq_to_atom1[resseq] = atom

    resseq_to_atom2 = {}
    for (cid, resseq), atom in ca_dict2.items():
        if resseq not in resseq_to_atom2:
            resseq_to_atom2[resseq] = atom

    for resseq in common_resseqs:
        atoms1.append(resseq_to_atom1[resseq])
        atoms2.append(resseq_to_atom2[resseq])

    print(f"Matched residues: {len(atoms1)}")
    print(f"Residue-number range: {common_resseqs[0]} - {common_resseqs[(-1)]}")

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    print(f"RMSD = {sup.rms:.3f} Å")

    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)

    output_format = os.path.splitext(output_file)[1].upper()
    print(f"Aligned structure saved to: {output_file} (format: {output_format})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python tr_rmsd.py structure1 structure2 [output] [--simple] [--resid]')
        print()
        print('Parameters:')
        print('  structure1: Reference structure (.pdb or .cif)')
        print('  structure2: Structure to align (.pdb or .cif)')
        print('  output:     Output path (.pdb or .cif, default: aligned.pdb)')
        print('  --simple:   Use simple residue-number matching (optional)')
        print('  --resid:    Pair by residue number for chain-subset domains (RMSD=0)')
        print()
        print('Example:')
        print("  python tr_rmsd.py model1.pdb model2.pdb")
        print("  python tr_rmsd.py model1.cif model2.cif aligned.cif")
        print("  python tr_rmsd.py model1.pdb model2.cif aligned.pdb")
        print("  python tr_rmsd.py model1.pdb model2.pdb aligned.pdb --simple")
        print("  python tr_rmsd.py fitted_chain.pdb domain.pdb aligned.pdb --resid")
        sys.exit(1)

    structure_file1 = sys.argv[1]
    structure_file2 = sys.argv[2]

    use_simple = "--simple" in sys.argv
    use_resid = "--resid" in sys.argv
    if use_simple:
        sys.argv.remove("--simple")
    if use_resid:
        sys.argv.remove("--resid")

    output_file = sys.argv[3] if len(sys.argv) > 3 else "aligned.pdb"

    try:
        if use_resid:
            calculate_and_align_by_resid(structure_file1, structure_file2, output_file)
        elif use_simple:
            calculate_and_align_simple(structure_file1, structure_file2, output_file)
        else:
            calculate_and_align_with_sequence(structure_file1, structure_file2, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print('\nTry --simple to use residue-number matching')
        sys.exit(1)