import os
import sys
import re
import glob
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse


def convert_pdb_to_cif(pdb_file: str, cif_file: str):
    try:
        from Bio.PDB import PDBParser, MMCIFIO
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_file)
        
        logging.info(f"    ✅ Converted: {os.path.basename(pdb_file)} -> {os.path.basename(cif_file)}")
        return True
    except Exception as e:
        logging.warning(f"    ⚠️  PDB-to-CIF conversion failed ({os.path.basename(pdb_file)}): {e}")
        return False


class DomainComplexAssembler:

    def __init__(self, input_dir, output_filename=None):
        self.input_dir = Path(input_dir)
        self.output_filename = output_filename or "assembled_complex.cif"
        self.domain_files = []
        self.temp_cif_dir = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _extract_chain_from_folder_name(self, folder_path):
        folder_name = os.path.basename(str(folder_path))

        patterns = [
            r'chain[_-]?([A-Za-z0-9]+)',
            r'protein[_-]?([A-Za-z0-9]+)',
            r'domain[_-]?([A-Za-z0-9]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                chain_id = match.group(1).upper()
                if len(chain_id) <= 4:
                    return chain_id

        if re.match(r'^[A-Za-z0-9]{1,4}$', folder_name):
            return folder_name.upper()

        return None

    def _parse_domain_number_from_filename(self, filename):
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]

        patterns = [
            r'domain[_-]?(\d+)',
            r'd[_-]?(\d+)',
            r'model[_-]?(\d+)',
            r'pred[_-]?(\d+)',
            r'component[_-]?\d+.*[_-]d[_-]?(\d+)',
            r'^(\d+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _parse_component_filename(self, filename):
        basename = os.path.basename(filename)

        pattern1 = r'pred_([^_]+)_d_(\d+)\.(pdb|cif)'
        match1 = re.match(pattern1, basename, re.IGNORECASE)
        if match1:
            chain_id = match1.group(1).upper()
            domain_num = int(match1.group(2))
            return (chain_id, domain_num)

        pattern2 = r'component_\d+_pred_([^_]+)_\d+_d_(\d+)\.(pdb|cif)'
        match2 = re.match(pattern2, basename, re.IGNORECASE)
        if match2:
            chain_id = match2.group(1).upper()
            domain_num = int(match2.group(2))
            return (chain_id, domain_num)

        pattern3 = r'chain_([^_]+)_\d+_d_(\d+)\.(pdb|cif)'
        match3 = re.match(pattern3, basename, re.IGNORECASE)
        if match3:
            chain_id = match3.group(1).upper()
            domain_num = int(match3.group(2))
            return (chain_id, domain_num)

        return None

    def _read_structure_file(self, structure_file):
        try:
            ext = os.path.splitext(structure_file)[1].lower()
            
            if ext == '.pdb':
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                return parser.get_structure('structure', structure_file)
            elif ext == '.cif':
                from Bio.PDB import MMCIFParser
                parser = MMCIFParser(QUIET=True)
                return parser.get_structure('structure', structure_file)
            else:
                logging.warning(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            logging.warning(f"Failed to read structure ({structure_file}): {e}")
            return None

    def _get_residue_segments_from_file(self, structure_file):
        try:
            structure = self._read_structure_file(structure_file)
            
            if structure is None:
                return None
            
            residue_nums = set()
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':
                            continue
                        res_num = residue.id[1]
                        residue_nums.add(res_num)

            if not residue_nums:
                return None

            sorted_residues = sorted(residue_nums)
            segments = []

            segment_start = sorted_residues[0]
            segment_end = sorted_residues[0]

            for i in range(1, len(sorted_residues)):
                if sorted_residues[i] == segment_end + 1:
                    segment_end = sorted_residues[i]
                else:
                    segments.append((segment_start, segment_end))
                    segment_start = sorted_residues[i]
                    segment_end = sorted_residues[i]

            segments.append((segment_start, segment_end))

            return segments

        except Exception as e:
            logging.warning(f"Error reading residue segments ({structure_file}): {e}")
            return None

    def _convert_pdb_files_to_cif(self, file_list):
        if self.temp_cif_dir is None:
            self.temp_cif_dir = self.input_dir / ".temp_cif_files"
            self.temp_cif_dir.mkdir(exist_ok=True)
        
        converted_files = []
        pdb_count = 0
        
        for file_path in file_list:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdb':
                pdb_count += 1
                basename = os.path.basename(file_path)
                cif_filename = os.path.splitext(basename)[0] + '.cif'
                cif_path = self.temp_cif_dir / cif_filename
                
                if convert_pdb_to_cif(file_path, str(cif_path)):
                    converted_files.append(str(cif_path))
                else:
                    converted_files.append(file_path)
            else:
                converted_files.append(file_path)
        
        if pdb_count > 0:
            logging.info(f"  ✅ Converted {pdb_count} PDB files to CIF format")
        
        return converted_files

    def find_domain_files(self):
        logging.info(f"Searching directory for domain files: {self.input_dir}")

        domain_info_list = []
        files_to_convert = []

        subdirs = [d for d in self.input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if subdirs:
            logging.info(f"Detected {len(subdirs)} subdirectories, reading chain information...")

            for subdir in subdirs:
                chain_id = self._extract_chain_from_folder_name(subdir)

                if chain_id is None:
                    logging.warning(f"  Cannot extract chain ID from directory name: {subdir.name}; skipping")
                    continue

                logging.info(f"\n  Processing directory: {subdir.name} -> chain ID: {chain_id}")

                pdb_files = list(subdir.glob("*.pdb"))
                cif_files = list(subdir.glob("*.cif"))
                structure_files = pdb_files + cif_files
                
                if pdb_files:
                    logging.info(f"  Detected {len(pdb_files)} PDB files, converting to CIF...")
                    files_to_convert.extend([str(f) for f in pdb_files])

                if pdb_files:
                    converted = self._convert_pdb_files_to_cif([str(f) for f in pdb_files])
                    structure_files = [Path(f) for f in converted] + cif_files

                for structure_file in structure_files:
                    domain_num = self._parse_domain_number_from_filename(structure_file.name)

                    if domain_num is None:
                        logging.warning(f"    Cannot extract domain number from filename: {structure_file.name}; trying the default number")
                        domain_num = len([d for d in domain_info_list if d['chain_id'] == chain_id]) + 1

                    segments = self._get_residue_segments_from_file(str(structure_file))

                    if segments:
                        min_res = segments[0][0]
                        max_res = segments[-1][1]

                        domain_info_list.append({
                            'structure_file': str(structure_file),
                            'chain_id': chain_id,
                            'domain_num': domain_num,
                            'min_residue': min_res,
                            'max_residue': max_res,
                            'segments': segments
                        })

                        segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in segments])
                        file_format = os.path.splitext(structure_file)[1].upper()
                        logging.info(f"    Found: {structure_file.name} ({file_format}) -> domain{domain_num}, residues: {segments_str}")

        else:
            logging.info('No subdirectories found; searching the root directory...')
            pdb_files = list(self.input_dir.glob("*.pdb"))
            cif_files = list(self.input_dir.glob("*.cif"))
            
            if pdb_files:
                logging.info(f"Detected {len(pdb_files)} PDB files, converting to CIF...")
                converted = self._convert_pdb_files_to_cif([str(f) for f in pdb_files])
                structure_files = [Path(f) for f in converted] + cif_files
            else:
                structure_files = cif_files

            for structure_file in structure_files:
                result = self._parse_component_filename(structure_file.name)

                if result:
                    chain_id, domain_num = result
                    segments = self._get_residue_segments_from_file(str(structure_file))

                    if segments:
                        min_res = segments[0][0]
                        max_res = segments[-1][1]

                        domain_info_list.append({
                            'structure_file': str(structure_file),
                            'chain_id': chain_id,
                            'domain_num': domain_num,
                            'min_residue': min_res,
                            'max_residue': max_res,
                            'segments': segments
                        })

                        segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in segments])
                        file_format = os.path.splitext(structure_file)[1].upper()
                        logging.info(f"  Found: {structure_file.name} ({file_format}) -> chain{chain_id}, domain{domain_num}, residues: {segments_str}")

        if not domain_info_list:
            logging.warning('No valid domain files found')
            return []

        logging.info(f"\nTotal found: {len(domain_info_list)} domain files (CIF format)")
        self.domain_files = domain_info_list
        return domain_info_list

    def _group_domains_by_chain(self):
        logging.info('\nGrouping domains by chain...')

        chain_groups = defaultdict(list)

        for domain_info in self.domain_files:
            chain_id = domain_info['chain_id']
            chain_groups[chain_id].append(domain_info)

        chain_segments = {}

        for chain_id in chain_groups:
            all_segments = []

            for domain_info in chain_groups[chain_id]:
                segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in domain_info['segments']])
                logging.info(
                    f"  Chain {chain_id}, domain {domain_info['domain_num']}: residue segments [{segments_str}] ({os.path.basename(domain_info['structure_file'])})")

                for seg_start, seg_end in domain_info['segments']:
                    all_segments.append({
                        'start': seg_start,
                        'end': seg_end,
                        'domain_info': domain_info,
                        'segment': (seg_start, seg_end)
                    })

            all_segments.sort(key=lambda x: x['start'])
            chain_segments[chain_id] = all_segments

            logging.info(f"\n  Chain {chain_id} segments reordered by residue number:")
            for seg in all_segments:
                domain_num = seg['domain_info']['domain_num']
                logging.info(f"    Residue {seg['start']}-{seg['end']} (from domain {domain_num})")

        return chain_segments

    def _extract_atoms_from_file(self, structure_file, residue_range=None):
        atom_info_list = []

        try:
            structure = self._read_structure_file(structure_file)
            
            if structure is None:
                return atom_info_list
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':
                            continue
                        
                        res_num = residue.id[1]
                        insertion_code = residue.id[2] if residue.id[2] != ' ' else ''
                        
                        if residue_range is not None:
                            start, end = residue_range
                            if not (start <= res_num <= end):
                                continue
                        
                        residue_name = residue.resname
                        
                        for atom in residue:
                            atom_name = atom.name
                            alt_loc = atom.altloc if hasattr(atom, 'altloc') else ''
                            coord = atom.coord
                            occupancy = atom.occupancy if hasattr(atom, 'occupancy') else 1.0
                            temp_factor = atom.bfactor if hasattr(atom, 'bfactor') else 20.0
                            element = atom.element if hasattr(atom, 'element') else ''
                            
                            atom_info_list.append({
                                'atom_name': atom_name,
                                'alt_loc': alt_loc,
                                'residue_name': residue_name,
                                'res_num': res_num,
                                'insertion_code': insertion_code,
                                'x': coord[0],
                                'y': coord[1],
                                'z': coord[2],
                                'occupancy': occupancy,
                                'temp_factor': temp_factor,
                                'element': element,
                                'charge': ''
                            })
            
        except Exception as e:
            logging.error(f"Error reading structure file {structure_file}: {e}")

        return atom_info_list

    def assemble_complex(self, output_path=None):
        if not self.domain_files:
            logging.error('No domain files found; cannot assemble')
            return None

        if output_path is None:
            output_path = self.input_dir / self.output_filename
        else:
            output_path = Path(output_path)

        chain_segments = self._group_domains_by_chain()

        if not chain_segments:
            logging.error('Cannot group domains by chain')
            return None

        logging.info(f"\nStarting complex assembly...")
        logging.info(f"Total chains: {len(chain_segments)}")

        try:
            from Bio.PDB import Structure, Model, Chain, Residue, Atom
            from Bio.PDB import MMCIFIO
            
            new_structure = Structure.Structure('assembled_complex')
            new_model = Model.Model(0)
            new_structure.add(new_model)
            
            global_atom_counter = 1
            
            for chain_id in sorted(chain_segments.keys()):
                segments = chain_segments[chain_id]
                
                logging.info(f"\nProcessing chain {chain_id}: {len(segments)} segments")
                
                new_chain = Chain.Chain(chain_id)
                new_model.add(new_chain)
                
                for seg_info in segments:
                    structure_file = seg_info['domain_info']['structure_file']
                    domain_num = seg_info['domain_info']['domain_num']
                    seg_start = seg_info['start']
                    seg_end = seg_info['end']
                    
                    logging.info(
                        f"  Adding segment: residues {seg_start}-{seg_end} (from domain {domain_num}, {os.path.basename(structure_file)})")
                    
                    atom_info_list = self._extract_atoms_from_file(structure_file, (seg_start, seg_end))
                    
                    residues_dict = defaultdict(list)
                    for atom_info in atom_info_list:
                        res_key = (atom_info['res_num'], atom_info['insertion_code'])
                        residues_dict[res_key].append(atom_info)
                    
                    for res_key in sorted(residues_dict.keys()):
                        res_num, insertion_code = res_key
                        atoms = residues_dict[res_key]
                        
                        if not atoms:
                            continue
                        
                        residue_name = atoms[0]['residue_name']
                        
                        residue_id = (' ', res_num, insertion_code if insertion_code else ' ')
                        new_residue = Residue.Residue(residue_id, residue_name, '')
                        
                        for atom_info in atoms:
                            atom = Atom.Atom(
                                name=atom_info['atom_name'],
                                coord=[atom_info['x'], atom_info['y'], atom_info['z']],
                                bfactor=atom_info['temp_factor'],
                                occupancy=atom_info['occupancy'],
                                altloc=atom_info['alt_loc'],
                                fullname=f" {atom_info['atom_name']:<3s}",
                                serial_number=global_atom_counter,
                                element=atom_info['element']
                            )
                            new_residue.add(atom)
                            global_atom_counter += 1
                        
                        new_chain.add(new_residue)
            
            io = MMCIFIO()
            io.set_structure(new_structure)
            io.save(str(output_path))
            
            logging.info(f"\nComplex assembly completed!")
            logging.info(f"Output file: {output_path} (CIF format)")
            logging.info(f"Total atoms: {(global_atom_counter - 1)}")
            logging.info(f"Total chains: {len(chain_segments)}")

            return str(output_path)

        except Exception as e:
            logging.error(f"Complex assembly error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        logging.info("=" * 60)
        logging.info('Domain complex assembly tool v3.0 (CIF support)')
        logging.info("=" * 60)

        domain_files = self.find_domain_files()

        if not domain_files:
            logging.error('No valid domain files found')
            return None

        output_path = self.assemble_complex()

        logging.info("=" * 60)
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Domain complex assembly tool v3.0 with automatic PDB/CIF conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nSupported directory layouts:\n\n1. Chain subdirectories (recommended):\n   input_dir/\n   ├── chain_A/          # also chainA, A, protein_A\n   │   ├── domain_1.pdb  # converted to CIF\n   │   ├── domain_2.cif  # used directly\n   │   └── domain_3.pdb  # converted to CIF\n   ├── chain_B/\n   │   ├── domain_1.cif\n   │   └── domain_2.cif\n   └── ...\n\n2. Files in one directory:\n   input_dir/\n   ├── component_1_pred_A_1_d_1.pdb\n   ├── component_2_pred_A_2_d_2.cif\n   └── component_3_pred_B_1_d_1.cif\n\nFormats: .pdb (converted to CIF), .cif (used directly).\nSupported names:\n  domain_1.pdb/cif, domain_2.pdb/cif\n  domain1.pdb/cif, domain2.pdb/cif\n  d1.pdb/cif, d2.pdb/cif\n  1.pdb/cif, 2.pdb/cif\n  model_1.pdb/cif, pred_1.pdb/cif\n  component_N_pred_X_M_d_D.pdb/cif\n  pred_X_d_Y.pdb/cif\nChain directory names: chain_A, chainA, Chain_A, protein_A, domain_A, A, B, C.\nOutput: .cif; PDB inputs are converted automatically.\n\nExamples:\n  python zuzhuang_refine.py /path/to/chains/\n  python zuzhuang_refine.py /path/to/chains/ -o my_complex.cif\n  python zuzhuang_refine.py /path/to/pdb/files/\n        '
    )

    parser.add_argument(
        "input_dir",
        help='Directory containing domain files (PDB and CIF)'
    )
    parser.add_argument(
        "-o", "--output",
        help='Output filename (default: assembled_complex.cif)',
        default="assembled_complex.cif"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: input directory does not exist - {args.input_dir}")
        return 1

    if not os.path.isdir(args.input_dir):
        print(f"Error: input path is not a directory - {args.input_dir}")
        return 1

    assembler = DomainComplexAssembler(
        input_dir=args.input_dir,
        output_filename=args.output
    )

    result = assembler.run()

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())