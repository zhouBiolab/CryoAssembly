#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
import subprocess
import shutil
from Bio.PDB import PDBParser, MMCIFParser
from numba.typed import List
import tempfile
import glob
import re

# Add project root for protassem imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from protassem.core.constants import VDW_RADII
from protassem.core.numba_kernels import add_sphere_mask as _njit_add_sphere_mask
from protassem.core.io import read_structure
from protassem.core import points_txt

# VDW_RADII imported from protassem.core.constants
# _njit_add_sphere_mask imported from protassem.core.numba_kernels


def load_sample_points_with_info(file_path):
    cloud = points_txt.read_point_cloud(file_path)
    with open(file_path, encoding="utf-8") as handle:
        original_lines = handle.readlines()
    line_mapping = points_txt.line_mapping(cloud)
    header_info = {"sample": cloud.sample,
                   "origin": [float(v) for v in cloud.origin],
                   "line1": cloud.header_lines[1],
                   "line2": cloud.header_lines[2],
                   "origin_line": cloud.header_lines[3],
                   "line4": cloud.header_lines[4]}
    return (cloud.points, cloud.vectors, cloud.densities,
            original_lines, line_mapping, header_info)


def save_domain_txt(original_lines, kept_indices, line_mapping, header_info, output_path):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for line in original_lines[:points_txt.HEADER_LINES]:
            handle.write(line)
        for point_idx in kept_indices:
            coord_line_idx, vector_line_idx = line_mapping[point_idx]
            handle.write(original_lines[coord_line_idx])
            handle.write(original_lines[vector_line_idx])
    print('Saved domain TXT file: %s' % output_path)
    print('Retained %d points' % len(kept_indices))


def copy_pdb_with_new_name(source_pdb, target_pdb):
    shutil.copy2(source_pdb, target_pdb)
    print('Copied PDB file: %s -> %s' % (source_pdb, target_pdb))


def get_atom_list(pdb_file, backbone_only=False):
    """Wrapper around core.io.read_structure for backward compat."""
    return read_structure(pdb_file, backbone_only=backbone_only)



def calculate_unified_grid_parameters(sample_points, complete_pdb_atoms, resolution=6.0, voxel_size=None):
    import math
    
    print('Computing shared grid parameters...')
    
    if voxel_size is None:
        grid_spacing = resolution / 3.0
        voxel_size_array = np.array([grid_spacing, grid_spacing, grid_spacing])
    else:
        voxel_size_array = np.array([voxel_size, voxel_size, voxel_size])
    
    all_coords = np.vstack([sample_points, complete_pdb_atoms])
    max_x, max_y, max_z = all_coords.max(axis=0)
    min_x, min_y, min_z = all_coords.min(axis=0)
    
    pad = 3.0 * resolution
    
    x_size = math.ceil((max_x - min_x + 2 * pad) / voxel_size_array[0])
    y_size = math.ceil((max_y - min_y + 2 * pad) / voxel_size_array[1])
    z_size = math.ceil((max_z - min_z + 2 * pad) / voxel_size_array[2])
    
    x_origin = min_x - pad
    y_origin = min_y - pad
    z_origin = min_z - pad
    origin = np.array([x_origin, y_origin, z_origin])
    
    box_size = (z_size, y_size, x_size)
    
    print(f"Shared grid parameters:")
    print(f"  Origin: {origin}")
    print(f"  Box dimensions: {box_size}")
    print(f"  Voxel size: {voxel_size_array}")
    print(f"  Total voxels: {np.prod(box_size):,}")
    
    return origin, box_size, voxel_size_array

def make_atomic_mask_optimized(origin, voxel_size, box_size, atom_list, atom_type_list, solvent_radius=1.1):
    mask = np.zeros(box_size, dtype=np.bool_)
    nz, ny, nx = mask.shape
    
    for coord, atom_type in zip(atom_list, atom_type_list):
        r_vdw = VDW_RADII.get(atom_type, 1.70)
        radius = r_vdw + solvent_radius
        radius_sq = radius * radius
        
        i_center = (coord[0] - origin[0]) / voxel_size[0]
        j_center = (coord[1] - origin[1]) / voxel_size[1]
        k_center = (coord[2] - origin[2]) / voxel_size[2]
        
        radius_voxels = np.array([
            radius / voxel_size[0],
            radius / voxel_size[1],
            radius / voxel_size[2]
        ])
        
        i0 = max(0, int(np.floor(i_center - radius_voxels[0])))
        i1 = min(nx, int(np.ceil(i_center + radius_voxels[0])) + 1)
        j0 = max(0, int(np.floor(j_center - radius_voxels[1])))
        j1 = min(ny, int(np.ceil(j_center + radius_voxels[1])) + 1)
        k0 = max(0, int(np.floor(k_center - radius_voxels[2])))
        k1 = min(nz, int(np.ceil(k_center + radius_voxels[2])) + 1)
        
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        
        _njit_add_sphere_mask(
            mask, i_center, j_center, k_center,
            voxel_size[0], voxel_size[1], voxel_size[2],
            radius_sq, i0, i1, j0, j1, k0, k1
        )
    
    return mask

def filter_points_by_mask_keep_inside(sample_points, mask, origin, voxel_size):
    nz, ny, nx = mask.shape
    kept_indices = []
    removed_indices = []
    
    for i, point in enumerate(sample_points):
        voxel_coords = (point - origin) / voxel_size
        vi = int(np.round(voxel_coords[0]))
        vj = int(np.round(voxel_coords[1]))
        vk = int(np.round(voxel_coords[2]))
        
        if 0 <= vi < nx and 0 <= vj < ny and 0 <= vk < nz:
            if mask[vk, vj, vi]:
                kept_indices.append(i)
            else:
                removed_indices.append(i)
        else:
            removed_indices.append(i)
    
    return kept_indices, removed_indices

def extract_domain_number(filename):
    match = re.search(r'_d_(\d+)\.pdb$', filename)
    if match:
        return int(match.group(1))
    else:
        return 0

def call_domain_parser(pdb_file, domain_parser_path):
    print(f"Partitioning PDB with DomainParser: {pdb_file}")
    
    if not os.path.exists(domain_parser_path):
        raise FileNotFoundError(f"DomainParser not found: {domain_parser_path}")
    
    current_dir = os.getcwd()
    
    try:
        cmd = [sys.executable, domain_parser_path, pdb_file]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode != 0:
            print(f"DomainParser failed: {result.stderr}")
            return {}, ""
        
        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        domain_files = glob.glob(f"{base_name}*_d_*.pdb")
        
        domain_dict = {}
        for domain_file in domain_files:
            domain_num = extract_domain_number(domain_file)
            if domain_num > 0:
                domain_dict[domain_num] = domain_file
        
        print(f"Found {len(domain_dict)} domain files: {list(domain_dict.values())}")
        
        return domain_dict, result.stdout.strip()
        
    except Exception as e:
        print(f"DomainParser error: {e}")
        return {}, ""

def handle_single_domain_case(input_txt, input_pdb, points, original_lines, line_mapping, header_info):
    print('No multiple-domain files found; the PDB may contain one domain')
    print('Generating single-domain TXT and PDB files...')
    
    base_name_txt = os.path.splitext(input_txt)[0]
    base_name_pdb = os.path.splitext(input_pdb)[0]
    
    output_txt = f"{base_name_txt}_domain1.txt"
    output_pdb = f"{base_name_pdb}_d_1.pdb"
    
    shutil.copy2(input_txt, output_txt)
    print(f"Copied original TXT file to: {output_txt}")
    print(f"Retained all {len(points)} points")
    
    copy_pdb_with_new_name(input_pdb, output_pdb)
    
    return [{
        'domain_num': 1,
        'domain_file': output_pdb,
        'txt_file': output_txt,
        'total_points': len(points),
        'kept_points': len(points),
        'removed_points': 0,
        'retention_rate': 100.0
    }]

def split_txt_by_domains_unified_grid(input_txt, input_pdb, domain_parser_path, 
                                     resolution=6.0, voxel_size=None, solvent_radius=1.1, backbone_only=False):
    print("=" * 80)
    print('Domain TXT splitter with shared coordinates, paired single-domain outputs and consistent names')
    print(f"Input TXT file: {input_txt}")
    print(f"Input PDB file: {input_pdb}")
    print(f"DomainParser path: {domain_parser_path}")
    print(f"Resolution: {resolution} Å")
    if voxel_size:
        print(f"Voxel size: {voxel_size} Å")
    else:
        print(f"Voxel size: {(resolution / 3.0):.2f} Å (resolution/3)")
    print(f"Solvent radius: {solvent_radius} Å")
    print(f"Backbone atoms only: {('Yes' if backbone_only else 'No')}")
    print("=" * 80)
    
    if not os.path.exists(input_txt):
        raise FileNotFoundError('Input TXT file not found: %s' % input_txt)
        
    if not os.path.exists(input_pdb):
        raise FileNotFoundError('Input PDB file not found: %s' % input_pdb)
    
    print('\n1. Reading the TXT file...')
    points, vectors, densities, original_lines, line_mapping, header_info = load_sample_points_with_info(input_txt)
    print(f"Read {len(points)} sampled points")
    
    print('\n2. Partitioning the PDB file with DomainParser...')
    domain_dict, domain_info = call_domain_parser(input_pdb, domain_parser_path)
    
    if not domain_dict:
        results = handle_single_domain_case(input_txt, input_pdb, points, original_lines, line_mapping, header_info)
        
        print("\n" + "=" * 80)
        print('Single-domain result summary:')
        result = results[0]
        print(f"Domain {result['domain_num']}:")
        print(f"  PDB file: {result['domain_file']}")
        print(f"  TXT file: {result['txt_file']}")
        print(f"  Retained points: {result['kept_points']}/{result['total_points']} ({result['retention_rate']:.1f}%)")
        print(f"\nPaired PDB and TXT files generated")
        print("=" * 80)
        print('Single-domain processing completed!')
        return
    
    print('\n3. Reading the complete PDB and computing shared grid parameters...')
    complete_atoms, complete_atom_types = get_atom_list(input_pdb, backbone_only=backbone_only)
    print(f"Complete PDB contains {len(complete_atoms)} atoms ({('Backbone' if backbone_only else 'All')})")
    
    unified_origin, unified_box_size, unified_voxel_size = calculate_unified_grid_parameters(
        points, complete_atoms, resolution=resolution, voxel_size=voxel_size
    )
    
    print('\n4. Generating domain masks and filtering the TXT file...')
    print('   ※ All domains share grid parameters and one coordinate frame')
    
    base_name_txt = os.path.splitext(input_txt)[0]
    base_name_pdb = os.path.splitext(input_pdb)[0]
    results = []
    
    for domain_num in sorted(domain_dict.keys()):
        domain_file = domain_dict[domain_num]
        print(f"\nProcessing domain {domain_num}: {domain_file}")
        
        try:
            print(f"  Reading domain PDB files...")
            domain_atoms, domain_atom_types = get_atom_list(domain_file, backbone_only=backbone_only)
            print(f"  Domain contains {len(domain_atoms)} atoms ({('Backbone' if backbone_only else 'All')})")
            
            print(f"  Generating masks with shared grid parameters...")
            print(f"    Grid origin: {unified_origin}")
            print(f"    Grid dimensions: {unified_box_size}")
            print(f"    Voxel size: {unified_voxel_size}")
            
            mask = make_atomic_mask_optimized(
                unified_origin, unified_voxel_size, unified_box_size, 
                domain_atoms, domain_atom_types, solvent_radius=solvent_radius
            )
            print(f"  Total mask voxels: {mask.size:,}")
            print(f"  Voxels inside mask: {np.sum(mask):,}")
            print(f"  Mask coverage: {((100.0 * np.sum(mask)) / mask.size):.2f}%")
            
            print(f"  Filtering sampled points by mask...")
            kept_indices, removed_indices = filter_points_by_mask_keep_inside(
                points, mask, unified_origin, unified_voxel_size
            )
            
            domain_txt = f"{base_name_txt}_domain{domain_num}.txt"
            save_domain_txt(original_lines, kept_indices, line_mapping, header_info, domain_txt)
            
            target_domain_pdb = f"{base_name_pdb}_d_{domain_num}.pdb"
            if os.path.exists(domain_file) and domain_file != target_domain_pdb:
                shutil.move(domain_file, target_domain_pdb)
                print(f"Renaming domain PDB file: {domain_file} -> {target_domain_pdb}")
            elif not os.path.exists(target_domain_pdb):
                print(f"Warning: domain PDB file not found: {domain_file}")
                target_domain_pdb = domain_file
            
            results.append({
                'domain_num': domain_num,
                'domain_file': target_domain_pdb,
                'txt_file': domain_txt,
                'total_points': len(points),
                'kept_points': len(kept_indices),
                'removed_points': len(removed_indices),
                'retention_rate': 100.0 * len(kept_indices) / len(points)
            })
            
        except Exception as e:
            print(f"Processing domain {domain_file} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print('Partitioning summary:')
    print(f"Shared grid parameters:")
    print(f"  Grid origin: {unified_origin}")
    print(f"  Grid dimensions: {unified_box_size}")
    print(f"  Voxel size: {unified_voxel_size}")
    print(f"  Total voxels: {np.prod(unified_box_size):,}")
    print()
    
    for result in results:
        print(f"Domain {result['domain_num']}:")
        print(f"  PDB file: {result['domain_file']}")
        print(f"  TXT file: {result['txt_file']}")
        print(f"  Retained points: {result['kept_points']}/{result['total_points']} ({result['retention_rate']:.1f}%)")
    
    print(f"\nTotal generated: {len(results)} domain TXT and PDB files")
    print("=" * 80)
    print('Partitioning completed!')
    print('\nMain improvements:')
    print('✓ All domains share grid parameters and one coordinate frame')
    print('✓ Reuse grid parameters to avoid redundant computation')
    print('✓ Grid bounds come from the complete PDB for consistent domain boundaries')
    print('✓ Each domain TXT contains only points inside its domain mask')
    print('✓ Single-domain inputs also produce a paired PDB file')
    print('✓ PDB filenames use _d_number (e.g. protein_d_1.pdb)')

def main():
    parser = argparse.ArgumentParser(
        description='Domain TXT splitter with shared coordinates, paired single-domain outputs and consistent names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n  python optimized_domain_splitter.py input.txt protein.pdb\n  python optimized_domain_splitter.py input.txt protein.pdb --domain_parser /path/to/DomainParser.py\n  python optimized_domain_splitter.py input.txt protein.pdb --resolution 4.0 --voxel_size 1.5 --backbone_only\n\nFeatures:\n  - Shared origin, box dimensions and voxel size for all domains.\n  - Grid parameters are computed once from the complete PDB.\n  - Each domain mask uses its own atoms on the shared grid.\n  - Single-domain inputs produce paired TXT/PDB files named with _d_1.\n  - PDB names use the DomainParser _d_number convention.\n\nWorkflow:\n  1. Compute grid parameters from all atoms and sampled points.\n  2. Partition the PDB with DomainParser.\n  3. For one domain, copy TXT/PDB and rename them with _d_1.\n  4. For multiple domains, generate atom masks on the shared grid.\n  5. Filter sampled points by each mask and name PDB files with _d_number.\n\nOutputs:\n  - Domain PDB files: protein_d_1.pdb, protein_d_2.pdb, ...\n  - Domain TXT files: input_domain1.txt, input_domain2.txt, ...\n  - All files use one coordinate frame; every TXT has a paired PDB.\n        '
    )
    parser.add_argument("input_txt", help='Input TXT path')
    parser.add_argument("input_pdb", help='Input PDB path')
    parser.add_argument("--domain_parser", 
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "DomainParser.py"),
                       help='Path to DomainParser.py')
    parser.add_argument("--resolution", type=float, default=6.0, help='Resolution for grid computation (default: 6.0 Å)')
    parser.add_argument("--voxel_size", type=float, default=None, help='Voxel size (default: resolution/3)')
    parser.add_argument("--solvent_radius", type=float, default=1.1, help='Solvent radius (default: 1.1 Å)')
    parser.add_argument("--backbone_only", action="store_true", help='Generate masks using backbone atoms only')
    
    args = parser.parse_args()
    
    split_txt_by_domains_unified_grid(
        input_txt=args.input_txt,
        input_pdb=args.input_pdb,
        domain_parser_path=args.domain_parser,
        resolution=args.resolution,
        voxel_size=args.voxel_size,
        solvent_radius=args.solvent_radius,
        backbone_only=args.backbone_only
    )

if __name__ == "__main__":
    main()