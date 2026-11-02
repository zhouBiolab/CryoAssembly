#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'Detect overlap between input structures.'

import os
import sys
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print('Warning: PyTorch is unavailable; using NumPy')


def extract_residue_locations(pdb_file):
    residue_locations = []
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # CA for protein, P for DNA/RNA
                if line.startswith("ATOM") and (line[12:16].strip() == "CA" or line[12:16].strip() == "P"):
                    residue_locations.append(
                        np.array(
                            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        )
                    )
    except Exception as e:
        print(f"Warning: reading file {pdb_file} failed: {e}")
        return np.array([])
    
    return np.array(residue_locations)


def calculate_overlap_ratio_torch(pdb_file1, pdb_file2, clash_distance=3.0):
    locations1 = extract_residue_locations(pdb_file1)
    locations2 = extract_residue_locations(pdb_file2)
    
    if len(locations1) == 0 or len(locations2) == 0:
        return 0.0
    
    locations1_torch = torch.from_numpy(locations1).float().cuda()
    locations2_torch = torch.from_numpy(locations2).float().cuda()
    
    distance_array = torch.cdist(locations1_torch, locations2_torch, p=2)
    
    min_distances = torch.amin(distance_array, dim=1)
    
    ratio_close = (min_distances < clash_distance).sum().item() / len(locations1)
    
    return ratio_close


def calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance=3.0):
    locations1 = extract_residue_locations(pdb_file1)
    locations2 = extract_residue_locations(pdb_file2)
    
    if len(locations1) == 0 or len(locations2) == 0:
        return 0.0
    
    diff = locations1[:, np.newaxis, :] - locations2[np.newaxis, :, :]
    distance_array = np.sqrt(np.sum(diff ** 2, axis=2))
    
    min_distances = np.min(distance_array, axis=1)
    
    ratio_close = np.sum(min_distances < clash_distance) / len(locations1)
    
    return ratio_close


def calculate_overlap_ratio(pdb_file1, pdb_file2, clash_distance=3.0):
    if TORCH_AVAILABLE:
        try:
            return calculate_overlap_ratio_torch(pdb_file1, pdb_file2, clash_distance)
        except Exception as e:
            print(f"Warning: PyTorch computation failed, falling back to NumPy: {e}")
            return calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance)
    else:
        return calculate_overlap_ratio_numpy(pdb_file1, pdb_file2, clash_distance)


def scan_directory(root_dir, clash_distance=3.0, ratio_threshold=0.05):
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: directory '{root_dir}' does not exist")
        return
    
    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory")
        return
    
    pdb_files = list(root_path.rglob("*.pdb"))
    
    if not pdb_files:
        print(f"In directory '{root_dir}' contains no PDB files")
        return
    
    if len(pdb_files) < 2:
        print(f"Found only {len(pdb_files)} PDB files; pairwise comparison requires at least 2")
        return
    
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Clash distance threshold: {clash_distance} Å")
    print(f"Overlap threshold: {(ratio_threshold * 100)}%")
    print(f"Computation backend: {('PyTorch (GPU)' if TORCH_AVAILABLE else 'NumPy (CPU)')}")
    print("=" * 80)
    
    overlap_pairs = []
    total_comparisons = len(pdb_files) * (len(pdb_files) - 1) // 2
    current_comparison = 0
    
    for i in range(len(pdb_files)):
        for j in range(i + 1, len(pdb_files)):
            current_comparison += 1
            
            if current_comparison % 10 == 0 or current_comparison == total_comparisons:
                print(f"Progress: {current_comparison}/{total_comparisons} ({((current_comparison / total_comparisons) * 100):.1f}%)")
            
            file1 = pdb_files[i]
            file2 = pdb_files[j]
            
            try:
                ratio = calculate_overlap_ratio(file1, file2, clash_distance)
                
                if ratio >= ratio_threshold:
                    overlap_pairs.append((file1, file2, ratio))
                    
            except Exception as e:
                print(f"Warning: comparing {file1.name} and {file2.name} failed: {e}")
    
    print("\n" + "=" * 80)
    print(f"Scan completed!")
    print(f"Total compared: {total_comparisons} file pairs")
    print(f"Found {len(overlap_pairs)} file pairs with substantial overlap\n")
    
    if overlap_pairs:
        overlap_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print('Overlapping file-pair details:')
        print("-" * 80)
        for file1, file2, ratio in overlap_pairs:
            print(f"\nFile 1: {file1}")
            print(f"File 2: {file2}")
            print(f"Overlap fraction: {(ratio * 100):.2f}%")
    else:
        print('No file pair exceeds the overlap threshold')


def main():
    if len(sys.argv) < 2:
        print('PDB overlap detection tool')
        print('\nUsage:')
        print('  python -m protassem.core.check_clash <directory> [clash_distance] [overlap_threshold]')
        print('\nParameters:')
        print('  directory: Root directory containing PDB files (required)')
        print('  clash_distance: Distance threshold in angstroms (default: 3.0)')
        print('  overlap_threshold: Fraction defining substantial overlap (default: 0.05, 5%)')
        print('\nExample:')
        print("  python -m protassem.core.check_clash ./pdb_files")
        print("  python -m protassem.core.check_clash ./pdb_files 3.0")
        print("  python -m protassem.core.check_clash ./pdb_files 3.0 0.1")
        print('\nDescription:')
        print('  - Overlap is the fraction of file-1 residues within the clash distance of file-2 residues')
        print('  - Compare CA atoms (proteins) or P atoms (DNA/RNA)')
        print('  - Use GPU acceleration automatically when PyTorch is installed')
        sys.exit(1)
    
    directory = sys.argv[1]
    clash_distance = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    ratio_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    
    scan_directory(directory, clash_distance, ratio_threshold)


if __name__ == "__main__":
    main()