import os
import sys
import shutil
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

from protassem.assembly.refine.refine_energy import (
    calculate_chain_connection_energy,
    calculate_chain_connection_score,
    _calculate_tm_score
)
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence


@dataclass
class DomainInfo:
    chain_name: str
    domain_file: str
    domain_index: int

    def __hash__(self):
        return hash((self.chain_name, self.domain_file, self.domain_index))

    def __eq__(self, other):
        return (self.chain_name == other.chain_name and
                self.domain_file == other.domain_file and
                self.domain_index == other.domain_index)


@dataclass
class AlignedDomain:
    original_domain: DomainInfo
    target_domain: DomainInfo
    aligned_file: str
    is_original: bool
    display_name: str
    variant_index: int


@dataclass
class HomologousGroup:
    group_id: int
    domains: List[DomainInfo]
    tm_scores: Dict[Tuple[str, str], float]


def convert_pdb_to_cif(pdb_file: str, cif_file: str):
    try:
        from Bio.PDB import PDBParser, MMCIFIO
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_file)
        
        return True
    except Exception as e:
        print(f"    ⚠️  PDB-to-CIF conversion failed: {e}")
        return False


def calculate_config_energy_and_score(args):
    (config_idx, chain_config, temp_chain_dir, ideal_distance,
     tight_tolerance, loose_min, loose_max) = args

    config_dir = os.path.join(temp_chain_dir, f"config_{config_idx}")
    os.makedirs(config_dir, exist_ok=True)

    config_files = []
    config_files_info = []

    for pos_idx, aligned_domain in enumerate(chain_config):
        src_file = aligned_domain.aligned_file
        dst_filename = f"pos_{pos_idx}_{aligned_domain.display_name}.cif"
        dst_file = os.path.join(config_dir, dst_filename)
        shutil.copy2(src_file, dst_file)
        config_files.append(dst_file)

        file_info = {
            'position': pos_idx,
            'display_name': aligned_domain.display_name,
            'variant_index': aligned_domain.variant_index,
            'is_original': aligned_domain.is_original,
            'file': dst_filename
        }
        if not aligned_domain.is_original:
            file_info['source_chain'] = aligned_domain.original_domain.chain_name
            file_info['source_file'] = os.path.basename(aligned_domain.original_domain.domain_file)

        config_files_info.append(file_info)

    energy = calculate_chain_connection_energy(config_files, ideal_distance)

    score, score_details = calculate_chain_connection_score(
        config_files,
        ideal_distance,
        tight_tolerance,
        loose_min,
        loose_max
    )

    return (config_idx, energy, score, score_details, chain_config, config_files_info)


class ChainEnumerator:

    def __init__(self, root_dir: str, usalign_path: str = "USalign",
                 tm_threshold: float = 0.5, ideal_distance: float = 3.8,
                 tight_tolerance: float = 1.5, loose_min: float = 5.3,
                 loose_max: float = 20.0, n_processes: int = None):
        self.root_dir = root_dir
        self.usalign_path = usalign_path
        self.tm_threshold = tm_threshold
        self.ideal_distance = ideal_distance
        self.tight_tolerance = tight_tolerance
        self.loose_min = loose_min
        self.loose_max = loose_max

        if n_processes is None:
            self.n_processes = min(20, cpu_count())
        else:
            self.n_processes = min(n_processes, cpu_count())

        print(f"🚀 Using {self.n_processes} processes for parallel computation\n")

        self.chains = {}
        self.domain_infos = {}
        self.homologous_groups = []
        self.aligned_domains = defaultdict(list)

        self.fully_homologous_chain_groups = []

        self.work_dir = os.path.join(root_dir, "optimization_workspace")
        self.backup_dir = os.path.join(self.work_dir, "original_domains")
        self.aligned_dir = os.path.join(self.work_dir, "aligned_domains")

        self._initialize_workspace()

    def _initialize_workspace(self):
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.aligned_dir, exist_ok=True)
        print(f"✅ Workspace created at: {self.work_dir}\n")

    def step1_load_chains_and_backup(self):
        print("=" * 80)
        print('Step 1: load chains and back up original files (CIF supported)')
        print("=" * 80)

        for chain_name in sorted(os.listdir(self.root_dir)):
            chain_path = os.path.join(self.root_dir, chain_name)
            if not os.path.isdir(chain_path) or chain_name == "optimization_workspace":
                continue

            domain_files = []
            files_to_rename = []
            files_to_convert = []

            for file in sorted(os.listdir(chain_path)):
                if file.endswith('.cif'):
                    if '_d_' in file:
                        domain_files.append(os.path.join(chain_path, file))
                    elif file.startswith('domain_'):
                        files_to_rename.append(file)
                elif file.endswith('.pdb'):
                    if '_d_' in file:
                        files_to_convert.append((file, 'original_format'))
                    elif file.startswith('domain_'):
                        files_to_convert.append((file, 'new_format'))

            if files_to_convert:
                print(f"\n  Chain {chain_name}: PDB files detected, converting to CIF...")
                
                for pdb_file, format_type in files_to_convert:
                    pdb_path = os.path.join(chain_path, pdb_file)
                    
                    if format_type == 'original_format':
                        cif_filename = pdb_file.replace('.pdb', '.cif')
                    else:
                        cif_filename = pdb_file.replace('.pdb', '.cif')
                    
                    cif_path = os.path.join(chain_path, cif_filename)
                    
                    if convert_pdb_to_cif(pdb_path, cif_path):
                        print(f"    ✅ {pdb_file} -> {cif_filename}")
                        
                        if format_type == 'original_format':
                            domain_files.append(cif_path)
                        else:
                            files_to_rename.append(cif_filename)
                    else:
                        print(f"    ❌ Conversion failed: {pdb_file}")

            if files_to_rename:
                print(f"\n  Chain {chain_name}: domain_*.cif naming detected, converting...")

                def extract_number(filename):
                    try:
                        # domain_1.cif -> 1
                        num_str = filename[7:-4]
                        return int(num_str)
                    except ValueError:
                        return 999999

                files_to_rename.sort(key=extract_number)

                for idx, old_filename in enumerate(files_to_rename):
                    old_path = os.path.join(chain_path, old_filename)

                    try:
                        domain_num = extract_number(old_filename)
                        new_filename = f"pred_{chain_name}_d_{domain_num}.cif"
                    except:
                        new_filename = f"pred_{chain_name}_d_{idx}.cif"

                    new_path = os.path.join(chain_path, new_filename)

                    os.rename(old_path, new_path)
                    domain_files.append(new_path)

                    print(f"    ✅ {old_filename} -> {new_filename}")

            if not domain_files:
                for file in sorted(os.listdir(chain_path)):
                    if file.endswith('.cif') and '_d_' in file:
                        domain_files.append(os.path.join(chain_path, file))

            if domain_files:
                domain_files.sort()

                self.chains[chain_name] = domain_files

                backup_chain_dir = os.path.join(self.backup_dir, chain_name)
                os.makedirs(backup_chain_dir, exist_ok=True)
                for df in domain_files:
                    shutil.copy2(df, backup_chain_dir)

                for idx, df in enumerate(domain_files):
                    domain_info = DomainInfo(chain_name, df, idx)
                    self.domain_infos[df] = domain_info

                print(f"\n  Chain {chain_name}: {len(domain_files)} domains")
                for idx, df in enumerate(domain_files):
                    print(f"    [{idx}] {os.path.basename(df)}")

        print(f"\n✅ Loaded {len(self.chains)} chains")
        print(f"✅ Original files backed up to: {self.backup_dir}\n")

    def step2_find_homologous_domains(self):
        print("=" * 80)
        print('Step 2: compute TM-score and identify homologous domains')
        print("=" * 80)

        all_domains = []
        domain_to_chain = {}

        for chain_name, domain_files in self.chains.items():
            for df in domain_files:
                all_domains.append(df)
                domain_to_chain[df] = chain_name

        tm_matrix = {}
        n = len(all_domains)
        print(f"Need to compute {((n * (n - 1)) // 2)} TM-score pairs...\n")

        for i in range(n):
            for j in range(i + 1, n):
                df1, df2 = all_domains[i], all_domains[j]
                chain1, chain2 = domain_to_chain[df1], domain_to_chain[df2]

                if chain1 != chain2:
                    name1 = os.path.basename(df1)
                    name2 = os.path.basename(df2)

                    rmsd, tm1, tm2 = _calculate_tm_score(df1, df2, self.usalign_path)
                    if tm1 is not None and tm2 is not None:
                        max_tm = min(tm1, tm2)
                        tm_matrix[(df1, df2)] = max_tm
                        tm_matrix[(df2, df1)] = max_tm

                        if max_tm >= self.tm_threshold:
                            print(f"  ✅ Homologous pair: {chain1}/{name1} ↔ {chain2}/{name2} (TM={max_tm:.3f})")

        parent = {df: df for df in all_domains}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for (df1, df2), tm_score in tm_matrix.items():
            if tm_score >= self.tm_threshold:
                union(df1, df2)

        groups = defaultdict(list)
        for df in all_domains:
            root = find(df)
            groups[root].append(df)

        group_id = 0
        for root, members in groups.items():
            if len(members) > 1:
                group_tm_scores = {}
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        key = (members[i], members[j])
                        if key in tm_matrix:
                            group_tm_scores[key] = tm_matrix[key]

                group = HomologousGroup(
                    group_id=group_id,
                    domains=[self.domain_infos[df] for df in members],
                    tm_scores=group_tm_scores
                )
                self.homologous_groups.append(group)
                group_id += 1

                print(f"\nHomologous group {group_id}:")
                for domain_info in group.domains:
                    print(f"  - {domain_info.chain_name}/{os.path.basename(domain_info.domain_file)}")

        print(f"\n✅ Identified {len(self.homologous_groups)} homologous domain groups\n")

        self._identify_fully_homologous_chains(tm_matrix)

    def _identify_fully_homologous_chains(self, tm_matrix: Dict):
        print("=" * 80)
        print('Step 2.5: identify fully homologous chains')
        print("=" * 80)

        chain_names = list(self.chains.keys())
        n_chains = len(chain_names)

        def are_chains_fully_homologous(chain1: str, chain2: str) -> bool:
            domains1 = self.chains[chain1]
            domains2 = self.chains[chain2]

            if len(domains1) != len(domains2):
                return False

            for i in range(len(domains1)):
                df1 = domains1[i]
                df2 = domains2[i]

                key1 = (df1, df2)
                key2 = (df2, df1)

                tm_score = tm_matrix.get(key1) or tm_matrix.get(key2)

                if tm_score is None or tm_score < self.tm_threshold:
                    return False

            return True

        parent = {chain: chain for chain in chain_names}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                chain1 = chain_names[i]
                chain2 = chain_names[j]

                if are_chains_fully_homologous(chain1, chain2):
                    union(chain1, chain2)
                    print(f"  ✅ Fully homologous chain pair found: {chain1} ↔ {chain2}")

        groups = defaultdict(list)
        for chain in chain_names:
            root = find(chain)
            groups[root].append(chain)

        for root, members in groups.items():
            if len(members) >= 2:
                self.fully_homologous_chain_groups.append(sorted(members))

        if self.fully_homologous_chain_groups:
            print(f"\n✅ Identified {len(self.fully_homologous_chain_groups)} fully homologous chain groups:")
            for idx, group in enumerate(self.fully_homologous_chain_groups, 1):
                print(f"  Group {idx}: {', '.join(group)}")
        else:
            print('\n  ℹ️  No fully homologous chain group found')

        print()

    def step3_generate_aligned_domains(self):
        print("=" * 80)
        print('Step 3: precompute aligned domain variants (CIF format)')
        print("=" * 80)

        total_aligned = 0

        for chain_name, domain_files in self.chains.items():
            print(f"\nProcessing chain: {chain_name}")
            print("-" * 60)

            for pos_idx, target_file in enumerate(domain_files):
                target_domain = self.domain_infos[target_file]
                target_basename = os.path.basename(target_file).replace('.cif', '')

                print(f"  Position {pos_idx}: {target_basename}")

                variant_0_filename = f"{target_basename}_0.cif"
                variant_0_filepath = os.path.join(self.aligned_dir, variant_0_filename)

                shutil.copy2(target_file, variant_0_filepath)

                aligned_original = AlignedDomain(
                    original_domain=target_domain,
                    target_domain=target_domain,
                    aligned_file=variant_0_filepath,
                    is_original=True,
                    display_name=f"{target_basename}_0",
                    variant_index=0
                )
                self.aligned_domains[(chain_name, pos_idx)].append(aligned_original)
                print(f"    ✅ Variant 0 (original): {variant_0_filename}")

                homolog_domains = self._get_homologous_domains(target_domain)

                if not homolog_domains:
                    print(f"    ℹ️  No homologous domains")
                    continue

                variant_idx = 1
                for homolog in homolog_domains:
                    if homolog.chain_name == chain_name:
                        continue

                    aligned_filename = f"{target_basename}_{variant_idx}.cif"
                    aligned_filepath = os.path.join(self.aligned_dir, aligned_filename)

                    try:
                        homolog_name = f"{homolog.chain_name}/{os.path.basename(homolog.domain_file)}"
                        print(f"    🔄 Variant {variant_idx}: aligned {homolog_name} -> {target_basename}")

                        calculate_and_align_with_sequence(
                            homolog.domain_file,
                            target_file,
                            aligned_filepath
                        )

                        aligned_domain = AlignedDomain(
                            original_domain=homolog,
                            target_domain=target_domain,
                            aligned_file=aligned_filepath,
                            is_original=False,
                            display_name=f"{target_basename}_{variant_idx}",
                            variant_index=variant_idx
                        )
                        self.aligned_domains[(chain_name, pos_idx)].append(aligned_domain)
                        total_aligned += 1
                        variant_idx += 1

                        print(f"    ✅ Generated: {aligned_filename}")

                    except Exception as e:
                        print(f"    ⚠️  Alignment failed: {e}")
                        continue

        print(f"\n✅ Generated {total_aligned} aligned variants (excluding the original copy)\n")

    def _get_homologous_domains(self, domain: DomainInfo) -> List[DomainInfo]:
        for group in self.homologous_groups:
            if domain in group.domains:
                return [d for d in group.domains if d != domain]
        return []

    def step4_enumerate_chain_configurations(self):
        print("=" * 80)
        print('Step 4: enumerate per-chain configurations and compute energies/scores in parallel')
        print("=" * 80)

        for chain_name in sorted(self.chains.keys()):
            print(f"\n{'=' * 60}")
            print(f"Processing chain: {chain_name}")
            print(f"{'=' * 60}")

            temp_chain_dir = os.path.join(self.work_dir, f"temp_{chain_name}")
            os.makedirs(temp_chain_dir, exist_ok=True)

            n_positions = len(self.chains[chain_name])
            position_options = []
            for pos_idx in range(n_positions):
                options = self.aligned_domains[(chain_name, pos_idx)]
                position_options.append(options)
                print(f"  Position {pos_idx}: {len(options)} variants")

            total_configs = 1
            for options in position_options:
                total_configs *= len(options)

            print(f"\n  Total configurations: {total_configs}")
            print(f"  Using {self.n_processes} worker processes...\n")

            all_configs = self._generate_chain_configurations(position_options)

            calc_args = [
                (config_idx, chain_config, temp_chain_dir, self.ideal_distance,
                 self.tight_tolerance, self.loose_min, self.loose_max)
                for config_idx, chain_config in enumerate(all_configs)
            ]

            config_results = []

            with Pool(processes=self.n_processes) as pool:
                results = pool.imap_unordered(calculate_config_energy_and_score, calc_args)

                for idx, result in enumerate(results, 1):
                    (config_idx, energy, score, score_details,
                     chain_config, config_files_info) = result
                    config_results.append((config_idx, energy, score,
                                           score_details, chain_config))

                    config_dir = os.path.join(temp_chain_dir, f"config_{config_idx}")
                    config_info = {
                        'config_index': config_idx,
                        'energy': energy,
                        'score': score,
                        'score_details': {
                            'tight_connections': score_details['tight_connections'],
                            'loose_connections': score_details['loose_connections'],
                            'no_score_connections': score_details['no_score_connections'],
                            'total_connections': score_details['total_connections']
                        },
                        'domains': config_files_info
                    }

                    with open(os.path.join(config_dir, "config_info.json"), 'w') as f:
                        json.dump(config_info, f, indent=2)

                    if idx % 100 == 0 or idx == total_configs:
                        print(f"  Progress: {idx}/{total_configs} ({((idx / total_configs) * 100):.1f}%)")

            config_results.sort(key=lambda x: (-x[2], x[1]))

            report_file = os.path.join(temp_chain_dir, "energy_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Chain {chain_name} energy and score report\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total configurations: {total_configs}\n")
                f.write(f"Domain positions: {n_positions}\n")
                f.write(f"Worker processes: {self.n_processes}\n\n")

                f.write('Top 100 configurations (descending score, ascending energy):\n')
                f.write("-" * 80 + "\n")

                for rank, (config_idx, energy, score, score_details, chain_config) in enumerate(config_results[:100],
                                                                                                1):
                    f.write(f"\nRank {rank}:\n")
                    f.write(f"  Configuration ID: config_{config_idx}\n")
                    f.write(f"  Score: {score:.2f} (tight:{score_details['tight_connections']}, loose:{score_details['loose_connections']}, invalid:{score_details['no_score_connections']}, total:{score_details['total_connections']})\n")
                    f.write(f"  Energy: {energy:.6f}\n")
                    f.write(f"  Configuration details:\n")

                    for pos_idx, aligned_domain in enumerate(chain_config):
                        if aligned_domain.is_original:
                            f.write(f"    Position {pos_idx}: {aligned_domain.display_name} (original)\n")
                        else:
                            f.write(f"    Position {pos_idx}: {aligned_domain.display_name} (from {aligned_domain.original_domain.chain_name}/{os.path.basename(aligned_domain.original_domain.domain_file)})\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write('Energy and scores for all configurations:\n')
                f.write("-" * 80 + "\n")

                for config_idx, energy, score, score_details, _ in config_results:
                    f.write(f"config_{config_idx}: score={score:.2f}, energy={energy:.6f}\n")

            print(f"\n  ✅ Chain {chain_name} processing completed")
            print(f"  📁 Configuration saved in: {temp_chain_dir}")
            print(f"  📄 Energy report: {report_file}")
            print(f"  🏆 Best configuration: config_{config_results[0][0]} (score={config_results[0][2]:.2f}, energy={config_results[0][1]:.6f})")

    def _generate_chain_configurations(self, position_options: List[List[AlignedDomain]]) -> List[List[AlignedDomain]]:

        def generate(pos_idx):
            if pos_idx == len(position_options):
                return [[]]

            sub_configs = generate(pos_idx + 1)
            all_configs = []

            for option in position_options[pos_idx]:
                for sub_config in sub_configs:
                    all_configs.append([option] + sub_config)

            return all_configs

        return generate(0)

    def step5_find_global_optimum(self):
        print("=" * 80)
        print('Step 5: find the globally optimal configuration combination')
        print("=" * 80)

        chain_configs = {}

        for chain_name in sorted(self.chains.keys()):
            temp_chain_dir = os.path.join(self.work_dir, f"temp_{chain_name}")
            configs = []

            for config_dir in sorted(os.listdir(temp_chain_dir)):
                if not config_dir.startswith("config_"):
                    continue

                config_path = os.path.join(temp_chain_dir, config_dir)
                info_file = os.path.join(config_path, "config_info.json")

                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        config_info = json.load(f)
                        configs.append(config_info)

            chain_configs[chain_name] = configs
            print(f"  Chain {chain_name}: {len(configs)} configurations")

        print()

        if self.fully_homologous_chain_groups:
            print('  🔍 Fully homologous chain groups found, using progressive optimization\n')
            self._progressive_optimization(chain_configs)
        else:
            print('  ℹ️  No fully homologous chain group, using global optimization\n')
            self._global_optimization(chain_configs)

    def _progressive_optimization(self, chain_configs: Dict):

        optimized_chains = {}
        occupied_positions = set()

        for group_idx, chain_group in enumerate(self.fully_homologous_chain_groups, 1):
            print(f"{'=' * 60}")
            print(f"Optimizing fully homologous chain group {group_idx}: {', '.join(chain_group)}")
            print(f"{'=' * 60}\n")

            for chain_name in chain_group:
                print(f"  Optimizing chain: {chain_name}")

                best_config = None
                best_score = -float('inf')
                best_energy = float('inf')

                for config in chain_configs[chain_name]:
                    if self._config_conflicts_with_occupied(config, chain_name, occupied_positions):
                        continue

                    if (config['score'] > best_score or
                            (config['score'] == best_score and config['energy'] < best_energy)):
                        best_score = config['score']
                        best_energy = config['energy']
                        best_config = config

                if best_config:
                    optimized_chains[chain_name] = best_config
                    self._update_occupied_positions(best_config, chain_name, occupied_positions)
                    print(f"    ✅ Selected configuration config_{best_config['config_index']}: score={best_score:.2f}, energy={best_energy:.6f}")
                    print(f"    📍 Occupied positions: {len(occupied_positions)}\n")
                else:
                    print(f"    ❌ No configuration satisfies the constraints\n")
                    return

        remaining_chains = [c for c in sorted(self.chains.keys()) if c not in optimized_chains]

        if remaining_chains:
            print(f"{'=' * 60}")
            print(f"Optimizing remaining chains: {', '.join(remaining_chains)}")
            print(f"{'=' * 60}\n")

            remaining_homologous_groups = self._find_homologous_groups_in_chains(remaining_chains)

            if remaining_homologous_groups:
                print(f"  🔍 Remaining chains still contain {len(remaining_homologous_groups)} fully homologous groups, continuing progressive optimization\n")

                optimized_remaining = set()
                for group in remaining_homologous_groups:
                    for chain_name in group:
                        if chain_name in optimized_remaining:
                            continue

                        print(f"  Optimizing chain: {chain_name}")

                        best_config = None
                        best_score = -float('inf')
                        best_energy = float('inf')

                        for config in chain_configs[chain_name]:
                            if self._config_conflicts_with_occupied(config, chain_name, occupied_positions):
                                continue

                            if (config['score'] > best_score or
                                    (config['score'] == best_score and config['energy'] < best_energy)):
                                best_score = config['score']
                                best_energy = config['energy']
                                best_config = config

                        if best_config:
                            optimized_chains[chain_name] = best_config
                            optimized_remaining.add(chain_name)
                            self._update_occupied_positions(best_config, chain_name, occupied_positions)
                            print(f"    ✅ Selected configuration config_{best_config['config_index']}: score={best_score:.2f}, energy={best_energy:.6f}")
                            print(f"    📍 Occupied positions: {len(occupied_positions)}\n")
                        else:
                            print(f"    ❌ No configuration satisfies the constraints\n")
                            return

                remaining_chains = [c for c in remaining_chains if c not in optimized_remaining]

            if remaining_chains:
                print(f"  ℹ️  Remaining {len(remaining_chains)} chains without complete homology, switching to global optimization\n")

                remaining_chain_configs = {c: chain_configs[c] for c in remaining_chains}
                best_remaining = self._optimize_remaining_chains(
                    remaining_chain_configs,
                    occupied_positions
                )

                if best_remaining:
                    optimized_chains.update(best_remaining)
                    print(f"  ✅ Remaining-chain optimization completed\n")
                else:
                    print(f"  ❌ Remaining-chain optimization failed\n")
                    return

        if len(optimized_chains) == len(self.chains):
            total_score = sum(c['score'] for c in optimized_chains.values())
            total_energy = sum(c['energy'] for c in optimized_chains.values())

            print(f"{'=' * 80}")
            print('✅ Progressive optimization completed!')
            print(f"{'=' * 80}\n")

            self._save_optimization_result(
                optimized_chains,
                total_score,
                total_energy,
                "progressive"
            )
        else:
            print('\n❌ Optimization failed: no valid configuration for all chains')

    def _global_optimization(self, chain_configs: Dict):
        chain_names = sorted(self.chains.keys())
        total_combinations = 1
        for chain_name in chain_names:
            total_combinations *= len(chain_configs[chain_name])

        print(f"  Total combinations: {total_combinations:,}")

        if total_combinations > 100000:
            print(f"  ⚠️  Many combinations; computation may take some time...")

        print('  Enumerating configurations and checking constraints...\n')

        best_combination = None
        best_score = -float('inf')
        best_energy = float('inf')
        valid_count = 0
        checked_count = 0

        def generate_combinations(chain_idx=0, current_combination=None,
                                  current_score=0, current_energy=0):
            nonlocal best_combination, best_score, best_energy, valid_count, checked_count

            if current_combination is None:
                current_combination = []

            if chain_idx == len(chain_names):
                checked_count += 1

                if checked_count % 10000 == 0:
                    print(f"  Progress: checked {checked_count:,}/{total_combinations:,} ({((checked_count / total_combinations) * 100):.1f}%) | Valid: {valid_count:,}")

                if self._check_global_constraints(current_combination, chain_names):
                    valid_count += 1
                    if (current_score > best_score or
                            (current_score == best_score and current_energy < best_energy)):
                        best_score = current_score
                        best_energy = current_energy
                        best_combination = [config.copy() for config in current_combination]
                        print(f"  🎯 Better configuration found! Score: {best_score:.2f}, energy: {best_energy:.6f} (iteration {valid_count} valid configurations)")
                return

            chain_name = chain_names[chain_idx]
            for config in chain_configs[chain_name]:
                current_combination.append(config)
                generate_combinations(
                    chain_idx + 1,
                    current_combination,
                    current_score + config['score'],
                    current_energy + config['energy']
                )
                current_combination.pop()

        generate_combinations()

        print(f"\n  ✅ Enumeration completed!")
        print(f"  Total combinations: {checked_count:,}")
        print(f"  Valid configurations: {valid_count:,}")
        if checked_count > 0:
            print(f"  Constraint satisfaction: {((valid_count / checked_count) * 100):.2f}%")

        if best_combination:
            print(f"  🏆 Best score: {best_score:.2f}")
            print(f"  🏆 Best energy: {best_energy:.6f}\n")

            optimized_chains = {}
            for i, chain_name in enumerate(chain_names):
                optimized_chains[chain_name] = best_combination[i]

            self._save_optimization_result(
                optimized_chains,
                best_score,
                best_energy,
                "global"
            )
        else:
            print('\n❌ No configuration satisfies the constraints!')

    def _find_homologous_groups_in_chains(self, chain_list: List[str]) -> List[List[str]]:
        groups = []
        processed = set()

        for i, chain1 in enumerate(chain_list):
            if chain1 in processed:
                continue

            current_group = [chain1]

            for chain2 in chain_list[i + 1:]:
                if chain2 in processed:
                    continue

                is_homologous_with_all = True
                for chain in current_group:
                    if not self._are_chains_fully_homologous(chain, chain2):
                        is_homologous_with_all = False
                        break

                if is_homologous_with_all:
                    current_group.append(chain2)
                    processed.add(chain2)

            if len(current_group) >= 2:
                groups.append(current_group)
                processed.add(chain1)

        return groups

    def _are_chains_fully_homologous(self, chain1: str, chain2: str) -> bool:
        for group in self.fully_homologous_chain_groups:
            if chain1 in group and chain2 in group:
                return True
        return False

    def _config_conflicts_with_occupied(self, config: Dict, chain_name: str,
                                        occupied_positions: Set[Tuple[str, int]]) -> bool:
        for domain in config['domains']:
            if domain['is_original']:
                key = (chain_name, domain['position'])
                if key in occupied_positions:
                    return True
            else:
                source_chain = domain['source_chain']
                source_file = domain['source_file']
                source_position = self._find_source_position(source_chain, source_file)

                if source_position != -1:
                    key = (source_chain, source_position)
                    if key in occupied_positions:
                        return True

        return False

    def _update_occupied_positions(self, config: Dict, chain_name: str,
                                   occupied_positions: Set[Tuple[str, int]]):
        for domain in config['domains']:
            if domain['is_original']:
                key = (chain_name, domain['position'])
                occupied_positions.add(key)
            else:
                source_chain = domain['source_chain']
                source_file = domain['source_file']
                source_position = self._find_source_position(source_chain, source_file)

                if source_position != -1:
                    key = (source_chain, source_position)
                    occupied_positions.add(key)

    def _optimize_remaining_chains(self, remaining_chain_configs: Dict,
                                   occupied_positions: Set[Tuple[str, int]]) -> Dict:
        if not remaining_chain_configs:
            return {}

        chain_names = sorted(remaining_chain_configs.keys())
        best_combination = None
        best_score = -float('inf')
        best_energy = float('inf')

        def generate_combinations(chain_idx=0, current_combination=None,
                                  current_score=0, current_energy=0):
            nonlocal best_combination, best_score, best_energy

            if current_combination is None:
                current_combination = []

            if chain_idx == len(chain_names):
                temp_occupied = occupied_positions.copy()
                valid = True

                for i, config in enumerate(current_combination):
                    chain_name = chain_names[i]
                    if self._config_conflicts_with_occupied(config, chain_name, temp_occupied):
                        valid = False
                        break
                    self._update_occupied_positions(config, chain_name, temp_occupied)

                if valid:
                    if (current_score > best_score or
                            (current_score == best_score and current_energy < best_energy)):
                        best_score = current_score
                        best_energy = current_energy
                        best_combination = [config.copy() for config in current_combination]
                return

            chain_name = chain_names[chain_idx]
            for config in remaining_chain_configs[chain_name]:
                current_combination.append(config)
                generate_combinations(
                    chain_idx + 1,
                    current_combination,
                    current_score + config['score'],
                    current_energy + config['energy']
                )
                current_combination.pop()

        generate_combinations()

        if best_combination:
            result = {}
            for i, chain_name in enumerate(chain_names):
                result[chain_name] = best_combination[i]
            return result

        return {}

    def _check_global_constraints(self, combination, chain_names):
        domain_usage = {}

        for i, config in enumerate(combination):
            chain_name = chain_names[i]

            for domain in config['domains']:
                position = domain['position']
                is_original = domain['is_original']

                if is_original:
                    key = (chain_name, position)
                    if key in domain_usage:
                        return False
                    domain_usage[key] = chain_name
                else:
                    source_chain = domain['source_chain']
                    source_file = domain['source_file']
                    source_position = self._find_source_position(source_chain, source_file)

                    if source_position == -1:
                        continue

                    key = (source_chain, source_position)
                    if key in domain_usage:
                        return False
                    domain_usage[key] = chain_name

        return True

    def _find_source_position(self, source_chain, source_file):
        if source_chain not in self.chains:
            return -1

        for idx, domain_file in enumerate(self.chains[source_chain]):
            if os.path.basename(domain_file) == source_file:
                return idx

        return -1

    def _save_optimization_result(self, optimized_chains: Dict, total_score: float,
                                  total_energy: float, method: str):
        result_file = os.path.join(self.work_dir, "global_optimal_solution.json")

        result_data = {
            'optimization_method': method,
            'total_score': total_score,
            'total_energy': total_energy,
            'chains': optimized_chains
        }

        if method == "progressive":
            result_data['fully_homologous_chain_groups'] = self.fully_homologous_chain_groups

        print(f"\n{'=' * 80}")
        print('🏆 Best configuration')
        print(f"{'=' * 80}\n")
        print(f"Optimization method: {('Progressive optimization' if (method == 'progressive') else 'Global optimization')}")
        print(f"Total score: {total_score:.2f}")
        print(f"Total energy: {total_energy:.6f}\n")

        for chain_name in sorted(optimized_chains.keys()):
            config = optimized_chains[chain_name]
            score_details = config['score_details']
            print(f"Chain {chain_name}:")
            print(f"  Configuration ID: config_{config['config_index']}")
            print(f"  Score: {config['score']:.2f} (tight:{score_details['tight_connections']}, loose:{score_details['loose_connections']}, invalid:{score_details['no_score_connections']}, total:{score_details['total_connections']})")
            print(f"  Energy: {config['energy']:.6f}")
            print(f"  Configuration details:")

            for domain in config['domains']:
                if domain['is_original']:
                    print(f"    Position {domain['position']}: {domain['display_name']} (original)")
                else:
                    print(f"    Position {domain['position']}: {domain['display_name']} (from {domain['source_chain']}/{domain['source_file']})")
            print()

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        report_file = os.path.join(self.work_dir, "global_optimal_solution.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write('Best-configuration report\n')
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization method: {('Progressive optimization' if (method == 'progressive') else 'Global optimization')}\n")
            f.write(f"Total score: {total_score:.2f}\n")
            f.write(f"Total energy: {total_energy:.6f}\n\n")

            if method == "progressive" and self.fully_homologous_chain_groups:
                f.write('Fully homologous chain groups:\n')
                for idx, group in enumerate(self.fully_homologous_chain_groups, 1):
                    f.write(f"  Group {idx}: {', '.join(group)}\n")
                f.write("\n")

            for chain_name in sorted(optimized_chains.keys()):
                config = optimized_chains[chain_name]
                score_details = config['score_details']
                f.write(f"{'=' * 60}\n")
                f.write(f"Chain {chain_name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"Configuration ID: config_{config['config_index']}\n")
                f.write(f"Score: {config['score']:.2f} (tight:{score_details['tight_connections']}, loose:{score_details['loose_connections']}, invalid:{score_details['no_score_connections']}, total:{score_details['total_connections']})\n")
                f.write(f"Energy: {config['energy']:.6f}\n")
                f.write(f"Configuration directory: temp_{chain_name}/config_{config['config_index']}\n\n")
                f.write('Configuration details:\n')

                for domain in config['domains']:
                    if domain['is_original']:
                        f.write(f"  Position {domain['position']}: {domain['display_name']} (original)\n")
                    else:
                        f.write(f"  Position {domain['position']}: {domain['display_name']} (from {domain['source_chain']}/{domain['source_file']})\n")
                f.write("\n")

        print(f"✅ Results saved:")
        print(f"  📄 JSON: {result_file}")
        print(f"  📄 TXT:  {report_file}")

    def step6_extract_optimal_structures(self):
        print("=" * 80)
        print('Step 6: extract best-configuration structures and assemble the complex (CIF format)')
        print("=" * 80)

        solution_file = os.path.join(self.work_dir, "global_optimal_solution.json")

        if not os.path.exists(solution_file):
            print('❌ Best-configuration file not found; run optimization first')
            return

        with open(solution_file, 'r', encoding='utf-8') as f:
            solution = json.load(f)

        final_results_dir = os.path.join(self.work_dir, "final_results")
        if os.path.exists(final_results_dir):
            shutil.rmtree(final_results_dir)
        os.makedirs(final_results_dir)

        print(f"✅ Created output directory: {final_results_dir}\n")

        total_files = 0

        for chain_name in sorted(solution['chains'].keys()):
            config = solution['chains'][chain_name]
            config_index = config['config_index']

            print(f"Processing chain {chain_name}:")
            print(f"  Configuration ID: config_{config_index}")

            config_dir = os.path.join(self.work_dir, f"temp_{chain_name}", f"config_{config_index}")

            if not os.path.exists(config_dir):
                print(f"  ⚠️  Configuration directory does not exist: {config_dir}")
                continue

            cif_files = [f for f in os.listdir(config_dir) if f.endswith('.cif')]

            if not cif_files:
                print(f"  ⚠️  No CIF files in the configuration directory")
                continue

            for cif_file in sorted(cif_files):
                src_file = os.path.join(config_dir, cif_file)


                if cif_file.startswith("pos_"):
                    parts = cif_file.split('_', 2)
                    if len(parts) >= 3:
                        remaining = parts[2]

                        if remaining.endswith('.cif'):
                            remaining = remaining[:-4]

                        name_parts = remaining.rsplit('_', 1)
                        if len(name_parts) >= 2:
                            new_name = name_parts[0]
                        else:
                            new_name = remaining

                        new_filename = f"{new_name}.cif"
                    else:
                        new_filename = cif_file
                else:
                    new_filename = cif_file

                dst_file = os.path.join(final_results_dir, new_filename)

                if os.path.exists(dst_file):
                    base_name = new_filename[:-4]
                    new_filename = f"{chain_name}_{base_name}.cif"
                    dst_file = os.path.join(final_results_dir, new_filename)
                    print(f"  ⚠️  Filename collision, adding chain prefix: {new_filename}")

                shutil.copy2(src_file, dst_file)

                print(f"  ✅ {cif_file} -> {new_filename}")
                total_files += 1

            print()

        print(f"{'=' * 80}")
        print(f"✅ Extraction completed! Copied {total_files} structure files (CIF format)")
        print(f"📁 Output directory: {final_results_dir}")
        print(f"{'=' * 80}\n")

        readme_file = os.path.join(final_results_dir, "README.txt")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write('Best-configuration structure files (CIF format)\n')
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization method: {solution['optimization_method']}\n")
            f.write(f"Total score: {solution['total_score']:.2f}\n")
            f.write(f"Total energy: {solution['total_energy']:.6f}\n\n")

            f.write('File format: mmCIF (.cif)\n\n')

            f.write('Filename conventions:\n')
            f.write('  Format: {domain_name}.cif\n')
            f.write('  Note: position prefixes (pos_X_) and variant suffixes (_Y) were removed\n')
            f.write('  Note: duplicate filenames receive a chain prefix\n\n')

            f.write('Per-chain configuration details:\n')
            f.write("-" * 80 + "\n\n")

            for chain_name in sorted(solution['chains'].keys()):
                config = solution['chains'][chain_name]
                score_details = config['score_details']

                f.write(f"Chain {chain_name}:\n")
                f.write(f"  Configuration ID: config_{config['config_index']}\n")
                f.write(f"  Score: {config['score']:.2f} ")
                f.write(f"(tight:{score_details['tight_connections']}, ")
                f.write(f"Loose:{score_details['loose_connections']}, ")
                f.write(f"Invalid:{score_details['no_score_connections']}, ")
                f.write(f"Total:{score_details['total_connections']})\n")
                f.write(f"  Energy: {config['energy']:.6f}\n")
                f.write(f"  Domain count: {len(config['domains'])}\n\n")

            f.write("=" * 80 + "\n")

        print(f"📄 Description file generated: {readme_file}\n")

        print("=" * 80)
        print('Step 6.5: assemble the final complex (CIF format)')
        print("=" * 80)

        try:
            from protassem.assembly.refine.complex_assembler import DomainComplexAssembler

            assembler = DomainComplexAssembler(
                input_dir=final_results_dir,
                output_filename="final.cif"
            )

            domain_files = assembler.find_domain_files()

            if not domain_files:
                print('⚠️  No valid domain files, skipping assembly')
            else:
                output_path = os.path.join(final_results_dir, "final.cif")
                result = assembler.assemble_complex(output_path=output_path)

                if result:
                    print(f"\n✅ Complex assembly succeeded!")
                    print(f"📁 Final complex file (CIF format): {result}")
                else:
                    print('\n⚠️  Complex assembly failed')

        except ImportError as e:
            print(f"⚠️  Cannot import DomainComplexAssembler")
            print(f"   Ensure zuzhuang_refine.py is in the same directory")
            print(f"   Error details: {e}")
        except Exception as e:
            print(f"⚠️  Complex assembly error: {e}")
            import traceback
            traceback.print_exc()

        print()

    def run_enumeration(self):
        print("\n" + "🔬" * 40)
        print('Protein-complex domain optimization with CIF support')
        print("🔬" * 40 + "\n")

        try:
            self.step1_load_chains_and_backup()
            self.step2_find_homologous_domains()

            if not self.homologous_groups:
                print('⚠️ No homologous domains; optimization is unnecessary')
                return

            self.step3_generate_aligned_domains()
            self.step4_enumerate_chain_configurations()
            self.step5_find_global_optimum()
            self.step6_extract_optimal_structures()

            print("\n" + "=" * 80)
            print('✅ Pipeline completed!')
            print("=" * 80)
            print(f"\n📁 Results saved in: {self.work_dir}")
            print(f"\n🎯 View the best configuration:")
            print(f"  - {os.path.join(self.work_dir, 'global_optimal_solution.json')}")
            print(f"  - {os.path.join(self.work_dir, 'global_optimal_solution.txt')}")
            print(f"  - {os.path.join(self.work_dir, 'final_results')} (best structure, CIF format)")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print('Usage: python chain_enumerator_cif.py <root_dir> [options]')
        print('\nOptions:')
        print('  --tm-threshold <float>   TM-score threshold (default: 0.85)')
        print('  --ideal-distance <float> Ideal CA-CA distance (default: 3.8)')
        print('  --tight-tolerance <float> Tight-connection tolerance (default: 1.5)')
        print('  --loose-min <float>      Minimum loose-connection distance (default: 5.3)')
        print('  --loose-max <float>      Maximum loose-connection distance (default: 20.0)')
        print('  --usalign <path>         USalign path')
        print('  --n-processes <int>      Worker processes (default: 20, capped at CPU count)')
        print('\nExample:')
        print("  python chain_enumerator_cif.py ./protein_complex --tm-threshold 0.7 --n-processes 20")
        print('\nNote:')
        print('  - Supports .pdb and .cif input files')
        print('  - .pdb files are converted to .cif automatically')
        print('  - All output files use .cif format')
        sys.exit(1)

    root_dir = sys.argv[1]

    tm_threshold = 0.75
    ideal_distance = 3.8
    tight_tolerance = 1.5
    loose_min = 5.3
    loose_max = 10
    usalign_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "core", "USalign")
    n_processes = 20

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--tm-threshold" and i + 1 < len(sys.argv):
            tm_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--ideal-distance" and i + 1 < len(sys.argv):
            ideal_distance = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--tight-tolerance" and i + 1 < len(sys.argv):
            tight_tolerance = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--loose-min" and i + 1 < len(sys.argv):
            loose_min = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--loose-max" and i + 1 < len(sys.argv):
            loose_max = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--usalign" and i + 1 < len(sys.argv):
            usalign_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--n-processes" and i + 1 < len(sys.argv):
            n_processes = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    enumerator = ChainEnumerator(
        root_dir=root_dir,
        usalign_path=usalign_path,
        tm_threshold=tm_threshold,
        ideal_distance=ideal_distance,
        tight_tolerance=tight_tolerance,
        loose_min=loose_min,
        loose_max=loose_max,
        n_processes=n_processes
    )

    enumerator.run_enumeration()


if __name__ == "__main__":
    main()