#!/usr/bin/env python3

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PDB_URLS = {
    "cif": "https://files.rcsb.org/download/{pid}.cif",
    "pdb": "https://files.rcsb.org/download/{pid}.pdb",
}


def get_pdb_ids(npz_dir: str):
    ids = set()
    for f in os.listdir(npz_dir):
        if f.endswith(".npz") and len(f) >= 4:
            ids.add(f[:4].lower())
    return sorted(ids)


def download_one(pid: str, out_dir: str, fmt: str) -> str:
    url = PDB_URLS[fmt].format(pid=pid)
    out_path = os.path.join(out_dir, f"{pid}.{fmt}")
    if os.path.exists(out_path):
        return f"[Skipped] {pid}.{fmt} already exists"
    try:
        urllib.request.urlretrieve(url, out_path)
        return f"[Done] {pid}.{fmt}"
    except Exception as e:
        return f"[Failed] {pid}.{fmt} - {e}"


def main():
    parser = argparse.ArgumentParser(description='Extract PDB IDs from npz filenames and download structures')
    parser.add_argument("npz_dir", help='Directory containing .npz files')
    parser.add_argument("out_dir", help='Output directory for downloaded files')
    parser.add_argument("--format", choices=["pdb", "cif", "both"], default="cif",
                        help='Download format (default: cif)')
    parser.add_argument("--workers", type=int, default=32, help='Parallel download threads (default: 4)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_ids = get_pdb_ids(args.npz_dir)
    print(f"Found {len(pdb_ids)} PDB IDs: {', '.join(pdb_ids[:10])}{('...' if (len(pdb_ids) > 10) else '')}")

    fmts = ["pdb", "cif"] if args.format == "both" else [args.format]
    tasks = [(pid, args.out_dir, fmt) for pid in pdb_ids for fmt in fmts]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, *t): t for t in tasks}
        for future in as_completed(futures):
            print(future.result())

    print('Completed!')


if __name__ == "__main__":
    main()