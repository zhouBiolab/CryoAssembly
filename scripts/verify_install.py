#!/usr/bin/env python3
"""Check the documented baseline, compiled operators and inference imports."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    expected = {
        "torch": "1.10.0+cu113", "numpy": "1.21.5", "scipy": "1.10.1",
        "numba": "0.56.4", "llvmlite": "0.39.1", "scikit-learn": "1.3.2",
        "biopython": "1.79", "mrcfile": "1.5.3", "open3d": "0.14.1",
        "easydict": "1.13", "tqdm": "4.65.0", "threadpoolctl": "3.5.0",
        "einops": "0.6.1", "ipdb": "0.13.13", "ipython": "8.12.3",
        "ninja": "1.11.1.4",
    }
    assert sys.version_info[:3] == (3, 8, 20), "Expected Python 3.8.20"
    actual = {package: version(package) for package in expected}
    assert actual == expected, "Dependency version mismatch: " + repr(actual)
    for library in ("libGL.so.1", "libgomp.so.1"):
        ctypes.CDLL(library)

    # torch must load its shared libraries before either extension is imported.
    import torch
    import numpy as np
    import numba
    import Bio
    import open3d
    import mrcfile
    import sklearn
    import einops
    import easydict
    import tqdm
    import threadpoolctl
    import IPython
    import ipdb
    import pointops_cuda
    import pareconv
    from pareconv.modules.ops import grid_subsample, index_select
    from pareconv.extensions.pointops.functions import pointops

    assert np.__version__ == expected["numpy"]
    assert numba.__version__ == expected["numba"]
    assert Bio.__version__ == expected["biopython"]
    assert torch.__version__ == expected["torch"] and torch.version.cuda == "11.3"
    vendored = root / "protassem/fitting/pareconv_src"
    Path(pareconv.__file__).resolve().relative_to(vendored.resolve())

    points = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]])
    subsampled, lengths = grid_subsample(points, torch.tensor([3]), 0.1)
    assert int(lengths.sum()) == len(subsampled) and len(subsampled) > 0
    assert torch.isfinite(subsampled).all()
    assert torch.cuda.is_available(), "CUDA unavailable"
    matrix = torch.eye(32, device="cuda")
    assert (matrix @ matrix).sum().item() == 32.0
    xyz = torch.rand(1, 256, 3, device="cuda").contiguous()
    indices = pointops.ballquery(0.5, 16, xyz, xyz)
    torch.cuda.synchronize()
    assert tuple(indices.shape) == (1, 256, 16)
    assert indices.is_cuda and int(indices.min()) >= 0 and int(indices.max()) < 256

    # Import the complete real inference entry point, not just one small operator.
    from protassem.fitting import demo_mask
    from protassem.fitting.parenet.model import create_model
    for rel in ("protassem/core/USalign", "protassem/sampling/Sample",
                "protassem/assembly/domain_parser/domainparser2.LINUX",
                "protassem/assembly/domain_parser/dssp"):
        binary = root / rel
        assert binary.is_file() and os.access(str(binary), os.X_OK), rel + " is not executable"
        check = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
        assert "not found" not in check.stdout + check.stderr, rel + " has missing shared libraries"
    checkpoint = root / "protassem/fitting/parenet/weights/epoch-18.pth.tar"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0, "Checkpoint missing"
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({"status": "VERIFY_OK", "python": sys.version.split()[0],
                      "packages": actual, "pareconv_path": pareconv.__file__,
                      "pointops_path": pointops_cuda.__file__,
                      "device": properties.name,
                      "device_memory_mib": properties.total_memory // (1024 * 1024),
                      "grid_subsample": "passed", "pointops_cuda_kernel": "passed",
                      "inference_import": "passed"}, indent=2))


if __name__ == "__main__":
    main()
