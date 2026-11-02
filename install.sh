#!/usr/bin/env bash
# Install the supported CryoAssembly baseline without altering a system driver.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [ENV_NAME] [--skip-env | --verify-only]
       bash install.sh --help

Default environment: cryoassembly. Existing environments require --skip-env.
--skip-env     Install dependencies and build extensions in an existing env.
--verify-only  Check an existing installation; do not install or build anything.

Baseline: Linux x86_64, Python 3.8.20, CUDA toolkit 11.3, GCC/G++ 9.4,
          PyTorch 1.10.0+cu113. See README for system prerequisites.
Overrides: CONDA_BASE, TORCH_CUDA_ARCH_LIST (default 8.0), MAX_JOBS (default 8).
PIP_INDEX_URL / PIP_FIND_LINKS can select a package mirror / local wheel cache.
HELP
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ENV_NAME="${ENV_NAME:-cryoassembly}"
MODE=create
NAME_SEEN=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --skip-env|--verify-only)
            [ "$MODE" = create ] || fail 'Choose only one mode.'
            MODE="${arg#--}"
            ;;
        --*) fail "Unknown option: $arg" ;;
        *)
            [ "$NAME_SEEN" -eq 0 ] || fail 'Only one environment name is allowed.'
            ENV_NAME="$arg"
            NAME_SEEN=1
            ;;
    esac
done
[[ "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail 'Invalid environment name.'
[ "$ENV_NAME" != base ] || fail 'Do not install into the Conda base environment.'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] \
    || fail 'This installer supports Linux x86_64 only.'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
for item in requirements.txt environment.yml scripts/verify_install.py; do
    [ -f "$item" ] || fail "Missing repository file: $item"
done

if [ -n "${CONDA_BASE:-}" ]; then
    : # An explicit override is checked below; do not silently replace it.
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in "${CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda /opt/anaconda; do
        if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$candidate"
            break
        fi
    done
fi
[ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] \
    || fail 'Conda not found. Install Miniconda or set CONDA_BASE.'
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_EXISTS=0
if conda env list --json | "$CONDA_BASE/bin/python" -c \
    'import json, pathlib, sys; name=sys.argv[1]; sys.exit(0 if any(pathlib.Path(p).name == name for p in json.load(sys.stdin)["envs"]) else 1)' "$ENV_NAME"; then
    ENV_EXISTS=1
fi
if [ "$MODE" = create ]; then
    [ "$ENV_EXISTS" -eq 0 ] || fail "Environment '$ENV_NAME' already exists; use --skip-env explicitly or choose a new name."
else
    [ "$ENV_EXISTS" -eq 1 ] || fail "Environment '$ENV_NAME' does not exist."
fi

if [ "$MODE" != verify-only ]; then
    for command_name in gcc g++ make; do
        command -v "$command_name" >/dev/null 2>&1 \
            || fail 'Build tools missing. On Ubuntu 20.04: sudo apt-get install build-essential'
    done
    for compiler in gcc g++; do
        major="$("$compiler" -dumpversion | cut -d. -f1)"
        [[ "$major" =~ ^[0-9]+$ ]] && [ "$major" -ge 6 ] && [ "$major" -le 10 ] \
            || fail 'CUDA 11.3 requires a supported host compiler; use GCC/G++ 9.4 on Ubuntu 20.04.'
    done
    "$CONDA_BASE/bin/python" - <<'PY' || fail 'System libraries missing. On Ubuntu: sudo apt-get install libgl1 libgomp1'
import ctypes
for library in ("libGL.so.1", "libgomp.so.1"):
    ctypes.CDLL(library)
PY
fi
command -v nvidia-smi >/dev/null 2>&1 || fail 'NVIDIA driver tools not found.'
nvidia-smi -L >/dev/null || fail 'NVIDIA GPU is not accessible.'

CONDA_SPEC_TMP=""
trap 'if [ -n "$CONDA_SPEC_TMP" ]; then rm -f -- "$CONDA_SPEC_TMP"; fi' EXIT
if [ "$MODE" = create ]; then
    # Derive the Conda-only step from the same YAML. Install pip dependencies
    # separately so download progress and resolver errors are visible immediately.
    CONDA_SPEC_TMP="$(mktemp "$REPO_DIR/.conda-base-XXXXXX.yml")"
    "$CONDA_BASE/bin/python" - environment.yml "$CONDA_SPEC_TMP" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8")
marker = "  - pip:\n"
assert text.count(marker) == 1, "Expected one terminal pip section in environment.yml"
Path(sys.argv[2]).write_text(text.split(marker)[0], encoding="utf-8")
PY
    CONDA_CHANNEL_PRIORITY=strict conda env create --name "$ENV_NAME" --file "$CONDA_SPEC_TMP"
fi
conda activate "$ENV_NAME"
# Keep an externally injected source tree from masking the editable package.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python - <<'PY'
import sys
assert sys.version_info[:3] == (3, 8, 20), "Expected Python 3.8.20, got " + sys.version
PY

if [ "$MODE" != verify-only ]; then
    if [ "$MODE" = skip-env ]; then
        python -m pip install pip==24.3.1 setuptools==69.5.1 wheel==0.45.1
    fi
    python -m pip install -r requirements.txt
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    [ -x "$CUDA_HOME/bin/nvcc" ] || fail 'nvcc missing from CUDA_HOME; --skip-env requires the CUDA 11.3 toolkit.'
    "$CUDA_HOME/bin/nvcc" --version | grep -E 'release 11\.3([,[:space:]]|$)' >/dev/null \
        || fail 'CUDA_HOME must point to the CUDA 11.3 compiler toolchain.'
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    python - <<'PY'
import torch
assert torch.__version__ == "1.10.0+cu113", "Wrong PyTorch: " + torch.__version__
assert torch.version.cuda == "11.3", "Wrong PyTorch CUDA runtime"
PY
    python -m pip install -e "$REPO_DIR/protassem/fitting/pareconv_src" --no-build-isolation --no-deps
    python -m pip install "$REPO_DIR/protassem/fitting/pareconv_src/pareconv/extensions/pointops" --no-build-isolation --no-deps
    chmod +x protassem/core/USalign protassem/sampling/Sample \
        protassem/assembly/domain_parser/domainparser2.LINUX protassem/assembly/domain_parser/dssp
fi
python -m pip check
python -B "$REPO_DIR/scripts/verify_install.py"
printf '\nInstallation verified. Run:\n  conda activate %s\n  python main.py <data_dir> --log\n' "$ENV_NAME"
