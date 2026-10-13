<div align="center">

# CryoAssembly

**Protein structure assembly from cryo-EM and cryo-ET maps through geometric correspondence learning**

![Platform: Linux x86_64](https://img.shields.io/badge/platform-Linux_x86__64-347D83)
![Python: 3.8](https://img.shields.io/badge/Python-3.8-347D83)
![CUDA: 11.3](https://img.shields.io/badge/CUDA-11.3-347D83)
[![License: LGPL-3.0](https://img.shields.io/badge/License-LGPL--3.0-347D83)](LICENSE)

[**Open the CryoAssembly web server ↗**](http://zhouxglab.com/CryoAssembly/)

**English** · [简体中文](README.zh-CN.md)

[Web server](#web-server) · [Installation](#installation) · [Usage](#usage) · [Method](#method) · [FAQ](#faq) · [Citation](#citation) · [License](#license)

</div>

## Overview

CryoAssembly fits and assembles input protein structures into cryo-electron microscopy (cryo-EM) and cryo-electron tomography (cryo-ET) density maps. It combines deep feature-based structure-to-map registration with hierarchical chain- and domain-level assembly to produce a protein complex model in mmCIF format.

At its core, **CryoPARE-Net** learns geometric correspondences between structure-derived and map-derived point clouds to estimate fitting poses. Density-guided local optimization and domain-level assembly then position the input components within the map. Supply predicted or experimentally determined protein structures, a density map, its resolution and a contour level; structure prediction from sequence is outside this command-line workflow.

| Input | Output |
| :--- | :--- |
| Density map, protein structure models, resolution and contour level | Assembled complex models (`.cif`) and an assembly summary |

<details open>
<summary><strong>View the workflow</strong></summary>

<p align="center">
  <img src="workflow.png" alt="CryoAssembly workflow: density map and chain models, sliding masking and voxelization, point sampling, CryoPARE-Net fitting, local refinement and iterative assembly." width="400">
</p>

**Workflow.** Overall method illustration extracted from Fig. 1a of the associated CryoAssembly manuscript. A density map and chain models are converted to point-cloud representations for structure-to-map fitting, local refinement and iterative assembly. The current implementation accepts fitted domains; its operational details are described below and in [ALGORITHM.md](ALGORITHM.md).

</details>

<a id="web-server"></a>

## Web server

Use the [**CryoAssembly web server**](http://zhouxglab.com/CryoAssembly/) to submit your density map and input protein structures through a browser, without installing the local software. Follow the input instructions on the server's submission page.

The hosted service and this repository are maintained separately; refer to the server for its available options and submission limits. For a local run, follow the [installation instructions](#installation) below.

<a id="installation"></a>

## Installation

Use **Linux x86_64**, an accessible NVIDIA GPU, Conda and a compatible C/C++ compiler. The verified installation baseline is Ubuntu 20.04 with GCC/G++ 9.4 and the CUDA 11.3 toolchain; the default extension build targets Ampere GPUs such as A100/A800.

```bash
git clone --branch main --single-branch https://github.com/zhouBiolab/CryoAssembly.git
cd CryoAssembly
bash install.sh
conda activate cryoassembly
```

The installer creates a separate `cryoassembly` environment, installs the pinned dependencies and builds the bundled `pareconv` and `pointops` extensions. `requirements.txt` is the single Python dependency definition.

<details>
<summary><strong>System requirements and verified environment</strong></summary>

The installer requires an NVIDIA driver compatible with CUDA 11.3, build tools, `libGL.so.1` and `libgomp.so.1`. On Ubuntu 20.04, install missing system packages before running it:

```bash
sudo apt-get install build-essential libgl1 libgomp1
```

The installer reports missing prerequisites and stops; it does not run `apt-get` automatically.

| Component | Verified baseline |
| :--- | :--- |
| Operating system | Ubuntu 20.04.6, Linux x86_64 |
| Compiler | GCC/G++ 9.4.0 |
| Python | 3.8.20 |
| PyTorch | 1.10.0+cu113 |
| CUDA toolkit / nvcc | 11.3.1 / 11.3.122 |
| NumPy / Numba / llvmlite | 1.21.5 / 0.56.4 / 0.39.1 |
| SciPy / Biopython / Open3D | 1.10.1 / 1.79 / 0.14.1 |

Installation was verified on an A800 MIG `1g.10gb` allocation with 9,728 MiB visible to PyTorch. This records the tested allocation, not a universal minimum GPU-memory requirement. Input size affects memory usage. Other GPU architectures need an appropriate `TORCH_CUDA_ARCH_LIST` and are outside this recorded validation baseline.

The CUDA version reported by `nvidia-smi` describes driver capability; it is different from the toolkit compiler version. The installer obtains PyTorch from the official cu113 index and validates the installed versions, extension paths and imports.

</details>

<details>
<summary><strong>Environment selection and installation checks</strong></summary>

```bash
bash install.sh myenv                  # create a new environment
bash install.sh myenv --skip-env       # install dependencies and rebuild in an existing environment
bash install.sh myenv --verify-only    # check an existing installation without installing or building
bash install.sh --help
```

An existing environment is rejected unless `--skip-env` is explicit. That mode requires Python 3.8.20 and the CUDA 11.3 toolkit already present. `--verify-only` performs dependency, import and small operator checks; it does not run a full assembly job.

`CONDA_BASE` selects a Conda installation. `MAX_JOBS` controls extension-build workers (default 8). `PIP_INDEX_URL` and `PIP_FIND_LINKS` can select a package mirror or prepared Python wheel cache. A Python wheel cache does not replace the Conda toolkit packages.

</details>

<a id="usage"></a>

## Usage

### Prepare the inputs

Create a directory containing your map and protein structures:

```text
data_dir/
├── map.mrc
├── model_A.pdb
├── model_B.cif
├── resolution.txt
└── contour_level.txt
```

`resolution.txt` contains one resolution value in ångströms. `contour_level.txt` contains one density contour value appropriate for the map. The filenames above illustrate the layout; no example dataset is bundled or downloaded by these commands. Chain identifiers are read from structure contents rather than filenames.

### Run the assembly

From the repository directory, with `cryoassembly` activated:

```bash
python main.py data_dir --log
```

Keep domain splitting enabled. **For structure files containing multiple chains, add `--complex-domain-opt`.** Components with no domains are rejected in the current implementation; `--no-domain-split` is therefore unsuitable for successful assembly.

### Locate the results

The default output directory is beside the input map. The main result files are:

```text
data_dir/output/assembly/final_results/
├── assembled_complex.cif
├── assembled_complex_all.cif
├── refined_complex.cif          # when refinement produces a result
└── assembly_summary.txt
```

`assembled_complex.cif` applies the final domain-level CC filter; `assembled_complex_all.cif` contains the complete accepted assembly. If produced, `refined_complex.cif` contains the refinement result. Consult the summary and log alongside the model.

<details>
<summary><strong>Manual input and an explicit output directory</strong></summary>

Manual mode accepts `.mrc` or `.map`; automatic directory mode discovers `.mrc` only. Specify the map, structure directory, resolution, contour level and optional output directory:

```bash
python main.py data_dir/map.mrc data_dir 5.6 0.04 result --log
```

Here, `5.6` and `0.04` are illustrative values; replace them with the values for your map. Manual mode does not require the two parameter text files. The final results for this command are under `result/assembly/final_results/`.

</details>

<details>
<summary><strong>Multi-chain inputs and refinement options</strong></summary>

For a template containing multiple chains:

```bash
python main.py data_dir --complex-domain-opt --log
```

Homologous-domain refinement is enabled by default. Add `--homo-chain-refine` for the optional subsequent homologous-chain refinement stage:

```bash
python main.py data_dir --complex-domain-opt --homo-chain-refine --log
```

Use `--no-refine` to disable the homologous-domain refinement stage. Refinement does not remove the input requirement for domain splitting.

</details>

<details>
<summary><strong>Parameters and memory controls</strong></summary>

| Option | CLI default | Meaning |
| :--- | :---: | :--- |
| `--chain-threshold` | 0.45 | Chain-fitting early-stop and pre-screen threshold; acceptance is per domain |
| `--domain-threshold` | 0.45 | Initial independent-domain threshold |
| `--complex-threshold` | 0.35 | Multi-chain template early-stop and pre-screen threshold |
| `--domain-min-cc` | 0.35 | Nominal adaptive-threshold floor; acceptance exceptions remain |
| `--complex-min-cc` | 0.25 | Final domain-level output filter |
| `--num-processes` | 8 | Shared CPU process-pool size for scoring and local optimization |
| `--batch-size` | 5 | Candidates accumulated per evaluation batch |
| `--mask-radius-factor` | 1.35 | Mask radius relative to the source radius of gyration |
| `--min-point-distance-factor` | 0.32 | Mask-center spacing factor relative to the mask radius |
| `--hypothesis-chunk` | 64* | Hypotheses scored per block to bound inference memory |

*The effective hypothesis-chunk default is 64 through the runtime configuration. An explicit CLI value overrides `--runtime-config`; `0` disables hypothesis chunks while retaining query tiling. CPU workers and candidate batches are separate from this GPU-memory control.

For the complete option list, use `python main.py --help`. See [ALGORITHM.md](ALGORITHM.md) for fitting thresholds, scheduling and acceptance exceptions. `--no-improve-accepted` is retained for compatibility but does not control the removed accepted-chain improvement path; retained retry helpers do not imply that the old retry-generation path is active.

</details>

<a id="method"></a>

## Method

1. **Represent structures and maps as point clouds.** Voxelization, sampling and sliding map masks prepare local structure-to-map registration pairs.
2. **Estimate fitting poses with CryoPARE-Net.** Learned geometric features establish correspondences and support rigid-pose estimation, followed by density-guided local optimization.
3. **Assemble through chains and domains.** Chain-level fitting provides candidate poses for domain processing. Remaining domains are fitted independently, and occupied density is masked as domains are accepted.
4. **Merge and refine.** Accepted domains are assembled into chains and a complex; homologous-domain pose reassignment uses connectivity information. Output filtering and optional refinement provide the final models.

The current code uses the bundled DomainParser/DSSP tools for domain partitioning. Single-domain chains can enter domain fitting directly. Detailed runtime behavior is documented in [ALGORITHM.md](ALGORITHM.md).

<a id="faq"></a>

## FAQ

<details>
<summary><strong>What structure inputs do I need?</strong></summary>

Supply existing protein models in PDB or mmCIF format, obtained from structure prediction or experiment. This entry point does not predict a structure from a FASTA sequence. Multi-chain structure files require `--complex-domain-opt` in the current domain-based assembly implementation.

</details>

<details>
<summary><strong>How should I choose the contour level?</strong></summary>

Use a level appropriate for the density map, informed by its recommended contour and inspection of the density. It determines the density retained for point-cloud sampling. A value suitable for one map need not be suitable for another.

</details>

<details>
<summary><strong>Why can an input component be rejected?</strong></summary>

Check the log and whether domains were generated. Zero-domain components are rejected, including chains excluded by `--no-domain-split` and multi-chain templates processed without `--complex-domain-opt`. Check input preparation before changing fitting thresholds.

</details>

<details>
<summary><strong>Is the minimum domain CC a hard acceptance guarantee?</strong></summary>

No. Plateau and round-end selection retain paths that can accept domains below the current nominal threshold. Chain-pose and independent-domain homologous thresholds also differ. The final `--complex-min-cc` output filter is a separate step; inspect map-model agreement and the resulting structure.

</details>

<details>
<summary><strong>What if GPU memory is insufficient?</strong></summary>

Check the memory actually available to the process, especially on MIG devices. Hypothesis scoring is chunked by default; a smaller positive `--hypothesis-chunk` can reduce memory used by that scoring stage, but does not bound every allocation. `--num-processes` controls CPU workers. The tested GPU allocation is listed under installation requirements.

</details>

<details>
<summary><strong>Where is the final model?</strong></summary>

Look in `<output_dir>/assembly/final_results/`. Inspect `assembled_complex.cif`, the assembly summary and the log; if a refinement result was produced, also inspect `refined_complex.cif`. The method places input chains and domains predominantly as rigid units, so errors within an input domain can remain.

</details>

<a id="citation"></a>

## Citation

Associated manuscript: **Accurate protein structure assembly from cryo-EM and cryo-ET maps through geometric correspondence learning**.

Publication details and a verified BibTeX entry will be added when available.

<a id="license"></a>

## License

The original software contributions of the CryoAssembly copyright holders are licensed under the **GNU Lesser General Public License, version 3.0 only (LGPL-3.0-only)**. See [LICENSE](LICENSE) and the incorporated [GNU GPL version 3 terms](licenses/GPL-3.0.txt).

Third-party code, bundled executables and model artifacts are not relicensed by this grant. Their provenance, retained notices and unresolved upstream permissions are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This statement grants rights only in the original contributions that the CryoAssembly copyright holders have authority to license.

## Acknowledgements

CryoPARE-Net builds on PARE-Net and related point-cloud registration components. The implementation includes `pareconv` and `pointops`, US-align, the VoxEM sampling executable, DomainParser and DSSP. Please acknowledge and cite the relevant original methods when appropriate, and consult the original components for attribution and applicable licensing terms.
