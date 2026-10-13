# CryoAssembly algorithm

This document describes the implementation distributed on `main`. Installation
instructions and the tested environment are in [README.md](README.md).

## Inputs and preparation

The pipeline takes an experimental density map, PDB/CIF structure templates,
resolution and contour level. Automatic directory mode discovers `.mrc` files;
manual mode accepts `.mrc` or `.map`. Structure chain IDs are read from contents.
Input normalization preserves multi-chain templates and resolves duplicate IDs;
the assembly entry also prevents duplicate internal component IDs.

Templates are voxelized and sampled into point clouds. Structural domains are
identified with the bundled DomainParser/DSSP tools. Multi-chain templates need
`--complex-domain-opt` to supply domains for this domain-based assembly version.

## Fitting and scheduling

PARE-Net proposes rigid poses for a template and the remaining density. Candidate
CC-mask scores are evaluated with the shared CPU pool, followed by local density
optimization. CLI defaults are eight CPU workers and five candidates per
monitoring batch; neither is a GPU count or a model batch size.

1. Pending chains are considered first, ordered by descending radius of gyration.
   Single-domain ordinary chains can be routed directly to domain fitting.
2. A fitted multi-domain chain supplies a pose for its domains. With domain
   optimization enabled, domains are aligned to that pose, optimized and
   individually accepted when their chain-pose threshold is reached. Accepted
   domains are masked immediately, even during the chain phase.
3. Once the chain queue finishes, remaining available domains are fitted in
   global rounds. Larger domains and domains homologous to recently accepted
   domains receive priority according to the current queue logic.
4. Accepted domains from either phase enter domain-chain assembly. Parent chains
   missing from the assembly list are added automatically.

Whole chains are not directly accepted by normal fitting/pre-screening paths.
The chain threshold still controls fitting early stop and pre-screen selection.
Pre-screening identifies original-pose candidates and routes them to domains.

## Thresholds and candidates

| Setting | CLI default | Role |
|---|---:|---|
| chain threshold | 0.45 | Whole-chain early stop and pre-screen selection |
| initial domain threshold | 0.45 | Initial independent-domain threshold |
| domain minimum | 0.35 | Nominal threshold-adaptation floor |
| complex threshold | 0.35 | Multi-chain template early stop/pre-screen selection |
| similarity threshold | 0.85 | TM-score grouping |
| complex minimum CC | 0.25 | Final per-domain output filtering |
| refinement TM | 0.75 | Step-4 homologous-domain grouping |

Homologous thresholds use the minimum recorded accepted-group CC. With no
accepted member, the base threshold applies. A group minimum >=0.55 uses the
fixed high-band threshold 0.50. Chain fitting also uses its low-band relaxation;
independent domain fitting otherwise retains fixed domain relaxation and the
nominal minimum. The independent-domain effective threshold controls both early
stop and acceptance.

Chain-pose domain handling retains a separate threshold: initial domain threshold,
or initial threshold minus domain homologous relaxation with the nominal minimum.
It does not use the independent-domain high-band rule.

Ordinary and masked sampling tables both enable configurations 0-4, each with
voxel and FPS final-stage sampling. Configuration and sampling mode are passed
explicitly into geometry construction rather than silently using global config 0.

Without early stop, final optimization takes the top ten unoptimized candidates
by CC and up to ten additional candidates by hybrid score. It excludes CC<0.03
and already optimized files; ties use stable candidate IDs. Existing best-optimized
and raw-candidate fallbacks remain. This shortlist rule does not guarantee that
all raw fallback results pass the fitting threshold.

## Assembly and refinement

Accepted domains are merged in residue order and restored to final chain-ID
space. `assembled_complex_all.cif` contains the complete accepted assembly;
`assembled_complex.cif` applies domain-level output filtering. Optional homologous
domain and chain refinement retain their existing gates and output priorities.

Final filtering is separate from fitting acceptance. Hypothesis scoring uses
blocks of 64 by default and query tiling remains enabled; these memory controls
do not change the CPU worker count.

## Current compatibility limits

- Zero-domain components are rejected, including `--no-domain-split` inputs and
  unsplit multi-chain templates. Keep splitting enabled and use
  `--complex-domain-opt` for multi-chain templates.
- Round-end selection and plateau acceptance retain branches that may accept
  domains below the nominal current/minimum threshold. The minimum is not a
  universal hard acceptance guarantee.
- Chain-pose and independent-domain homologous thresholds are not fully unified.
- `--no-improve-accepted` remains parseable, but the former accepted-chain
  improvement function is removed. Retry helpers do not mean the removed
  retry-generation path is active.

These limits are retained. Installation changes do not repair them or guarantee
identical numerical results across dependency environments.
