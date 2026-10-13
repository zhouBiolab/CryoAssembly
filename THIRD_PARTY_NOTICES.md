# Third-party notices and license scope

## Original CryoAssembly contributions

Copyright (c) 2026 CryoAssembly contributors.

The original software contributions that the CryoAssembly copyright holders have authority to license are made available under GNU LGPL version 3.0 only (SPDX: `LGPL-3.0-only`). The full LGPL text is in [LICENSE](LICENSE); the GPL version 3 text incorporated by LGPL is in [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt).

This grant does not transfer ownership of, or create additional permissions in, third-party source code, binaries or model artifacts. Where a file contains material originating upstream, the grant applies only to original CryoAssembly contributions; the upstream material remains subject to its applicable rights and terms. Retain original copyright and license notices when redistributing.

## Component inventory

| Component | Location or provenance in this repository | License evidence and scope |
| --- | --- | --- |
| PARE-Net / pareconv and related registration components | `protassem/fitting/pareconv_src/` and PARE-Net-derived network code under `protassem/fitting/parenet/`; upstream: [yaorz97/PARENet](https://github.com/yaorz97/PARENet) | At the 2026-10-13 check, the public upstream default branch had no standalone LICENSE/COPYING/NOTICE file and GitHub reported no detected license. Some nested files have their own notices. Do not interpret publication, citation or this repository's LGPL grant as blanket permission for upstream material; upstream redistribution and reuse terms still require confirmation. |
| nanoflann | `protassem/fitting/pareconv_src/pareconv/extensions/extra/nanoflann/nanoflann.hpp` | The bundled header contains a two-condition BSD notice. Its original copyright, conditions and disclaimer remain in that file. The header's license applies to that component, not to all of pareconv. |
| pointops and other bundled extension code | `protassem/fitting/pareconv_src/pareconv/extensions/` | These are included with the vendored registration stack. This notice does not infer a standalone license for every extension; retain file-level notices and verify the upstream revision and terms for redistribution. |
| US-align | `protassem/core/USalign` | A bundled executable. Its provenance and the license applicable to this particular binary revision need to be recorded separately; the CryoAssembly LGPL grant does not relicense it. |
| CryoAlign sampling tool | `protassem/sampling/Sample`; source attribution: [HeracleBT/CryoAlign](https://github.com/HeracleBT/CryoAlign) | The project authors identify this bundled point-sampling tool as originating from CryoAlign. The permission for this particular bundled revision still needs confirmation. |
| DomainParser and DSSP | `protassem/assembly/domain_parser/`, including `domainparser2.LINUX` and `dssp` | The bundle includes upstream tools and an adapted wrapper. Verify the corresponding tool versions, licenses and any binary/source distribution obligations; do not infer the bundled versions' licenses from a different or newer upstream release. |
| Model checkpoint | `protassem/fitting/parenet/weights/epoch-18.pth.tar` | Provided model artifact. This software-license addition does not establish separate rights in model weights, training inputs or datasets. |
| Manuscript workflow illustration | `workflow.png` | Figure supplied by the CryoAssembly authors. It is documented separately from the software-license grant; this notice does not assign the image an additional content license. |

Python/Conda dependencies installed by `requirements.txt` or `environment.yml` retain their own licenses. Their presence as dependencies does not make them LGPL-licensed CryoAssembly code.

## License text references

The LGPL version 3 document is reproduced verbatim from the group's [DEMO-EMReF LICENSE](https://github.com/zhouBiolab/DEMO-EMReF/blob/main/LICENSE). Its README badge and actual LICENSE text differ; this repository follows the actual LGPL document. The incorporated GPL version 3 text is reproduced from [CryoHNF LICENSE](https://github.com/zhouBiolab/CryoHNF/blob/main/LICENSE). The references are for the license texts and do not imply that these projects' source code is included here.

This inventory records the evidence available for the bundled components. It is not a blanket third-party permission grant or a certification that every pending upstream permission has been resolved. Acknowledging or citing a method does not replace compliance with its software terms.

## 中文说明

LGPL-3.0-only 仅适用于 CryoAssembly 版权持有人有权授权的原创软件贡献。第三方源码、工具二进制、模型及论文图片不会因新增本许可证而获得新的授权。

当前仍需确认 PARE-Net 上游整体授权，以及所附 US-align、CryoAlign 采样工具、DomainParser、DSSP 等具体版本的许可与再分发条件。nanoflann 的 BSD 声明已保留在对应文件头。本文如实记录已知信息，不以论文引用、公开下载或本项目许可证替代第三方授权。
