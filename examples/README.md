# Example inputs / 示例输入

Three example input sets are provided for CryoAssembly. Each archive contains only the case's original PDB structures, MRC map, `resolution.txt` and `contour_level.txt`. Extracting an archive restores the original filenames and file contents. Output folders, intermediate files and logs are excluded.

提供三组 CryoAssembly 示例输入。每个压缩包仅包含对应案例的原始 PDB 结构、MRC 密度图、`resolution.txt` 和 `contour_level.txt`，解压后保留原文件名与文件内容，不包含输出目录、中间文件或日志。

| Case | MRC map / 密度图 | PDB files / 文件数 | Resolution / 分辨率 (Å) | Contour / 轮廓阈值 | Download / 下载 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `8dme` | `EMD-27534.mrc` | 2 | 6.5 | 0.8 | [21.27 MiB](https://github.com/zhouBiolab/CryoAssembly/raw/refs/heads/main/examples/8dme.tar.gz) |
| `7ptt` | `EMD-13637.mrc` | 6 | 7.968 | 0.0242 | [2.04 MiB](https://github.com/zhouBiolab/CryoAssembly/raw/refs/heads/main/examples/7ptt.tar.gz) |
| `8fzc` | `EMD-29607.mrc` | 3 | 5.5 | 0.011 | [0.63 MiB](https://github.com/zhouBiolab/CryoAssembly/raw/refs/heads/main/examples/8fzc.tar.gz) |

The values above are read from the supplied TXT files. PDB file counts describe separate input files; internal chain identifiers are retained as supplied.

以上参数来自随附的 TXT 文件。PDB 数量表示独立输入文件数；结构内部的链标识保持原样。

## Extract / 解压

Run from the CryoAssembly repository directory. Archives are included in both a Git clone and GitHub's **Code → Download ZIP** archive.

在 CryoAssembly 仓库目录执行。Git 克隆和 GitHub 的 **Code → Download ZIP** 均包含这些输入包。

```bash
tar -xzf examples/8dme.tar.gz -C examples
tar -xzf examples/7ptt.tar.gz -C examples
tar -xzf examples/8fzc.tar.gz -C examples
```

For example, the first command creates:

例如，第一个命令会生成：

```text
examples/8dme/
├── EMD-27534.mrc
├── chain_a_1.pdb
├── chain_b_2.pdb
├── contour_level.txt
└── resolution.txt
```

## Run / 运行

Complete the [installation](../README.md#installation) first, then run a selected case from the repository directory:

先完成[安装](../README.zh-CN.md#installation)，再从仓库目录运行所选案例：

```bash
conda activate cryoassembly
python main.py examples/8dme --log
```

For either of the other inputs / 运行另一个案例：

```bash
python main.py examples/7ptt --log
python main.py examples/8fzc --log
```

Automatic directory mode reads the MRC and PDB files from the case directory and obtains resolution and contour level from the two TXT files. Keep the supplied filenames. Each input PDB contains one chain, so `--complex-domain-opt` is not required solely because several PDB files are provided.

自动目录模式从案例目录读取 MRC 和 PDB，并从两个 TXT 文件读取分辨率与轮廓阈值，请保留随附文件名。每个输入 PDB 均为单链，因此不必仅因为提供了多个 PDB 就添加 `--complex-domain-opt`。

Results are generated locally under `examples/<case>/output/assembly/final_results/`. See the main [usage guide](../README.md#usage) for output descriptions and optional parameters.

结果会在本地生成于 `examples/<case>/output/assembly/final_results/`。输出说明和可选参数见主文档的[使用说明](../README.zh-CN.md#usage)。

## Verify downloads / 校验下载

On Linux, verify the compressed archives with / 在 Linux 中可用以下命令校验压缩包：

```bash
(cd examples && sha256sum -c SHA256SUMS)
```
