<div align="center">

# CryoAssembly

**通过几何对应学习，从 cryo-EM 和 cryo-ET 密度图中装配蛋白结构**

![平台：Linux x86_64](https://img.shields.io/badge/platform-Linux_x86__64-347D83)
![Python：3.8](https://img.shields.io/badge/Python-3.8-347D83)
![CUDA：11.3](https://img.shields.io/badge/CUDA-11.3-347D83)

[English](README.md) · **简体中文**

[安装](#installation) · [使用](#usage) · [方法](#method) · [常见问题](#faq) · [引用](#citation)

</div>

## 概述

CryoAssembly 将输入的蛋白结构拟合并装配到冷冻电镜（cryo-EM）和冷冻电子断层成像（cryo-ET）密度图中。它结合基于深度特征的结构与密度图配准，以及链和域两个层级的装配，输出 mmCIF 格式的蛋白复合物模型。

核心模块 **CryoPARE-Net** 学习结构点云与密度图点云之间的几何对应关系，估计拟合姿态；随后通过密度引导的局部优化和域级装配，将输入组分放置到密度图中。运行时需要提供预测或实验获得的蛋白结构、密度图、分辨率和轮廓阈值；此命令行流程不包含从序列预测结构的步骤。

| 输入 | 输出 |
| :--- | :--- |
| 密度图、蛋白结构模型、分辨率和轮廓阈值 | 装配后的复合物模型（`.cif`）与装配摘要 |

<details>
<summary><strong>查看方法流程图</strong></summary>

<p align="center">
  <img src="docs/assets/workflow.png" alt="CryoAssembly 总体流程：密度图与链模型、滑动掩码与体素化、点云采样、CryoPARE-Net 拟合、局部优化及迭代装配。" width="400">
</p>

**总体流程。** 本图提取自对应 CryoAssembly 正文的 Fig. 1a。密度图和链模型转换为点云表示，用于结构与密度图拟合、局部优化和迭代装配。当前实现按拟合后的域接受组分；具体运行行为见下文及 [ALGORITHM.md](ALGORITHM.md)。

</details>

<a id="installation"></a>

## 安装

需要 **Linux x86_64**、可访问的 NVIDIA GPU、Conda 和兼容的 C/C++ 编译器。已验证的安装基线为 Ubuntu 20.04、GCC/G++ 9.4 和 CUDA 11.3 工具链；扩展默认编译目标为 A100/A800 等 Ampere GPU。

```bash
git clone --branch main --single-branch https://github.com/zhouBiolab/CryoAssembly.git
cd CryoAssembly
bash install.sh
conda activate cryoassembly
```

安装脚本创建独立的 `cryoassembly` 环境，安装固定版本的依赖，并编译仓库附带的 `pareconv` 与 `pointops` 扩展。`requirements.txt` 是唯一的 Python 依赖定义。

<details>
<summary><strong>系统要求与已验证环境</strong></summary>

安装需要兼容 CUDA 11.3 的 NVIDIA 驱动、编译工具、`libGL.so.1` 和 `libgomp.so.1`。在 Ubuntu 20.04 中，运行脚本前可以用以下命令补齐缺少的系统包：

```bash
sudo apt-get install build-essential libgl1 libgomp1
```

发现缺少前置依赖时，安装脚本会报错并停止，不会自动执行 `apt-get`。

| 组件 | 已验证基线 |
| :--- | :--- |
| 操作系统 | Ubuntu 20.04.6，Linux x86_64 |
| 编译器 | GCC/G++ 9.4.0 |
| Python | 3.8.20 |
| PyTorch | 1.10.0+cu113 |
| CUDA 工具链 / nvcc | 11.3.1 / 11.3.122 |
| NumPy / Numba / llvmlite | 1.21.5 / 0.56.4 / 0.39.1 |
| SciPy / Biopython / Open3D | 1.10.1 / 1.79 / 0.14.1 |

安装核验使用 A800 的 MIG `1g.10gb` 分配，PyTorch 可见显存为 9,728 MiB。这是实际核验的设备分配记录，不是所有输入通用的最低显存要求。显存需求受输入规模影响。其他 GPU 架构需要设置合适的 `TORCH_CUDA_ARCH_LIST`，不在本次已记录的验证基线内。

`nvidia-smi` 显示的 CUDA 版本代表驱动能力，与工具链编译器版本不同。脚本从官方 cu113 索引获取 PyTorch，并核对实际版本、扩展路径和导入情况。

</details>

<details>
<summary><strong>环境选择与安装核验</strong></summary>

```bash
bash install.sh myenv                  # 创建新环境
bash install.sh myenv --skip-env       # 在已有环境中安装依赖并重新编译
bash install.sh myenv --verify-only    # 仅核验已有安装，不安装或编译
bash install.sh --help
```

已有同名环境默认会被拒绝，除非显式使用 `--skip-env`。该模式要求已有环境具备 Python 3.8.20 和 CUDA 11.3 工具链。`--verify-only` 执行依赖、导入及小型算子核验，不运行完整装配任务。

`CONDA_BASE` 用于选择 Conda 安装位置。`MAX_JOBS` 控制扩展编译进程数，默认为 8。`PIP_INDEX_URL` 和 `PIP_FIND_LINKS` 可用于指定包镜像或准备好的 Python wheel 缓存；Python wheel 缓存不能替代 Conda 工具链包。

</details>

<a id="usage"></a>

## 使用

### 准备输入

创建目录并放入密度图及蛋白结构：

```text
data_dir/
├── map.mrc
├── model_A.pdb
├── model_B.cif
├── resolution.txt
└── contour_level.txt
```

`resolution.txt` 包含一个以 Å 为单位的分辨率数值；`contour_level.txt` 包含一个适合该密度图的轮廓阈值。上述文件名用于说明目录布局；这些命令不附带或下载示例数据。链标识从结构文件内容读取，不由文件名决定。

### 运行装配

在仓库目录中激活 `cryoassembly` 后运行：

```bash
python main.py data_dir --log
```

请保持域拆分开启。**如果结构文件包含多条链，需要添加 `--complex-domain-opt`。** 当前实现会拒绝没有域的组分，因此不应使用 `--no-domain-split` 来完成装配。

### 查看结果

默认输出目录位于输入密度图旁。主要结果文件为：

```text
data_dir/output/assembly/final_results/
├── assembled_complex.cif
├── assembled_complex_all.cif
├── refined_complex.cif          # 精修产生结果时存在
└── assembly_summary.txt
```

`assembled_complex.cif` 应用了最终域级 CC 过滤；`assembled_complex_all.cif` 包含全部已接受的装配组分。精修产生结果时，`refined_complex.cif` 保存精修后的模型。查看模型时请同时阅读摘要和日志。

<details>
<summary><strong>手动指定输入与输出目录</strong></summary>

手动模式支持 `.mrc` 和 `.map`；自动目录模式只发现 `.mrc`。依次指定密度图、结构目录、分辨率、轮廓阈值和可选输出目录：

```bash
python main.py data_dir/map.mrc data_dir 5.6 0.04 result --log
```

其中 `5.6` 和 `0.04` 是说明用数值，请替换为对应密度图的参数。手动模式不需要两个参数文本文件。该命令的最终结果位于 `result/assembly/final_results/`。

</details>

<details>
<summary><strong>多链输入与精修选项</strong></summary>

处理包含多条链的结构模板：

```bash
python main.py data_dir --complex-domain-opt --log
```

同源域精修默认开启。添加 `--homo-chain-refine` 可以开启其后的可选同源链精修阶段：

```bash
python main.py data_dir --complex-domain-opt --homo-chain-refine --log
```

使用 `--no-refine` 可以关闭同源域精修阶段。精修选项不会取消输入需要进行域拆分的要求。

</details>

<details>
<summary><strong>参数与显存控制</strong></summary>

| 选项 | CLI 默认值 | 含义 |
| :--- | :---: | :--- |
| `--chain-threshold` | 0.45 | 链拟合早停与预筛阈值；最终按域接受 |
| `--domain-threshold` | 0.45 | 独立域拟合的初始阈值 |
| `--complex-threshold` | 0.35 | 多链模板整体拟合的早停与预筛阈值 |
| `--domain-min-cc` | 0.35 | 自适应阈值的名义下限；仍有接受例外 |
| `--complex-min-cc` | 0.25 | 最终域级输出过滤阈值 |
| `--num-processes` | 8 | 评分与局部优化共用的 CPU 进程池大小 |
| `--batch-size` | 5 | 每批累积评估的候选数 |
| `--mask-radius-factor` | 1.35 | 掩码半径相对于源结构回转半径的倍数 |
| `--min-point-distance-factor` | 0.32 | 掩码中心间距相对于掩码半径的系数 |
| `--hypothesis-chunk` | 64* | 每块评分的位姿假设数，用于限制推理显存 |

*位姿假设分块的有效默认值通过运行配置设为 64。显式 CLI 值优先于 `--runtime-config`；设为 `0` 会关闭假设分块，但仍保留查询分块。CPU 进程数、候选批次与这项 GPU 显存控制相互独立。

使用 `python main.py --help` 查看完整选项。拟合阈值、调度方式和接受例外见 [ALGORITHM.md](ALGORITHM.md)。`--no-improve-accepted` 为兼容保留，但不控制已删除的整链接受后改善路径；保留重试辅助代码也不代表旧的重试生成路径仍然运行。

</details>

<a id="method"></a>

## 方法

1. **生成结构与密度图点云。** 通过体素化、采样和滑动密度掩码，准备局部结构与密度图配准输入。
2. **使用 CryoPARE-Net 估计拟合姿态。** 学习几何特征以建立对应关系并估计刚体姿态，随后进行密度引导的局部优化。
3. **通过链和域完成分层装配。** 链级拟合为域处理提供候选姿态，剩余域独立拟合；接受域时掩蔽其占据的密度区域。
4. **合并与精修。** 将已接受的域合并成链和复合物；同源域姿态重分配利用连接信息，随后通过输出过滤和可选精修生成最终模型。

当前代码使用附带的 DomainParser/DSSP 工具进行域划分。单域链可以直接进入域拟合。详细运行行为见 [ALGORITHM.md](ALGORITHM.md)。

<a id="faq"></a>

## 常见问题

<details>
<summary><strong>需要什么结构输入？</strong></summary>

需要提供结构预测或实验获得的 PDB/mmCIF 格式蛋白模型。该入口不从 FASTA 序列预测结构。当前按域装配的实现需要为多链结构文件开启 `--complex-domain-opt`。

</details>

<details>
<summary><strong>如何选择轮廓阈值？</strong></summary>

参考密度图的推荐轮廓阈值，并结合密度观察选择适合的值。这个参数决定点云采样时保留的密度范围；适合某张图的数值不一定适合其他图。

</details>

<details>
<summary><strong>为什么输入组分会被拒绝？</strong></summary>

检查日志以及是否生成了域。没有域的组分会被拒绝，包括通过 `--no-domain-split` 排除的链，以及未开启 `--complex-domain-opt` 的多链模板。调整拟合阈值之前，先检查输入准备是否正确。

</details>

<details>
<summary><strong>最低域 CC 是所有接受路径的硬保证吗？</strong></summary>

不是。平台期和轮末选择仍保留可能接受低于当前名义阈值的域的路径。链姿态逐域处理与独立域拟合的同源阈值也有差异。最终 `--complex-min-cc` 输出过滤是另一层步骤；请检查模型与密度图的吻合程度以及所得结构。

</details>

<details>
<summary><strong>GPU 显存不足时怎么办？</strong></summary>

检查进程实际可用的显存，尤其是 MIG 设备。位姿假设评分默认分块；减小正整数 `--hypothesis-chunk` 可降低该评分阶段的显存，但不能限制所有显存分配。`--num-processes` 控制 CPU 进程数。实际核验的 GPU 分配见安装要求。

</details>

<details>
<summary><strong>最终模型在哪里？</strong></summary>

查看 `<output_dir>/assembly/final_results/` 中的 `assembled_complex.cif`、装配摘要和日志；精修产生结果时，同时查看 `refined_complex.cif`。方法主要以刚体方式放置输入链和域，因此输入域内部的结构错误可能保留。

</details>

<a id="citation"></a>

## 引用

对应正文题目：**Accurate protein structure assembly from cryo-EM and cryo-ET maps through geometric correspondence learning**。

正式出版信息和核实后的 BibTeX 将在可用时补充。

## 致谢

CryoPARE-Net 基于 PARE-Net 及相关点云配准组件构建。实现中包含 `pareconv`、`pointops`、US-align、VoxEM 采样程序、DomainParser 和 DSSP。请根据实际使用情况致谢并引用相关原始方法，并查阅原始组件的署名信息和适用许可条款。
