# Quant Grouped MatMul MX 样例

## 概述

本示例演示基于 Blaze 框架的 Quant Grouped MatMul MX 分组矩阵乘算子在昇腾 NPU 上的运行方式。样例覆盖 MXFP4/MXFP8 输入、weight ND/NZ 数据排布、M 轴分组以及 K 轴分组场景（具体约束参考下方约束），并通过 CPU golden 对 NPU 输出进行数值正确性校验。

- **算子**：quant_grouped_matmul
- **场景**：quant_grouped_matmul_mx
- **算法特点**：支持 MX 量化分组矩阵乘，覆盖 M 轴分组、K 轴分组（具体约束参考下方约束）、weight ND/DN/NZ/ZN、单 Tensor weight、NZ 多 Tensor weight、Bias、L1 双缓冲/三缓冲以及 A Full Load 策略。

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | 支持 |

## 功能说明

本样例支持以下功能：

- MXFP4 E2M1、MXFP4 E1M2、MXFP8 E4M3 和 MXFP8 E5M2 数据类型；
- weight ND 和 NZ 数据排布；
- M 轴分组，以及 K 轴分组场景（具体约束参考下方约束）；
- weight 非转置和转置场景，其中 ND/DN 表示普通二维 weight 及其转置布局，NZ/ZN 表示 FRACTAL_NZ weight 及其转置布局；
- 连续存储的单 Tensor weight，以及 NZ 场景下的多 Tensor weight 和 scale；
- MXFP4 E2M1 场景下的可选 Bias 输入；
- Length、Offset 和 Sparse 三种 Group List 类型；
- L1 双缓冲、L1 三缓冲、L0C 双缓冲和 A Full Load 策略。

样例使用固定随机种子生成可复现的非零 MX 数据，并采用以下流程验证数值正确性：

1. `../scripts/gen_data.py` 根据 CSV 参数生成输入数据和 NumPy `golden_c.bin`；
2. `quant_grouped_matmul_mx.cpp` 读取输入数据，在 NPU 上执行内核并写出 `npu_out.bin`；
3. `../scripts/verify_result.py` 使用相对误差和绝对误差比较 NPU 输出与 CPU golden。

M 轴分组场景按 M 维度拆分多个分组并拼接输出结果；K 轴分组场景按 K 维度拆分输入与 weight，在每个分组内完成分段矩阵乘并累加到输出结果。

精度校验由 CSV 的 `outputDtype` 字段驱动，当前覆盖的 MX 场景均支持 FP16、BF16 和 FP32 输出；CSV 因此按用例选择 `float16`、`bfloat16` 或 `float32`，数据生成、kernel 和校验脚本全链路使用同一类型。FP16/BF16 的单点绝对误差阈值及超阈值元素占比阈值均为 `1e-3`，FP32 均为 `1e-4`。

## 使用约束

- `groupNum`、`m`、`n` 和 `k` 必须为正整数；
- 当前样例中，A 为 ND 布局时对应 M 轴分组路径，A 为 DN 布局时对应 K 轴分组路径；
- weight 支持 ND、DN、NZ 和 ZN 数据排布，其中 DN 表示二维 weight 转置布局，NZ 表示 FRACTAL_NZ 格式，ZN 表示 FRACTAL_NZ weight 转置布局；
- weight ND/DN 场景仅支持单 Tensor weight；weight NZ/ZN 场景支持单 Tensor weight 和多 Tensor weight；
- 多 Tensor weight 的数量与分组数量一致，各 weight 的 K 轴和 N 轴分别相同；
- weight NZ/ZN 场景要求 `k` 和 `n` 均大于 1；
- MXFP4 E1M2 仅覆盖 weight NZ/ZN 场景；
- MXFP4（E2M1/E1M2）场景要求 `k` 为偶数且不等于 2；当 weight 为非转置布局（ND/NZ）时，还要求 `n` 为偶数；
- K 轴分组当前仅覆盖 `groupType=2`、A 为 DN 布局、MXFP8、单 Tensor weight ND、无 Bias、Length/Offset Group List 场景；
- L1 buffer 数量仅支持 2 或 3。

## CSV 驱动测试

测试用例通过 `quant_grouped_matmul_mx.csv` 配置。`QgmmTilingData` 为样例侧定义的 tiling 参数组织结构，其字段来源于 Blaze `GroupedMatmulWithScaleMx` 内核入口使用的 `gmmParams`，用于描述 Quant Grouped MatMul MX 内核运行所需的分组、shape、切分块、buffer 和 weight 存储模式等参数。CSV 中 `groupNum` 至 `singleW` 的字段名称和排列顺序与该结构体保持一致，`quant_grouped_matmul_mx.conf` 按该顺序将 CSV 字段传入 C++ 入口，再由 C++ 入口逐项传入 aicore kernel。

| 字段 | 说明 |
|------|------|
| `caseName` | 测试用例名称 |
| `groupNum` | 分组数量 |
| `m` | 每个分组的 M 维度 |
| `n` | N 维度 |
| `k` | K 维度 |
| `baseM` | L0 计算块的 M 大小 |
| `baseN` | L0 计算块的 N 大小 |
| `baseK` | L0 计算块的 K 大小 |
| `kAL1` | A 在 L1 中的 K 切分大小 |
| `kBL1` | weight 在 L1 中的 K 切分大小 |
| `scaleKAL1` | A scale 在 L1 中的 K 切分大小 |
| `scaleKBL1` | weight scale 在 L1 中的 K 切分大小 |
| `isBias` | 是否启用 Bias，0 表示关闭，1 表示启用 |
| `dbL0C` | L0C buffer 数量 |
| `l1BufferStage` | L1 buffer 数量，支持 2 或 3 |
| `groupType` | 分组轴类型，0 表示 M 轴分组，2 表示 K 轴分组 |
| `groupListType` | Group List 类型，0 表示 Offset，1 表示 Length，2 表示 Sparse |
| `singleW` | weight 存储模式，1 表示单 Tensor，0 表示多 Tensor |
| `dtype` | MX 数据类型 |
| `outputDtype` | 输出数据类型，支持 `float16`、`bfloat16` 和 `float32` |
| `layoutA` | A 的数据排布，支持 ND 和 DN |
| `layoutB` | weight 的数据排布，支持 ND、DN、NZ 和 ZN |
| `groupList` | Group List 原始值，以分号分隔；具体格式由 `groupListType` 决定 |
| `aFullLoad` | A 是否全载入，0 表示关闭，1 表示启用 |

### Group List 说明

`groupList` 描述输入和输出在分组轴方向上的矩阵乘大小分布。当前样例支持 M 轴分组和 K 轴分组：

- M 轴分组时，分组轴总长度为 `groupNum * m`；
- K 轴分组时，分组轴总长度为 `k`。

`groupListType` 支持以下三种取值：

| `groupListType` | 名称 | `groupList` 含义 | 接口约束 | CSV 示例 |
|-----------------|------|------------------|----------|----------|
| 0 | Offset | 长度为 E 的累积偏移序列，每个值表示对应分组的结束位置 | 数值非负且单调非递减，末值不大于分组轴总长度；相邻值相等表示对应分组长度为 0 | `13;31;51` |
| 1 | Length | 长度为 E 的分组大小序列，每个值表示对应分组的长度 | 数值非负，所有值之和不大于分组轴总长度；允许分组长度为 0 | `16;16;16` |
| 2 | Sparse | 逻辑 shape 为 `[E, 2]`，每行为 `[groupIndex, groupSize]`；CSV 按行展开为 `groupIndex;groupSize` | 仅全量化 M 轴分组支持；数值非负，非零分组前置，第二列之和不大于分组轴总长度 | `0;16;1;16` |

其中 E 表示 `groupNum`。以上取值范围为 GroupedMatmul V5 接口约束。当前样例的数据生成脚本为保证输入、weight 和 golden 完整对应，对 CSV 配置增加了以下限制：

- Offset、Length 均须包含 E 个值，Sparse 须包含 E 组 `[groupIndex, groupSize]`；
- 每个分组长度必须大于 0，所有分组长度之和必须等于对应分组轴总长度；
- Sparse 中的 `groupIndex` 必须构成 `[0, E)` 的一个排列；
- K 轴分组仅支持 Offset 和 Length，不支持 Sparse。

## 编译和运行

在代码仓根目录执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_mx
```

运行脚本会读取 `quant_grouped_matmul_mx.csv` 并批量执行测试用例，结果保存到 `quant_grouped_matmul_mx_result.csv`。每个用例的输入、CPU golden 和 NPU 输出保存到 `output/` 目录中；结果文件中的 `stage` 字段用于区分参数校验、数据生成、内核运行和精度校验阶段。

仅执行编译：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_mx --build-only
```

跳过编译并执行测试：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_mx --skip-build
```

## 内核单元测试

在代码仓根目录执行：

```bash
bash build.sh --opkernel -u --ops=quant_grouped_matmul
```

内核单元测试覆盖数据类型、数据格式、矩阵形状、M 轴分组、weight 转置、Group List、Bias、L0C 双缓冲、L1 三缓冲以及 NZ 多 Tensor weight 和 scale 等场景。单元测试用于检查各内核分支能否正常执行，数值正确性由 NPU 样例验证。

## 目录结构

```text
grouped_matmul/
├── scripts/
│   ├── gen_data.py      # 输入数据和 CPU golden 生成脚本
│   └── verify_result.py # NPU 输出精度校验脚本
└── quant_grouped_matmul_mx/
    ├── CMakeLists.txt   # 构建配置
    ├── quant_grouped_matmul_mx.cpp  # NPU 内核执行和输出落盘
    ├── quant_grouped_matmul_mx.conf # 参数路由配置
    ├── quant_grouped_matmul_mx.csv  # 测试用例配置
    └── README.md        # 使用说明
```

构建配置在 op 层 `examples/acl/grouped_matmul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/grouped_matmul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。
