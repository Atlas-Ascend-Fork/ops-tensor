# Kernel QGMM MIX Fixpipe Quant

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h)

## 功能说明

本文件的 `GemmUniversal` 特化支持两种 Grouped Matmul 路径，由 `BlockMmad::DispatchPolicy::ScheduleType` 在编译期选择。两条路径共用组表读取、M/N tile 遍历和 A/B 切片。

| 场景 | `ScheduleType` | 分组轴 | 反量化与输出 |
|------|----------------|--------|--------------|
| [S8S4 / S4S4 Fixpipe 量化](#fixpipe-量化) | `KernelGroupedMmadWithScaleFixpipeQuant` | M | Fixpipe 执行 per-channel 或 per-group 反量化并写入 FP16 workspace，AIV 按 Epilogue 的 `OutType` 写出最终结果 |
| [HiFloat8 T-C InplaceAdd](#hifloat8-t-c-inplaceadd) | `KernelGroupedMmadWithScaleMixInplaceAdd` | K | Fixpipe 将 FP32 结果写入 UB，AIV 反量化后累加到 FP32 YRef |

## Fixpipe 量化

该路径使用算子侧准备的 `int8_t` Cube 输入，支持以下功能。

- 支持 Grouped Matmul 的多专家计算，并支持 `groupListType` 0、1、2。
- 支持 per-channel 和 per-group 两种反量化模式。
- 复用 Fixpipe 完成每个输出块的 scale 反量化。
- 支持可选的非对称量化 offset 修正和 per-token scale 后处理。
- 对最后一个专家复用 GMM tail-split 调度，提高尾块的核利用率。

INT4 数据展开、A4W4 激活预处理和行归约不属于本组件，由算子实现侧在调用 Tensor API 前完成。

### 组件关系

```text
KernelGmmFixpipeQuant
    -> BlockSchedulerGmmSwatWithTailSplit
    -> BlockMmadA8W8FixpipeQuant
    -> BlockEpiloguePerTokenScale
```

### 调度策略

该 Kernel 由 `KernelGroupedMmadWithScaleFixpipeQuant` 调度标签选择。`BlockMmad` 必须使用相同的 `ScheduleType`，防止普通 QBMM、StreamK 和 GMM Fixpipe 组件发生错误组合。

Kernel 逐专家读取 M 轴分组信息，为当前专家更新问题规模，并按 Scheduler 返回的基本块坐标调用 BlockMmad。per-channel 与 per-group 的选择由 `GMMTiling::quantMode` 决定。

### 量化模式

#### Per-channel

每个专家对应一个长度为 N 的 scale 向量。BlockMmad 完成完整 K 轴累加后，通过 Fixpipe 应用当前 N 分片的 scale。

#### Per-group

Kernel 按 `quantGroupSize` 沿 K 轴切分当前输出块，为每段选取对应的 N 列 scale 并调用 BlockMmad。第一段由 Fixpipe 写入 FP16 workspace，后续段原子累加到同一位置。AIV 在所有段完成后读取 workspace。

### 主要参数

`Params` 包含以下字段。

| 参数 | 说明 |
|------|------|
| `mmadParams` | A、B、scale 及中间输出相关的 BlockMmad 参数 |
| `epilogueParams` | per-token scale、offset、row sum 和最终输出相关的后处理参数 |
| `groupListGmAddr` | 专家分组信息地址 |
| `gmmParams` | GMM 问题规模、基本块、量化模式及 group 配置 |

`GMMTiling` 中与本功能直接相关的字段包括 `groupNum`、`m/n/k`、`baseM/baseN/baseK`、`quantMode`、`quantGroupSize` 和 `groupListType`。

### 使用约束

- Kernel 编译期通过 `static_assert` 校验模板组合。
- `AType` / `BType` 仅支持 `int8_t`。
- Fixpipe scale 类型仅支持 `uint64_t`。
- workspace 的 `CType` 为 `half`，`BiasType` 为 `int32_t`。最终输出由 `BlockEpiloguePerTokenScale::OutType` 指定，支持 `half`、`bfloat16_t` 和 `float`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`。
- `LayoutB` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。
- `BlockEpilogue::FixpipeType` 需与 `CType` 一致。
- `quantMode` 仅使用 `PERCHANNEL_MODE` 或 `PERGROUP_MODE`。
- per-group 模式下，算子侧应保证 `quantGroupSize` 和 scale 的 group 维度与 K 轴切分一致。
- 当前 GMM 组合不在 Cube MMAD 中处理 API bias。

## HiFloat8 T-C InplaceAdd

Ascend 950 的 HiFloat8 T-C 路径沿 K 分组。设 `G` 为组数、`Ki` 为第 `i` 组的 K 长度。每组将反量化结果与已有的 `YRef[i]` 相加并写回同一位置。

```text
YRef[i] += (A_i @ B_i) * scale2[i, :] * scale1[i]
```

`A_i` 和 `B_i` 分别为 `[M, Ki]` 和 `[Ki, N]`，更新该组的 `[M, N]` 输出切片。

### 组件关系

| 组件 | 实例与职责 |
|------|------------|
| Policy | `MatmulWithScaleMix<NONE_FULL_LOAD_MODE, true, KernelGroupedMmadWithScaleMixInplaceAdd>` |
| BlockMmad | `block_mmad_a8w8_mix.h` 中的 `BlockMmad` 特化，按组刷新 K 循环参数 |
| Scheduler | `BlockSchedulerGmmSwatWithTailSplit`，逐组分配 M/N tile |
| Epilogue | `BlockEpilogueDequant<float, float, float, float, float>`，读取组标量和列 scale，反量化并原子累加到 GM |

### 调度策略

使用 `KERNEL_TYPE_MIX_AIC_1_2`。AIC 完成组内矩阵乘，Fixpipe 按 M 将结果送至两个 AIV 的 UB。AIV 先乘列 scale，再乘组标量，随后原子累加到 YRef。两个 AIV 应答后，AIC 可复用该结果区。Kernel 在 AIV 上开启和关闭 FP32 原子累加。

### 主要参数

| 参数 | 说明 |
|------|------|
| `mmadParams.aGmAddr` | HiFloat8 A 的 GM 地址，逻辑矩阵 `[M, K]`，DN 布局，按连续 `[K, M]` 存放 |
| `mmadParams.bGmAddr` | HiFloat8 B 的 GM 地址，ND `[K, N]` |
| `gmmParams.m/n/k` | 问题规模，K 为各组长度之和 |
| `gmmParams.baseM/baseN/baseK` | 基本块大小 |
| `mmadParams.problemShape/l0TileShape` | 与 `gmmParams` 一致的整体问题规模和基本块大小 |
| `mmadParams.kAL1/kBL1` | A/B 各自的 L1 K 搬运长度。调用者按 DN A / ND B 对齐核算单级 L1 占用 |
| `mmadParams.l1BufferNum/enableL0CPingPong` | L1 使用 `DOUBLE_BUFFER_COUNT`，`enableL0CPingPong` 控制 L0C 双缓冲 |
| `epilogueParams.x1ScaleGmAddr` | FP32 `[G]`，每组一个标量，`x1QuantMode = PERTENSOR_MODE` |
| `epilogueParams.x2ScaleGmAddr` | FP32 `[G, N]`，每组每列一个值，`x2QuantMode = PERCHANNEL_MODE` |
| `epilogueParams.outGmAddr` | 输入输出矩阵 YRef 的 GM 地址，FP32 ND `[G, M, N]` |
| `groupListGmAddr` | INT64 `[G]` 组表的 GM 地址 |
| `gmmParams.groupNum/groupListType` | 组数和组表编码 |

调用者用同一份 tiling 填写 `gmmParams`、`mmadParams` 和 `epilogueParams`，并设置 `epilogueParams.disableOutL2Cache = true` 以保持输出写回的 L2 bypass。Kernel 用 `gmmParams` 调度，直接用 `mmadParams` 和 `epilogueParams` 初始化 BlockMmad 和 Epilogue。bias 保持默认关闭。

### 使用约束

- `AtomicAdd_` 必须为 `true`。A 和 B 均为 `hifloat8_t`，A 为 DN，B 为 ND。BlockMmad 与 Epilogue 的 L0C 类型均为 `float`，Epilogue 输出为 FP32 ND。
- 组表按顺序描述连续存放的 K 切片。`groupListType = 0` 时，各项为累计 K，单调非降且末项等于 K。`groupListType = 1` 时，各项为组长度，非负且总和等于 K。
- 调用者在启动 Kernel 前初始化 YRef。K 长度为零的组保留调用前的 `YRef[i]`。

### 调用示例

完整的组件组装和参数赋值见 [样例源码](../../../../examples/acl/grouped_matmul/quant_grouped_matmul_inplace_add/quant_grouped_matmul_inplace_add.cpp)。编译、运行和精度校验见 [T-C 端到端样例](../../../../examples/acl/grouped_matmul/quant_grouped_matmul_inplace_add/README.md)。
