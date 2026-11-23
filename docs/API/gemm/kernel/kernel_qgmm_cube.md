# Kernel Qgmm Cube
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_cube.h)

## 功能说明
Fixpipe 量化 Grouped Matmul Kernel，仅支持 AIC 计算。该 Kernel 组合 `BlockMmadA8W8FixpipeQuant` 与 `BlockSchedulerGmmSwatWithTailSplit`，按 group list 逐 group 完成量化 A/B 矩阵乘、Bias 处理、scale 反量化与 GM 输出。输出覆盖模式支持 SPLIT_M（groupType=0）和 NO_SPLIT（groupType=-1），HiFloat8 per-tensor 输出原子累加使用 SPLIT_K（groupType=2）。SPLIT_M 时每个 group 的 M 尺寸由 group list 动态给出，group 间通过 SWAT 调度器保持核负载均衡与双缓冲相位连续。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
支持 Tensor API Cube Mmad/Fixpipe 路径可处理的量化输入类型，典型包括：
- `int8_t`：A8W8 输入，L0C 通常累加为 `int32_t`
- `hifloat8_t`：HiFloat8 输入，L0C 通常累加为 `float`
- `fp8_e4m3fn_t`、`fp8_e5m2_t`：FP8 输入，支持 E4M3FN/E5M2 同型或混合组合，L0C 通常累加为 `float`
- C 输出由 `CType` 指定，支持 `half`、`bfloat16_t`、`float` 或 `int32_t`，最终以 Tensor API Fixpipe 静态检查为准

### Scale 因子要求
Scale 模式由 `x1QuantMode` 和 `x2QuantMode` 决定，取值来自 `QuantMode`（见 [Kernel Qbmm Cube](./kernel_qbmm_cube.md) 的枚举表）。

`x1QuantMode` 描述 A 矩阵的 scale 模式，`x2QuantMode` 描述 B 矩阵的 scale 模式。当前 Kernel 支持以下组合：

| A 矩阵模式 (`x1QuantMode`) | B 矩阵模式 (`x2QuantMode`) | 适用场景与 Scale 处理 |
|----------------------------|----------------------------|-----------------------|
| `DEFAULT` | `PERCHANNEL_MODE` | A 不提供 scale，B 的每个 N 通道分别使用一个 scale（每个 group 拥有 `n` 个 uint64）。`scaleBGmAddr` 指向 B 的 per-channel scale 数组；Kernel 按 group 偏移 `groupIdx * n`、按当前 block 的 N 范围截取对应 scale Tensor 并传给 Block，Block 将 scale 搬入 L1，Fixpipe 搬出结果时按通道完成反量化。 |
| `DEFAULT` | `PERTENSOR_MODE` | A 不提供 scale，每个 group 各自共用一个 B 侧 scalar scale。Kernel 在 `ProcessSingleGroup` 中按 group 从 `scaleBGmAddr + groupIdx` 读取该值，将其转换并封装为 Fixpipe 使用的 `uint64_t scaleScalar_`，再传给 Block。支持的 scale 存储类型为 `uint64_t/int64_t`、`bfloat16_t` 或 `uint32_t`。 |
| `PERTENSOR_MODE` | `PERTENSOR_MODE` | A、B 每个 group 各自提供一个 per-tensor scalar scale，A 侧存储类型为 `float`（来自 `perTokenScale` 输入，即 `scaleAGmAddr`），B 侧支持 `float` 或 `bfloat16_t`，bf16 先转换为 float。Kernel 分别读取 `scaleBGmAddr + groupIdx` 与 `scaleAGmAddr + groupIdx`，计算两者乘积，封装为 `uint64_t scaleScalar_` 后传给 Block。 |

其他组合当前模板未提供对应 scale 数据流。

当输出 `CType = int32_t` 时，无论传入 Block 的是 scalar scale 还是 per-channel scale，Block 都不会使用 scale，也不会执行 Fixpipe 反量化，而是将 L0C 中的 `int32_t` 累加结果直接写入 GM。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核在 `Run`/`Init` 中直接返回）。

### BlockMmad 限制
仅支持调度策略为 `MatmulWithScaleFixpipeQuant` 的 `BlockMmad`，即 `BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelGroupedMmadFixpipeQuant`。该标签通过 `MatmulWithScaleFixpipeQuant` 的第 3 个模板参数 `ScheduleType_` 指定。

### BlockScheduler 限制
使用 `BlockSchedulerGmmSwatWithTailSplit` 调度器按 group 连续串接 block，并按 SWAT 窗口分块。覆盖模式和原子累加模式的末组均按调度条件调用 `UpdateTailTile`，在 M/N 方向切分尾块。

### groupList 布局
NO_SPLIT 允许 `groupListGmAddr` 为空。SPLIT_M 和 SPLIT_K 使用 INT64 组表，OFFSET 和 LENGTH 的长度均为 `groupNum`。SPARSE 只用于 SPLIT_M，长度为 `2 * groupNum`。
- `0`（OFFSET）：第 i 个元素为该 group 的累计 offset，split 值为差分；
- `1`（LENGTH）：第 i 个元素为该 group 的 split 长度；
- `2`（SPARSE）：每 2 个 int64 一项，`[tensorIdx, splitValue]`。

## 模板参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `ProblemShape` | `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` | 全局 M/N/K 形状，split 轴在 group 循环中按 group list 刷新 |
| `BlockMmad` | `Blaze::Gemm::Block::BlockMmad<...>` | 计算块，`DispatchPolicy::ScheduleType` 必须为 `KernelGroupedMmadFixpipeQuant` |
| `BlockEpilogue` | `Blaze::Epilogue::Block::BlockEpilogueEmpty` | 后处理（AIC_ONLY 场景为空） |
| `BlockScheduler` | `Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit` | 调度器 |

调用方通过 `MatmulWithScaleFixpipeQuant` 的第 4 个模板参数指定 GMMArray 指针类型。普通 GM 输入可省略该参数；常量 Tiling 输入需传入 `const int32_t*`，也可以使用 `decltype(gmmArrayAddr)` 保留实际指针类型。Kernel 不判断算子侧的 Tiling 宏，也不转换地址空间。

第 6 个模板参数 `IsTensorList_` 决定调用方如何传入 A、B、C（对应 `aGmAddr`、`bGmAddr`、`cGmAddr`）。

| 参数值 | 调用方传入的地址 | Kernel 取址方式 |
| --- | --- | --- |
| `true`（默认） | 各自的 TensorList 描述符地址 | 通过 `GetDataPtr(0)` 取得列表中第一个 Tensor 的数据地址 |
| `false` | 每个 Tensor 的 GM 数据首地址 | 直接使用传入的数据地址 |

只含一个 Tensor 的 TensorList 也按描述符传入，使用 `true`。有 bias 时，`biasGmAddr` 传 TensorList 描述符地址，两组 scale 传各自的数据地址。`AtomicAdd_` 独立控制输出是否使用原子累加。

## 数据结构

### `GmmParams`
Kernel 的 group 级参数（聚合初始化，字段顺序敏感）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `groupNum` | `uint32_t` | group 数 |
| `m` / `n` / `k` | `int64_t` | 全局形状 |
| `baseM` / `baseN` / `baseK` | `uint32_t` | L0 分块大小 |
| `kAL1` / `kBL1` | `uint32_t` | A/B 的 K 方向 L1 窗口（分别为 `stepKa * baseK`、`stepKb * baseK`） |
| `x1QuantMode` / `x2QuantMode` | `uint32_t` | A/B 侧 QuantMode |
| `isBias` | `uint8_t` | 是否带 bias |
| `dbL0C` | `uint8_t` | L0C 双缓冲使能（>1 启用） |
| `groupType` | `int8_t` | 覆盖模式使用 -1（NO_SPLIT）或 0（SPLIT_M），HiFloat8 per-tensor 原子累加使用 2（SPLIT_K） |
| `groupListType` | `uint8_t` | 0=OFFSET，1=LENGTH，2=SPARSE（仅 SPLIT_M） |
| `singleW` / `singleX` / `singleY` | `uint8_t` | 保留兼容字段；singleW 参与尺寸选择，不表示支持独立多 tensor 寻址 |

### `Params`

| 字段 | 说明 |
|------|------|
| `problemShape` | 全局 ProblemShape |
| `mmadParams` | `BlockMmad::Params`，6 个 GM_ADDR（a/b/c/bias/scaleA/scaleB） |
| `epilogueParams` | `BlockEpilogueEmpty::Params`，空 |
| `groupListGmAddr` | group list 地址 |
| `gmmArrayGmAddr` | GMMArray 的 mList/kList/nList 地址；类型由 `DispatchPolicy::GmmArrayPtr` 指定，默认是 `__gm__ int32_t*` |
| `gmmParams` | `GmmParams` |

### HiFloat8 T-T 原子累加

使用 `MatmulWithScaleFixpipeQuant` 和 `BlockEpilogueEmpty`。策略的第 2 个模板参数 `AtomicAdd_` 设为 `true`，第 3 个参数使用 `KernelGroupedMmadFixpipeQuant`，第 6 个参数 `IsTensorList_` 设为 `false`。A（HiFloat8 DN）、B（HiFloat8 ND）和输出 `YRef[G, M, N]`（FP32 ND）各传单个 Tensor 的 GM 数据首地址，两组 FP32 per-tensor scale 各传数据地址。设置 `groupType = 2`，A/B 的量化模式均为 `PERTENSOR_MODE`，`isBias = 0`，`gmmArrayGmAddr = nullptr`。调用前初始化 YRef，零长度 K 组保留对应输出切片的原值。

## 执行流程

1. `Init` 保存调用方传入的 A/B/C、bias、GMMArray 和 scale 地址，并调用 Block 初始化。L1 缓冲数量固定为 `DOUBLE_BUFFER_COUNT`。
2. `Run`：构造 groupList Tensor 和 Scheduler，遍历列表，解析专家号及 M/N/K。
3. 每组先更新地址偏移，再判断空组和调度任务；SPARSE 的 X/Y 按遍历顺序累计，weight/bias/scale 按专家号定位。
4. `ProcessSingleGroup` 解析 scalar scale，取得 A/B/C 的数据首地址并叠加组偏移，然后构造 Tensor、遍历 block 并调用 BlockMmad。最后一组按调度条件进行尾块切分。

## 调用示例

```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using AType = int8_t;
using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;   // <weight, scale>
using BiasType = int32_t;
using YType = half;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;

using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    0, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
using GmmBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA,
                                                   BTypeTuple, LayoutB, YType, LayoutC,
                                                   BiasType, LayoutC>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, GmmBlockMmad,
                                                     BlockEpilogue, BlockScheduler>;

typename GmmKernel::GmmParams gmmParams{...};
typename GmmKernel::Params params = {
    {gmmParams.m, gmmParams.n, gmmParams.k, 1},
    {aTensorListAddr, bTensorListAddr, yTensorListAddr, biasTensorListAddr, perTokenScaleAddr, scaleDataAddr},
    {}, groupListAddr, gmmArrayAddr, gmmParams};
GmmKernel gmm;
gmm(params);
```

## 性能建议
- `kAL1/kBL1` 由 host tiling 的 `stepKa/stepKb * baseK` 给出，二者允许不相等（Block 内按 `kAL1_ == / > / < kBL1_` 三分支调度）；
- `nBufferNum` 固定传 2（双缓冲），与 host 侧 `depthA1 = 2 * stepKa` 的 L1 预算保持一致，L1 不溢出；
- `singleW` 保持与 host 一致，仅保留原有 N/K 尺寸选择语义。
- SPLIT_M 且 `groupListType=2` 时，`loopIdx` 表示列表遍历位置，`groupIdx` 表示真实专家号。X/Y 按遍历的数据量累计偏移；weight 按专家号乘每组物理跨度定位（NZ 包含对齐），bias/per-channel scale 按 `groupIdx * N` 定位。专家号为 0 时也重新赋值。
- 上层接口对多 tensor 的检查与旧内核地址实现存在不一致，本次迁移不修改该接口、不新增数据格式或选路。
