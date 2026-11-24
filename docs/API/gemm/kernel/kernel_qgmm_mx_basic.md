# Kernel Qgmm Mx
> 基础 QGMM [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx.h)；
> ActivationQuant [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h)

## 功能说明
MX 量化 Grouped Matmul 的 Kernel 组件，基于 Tensor API 实现。普通 QGMM 与 GMMAQ
ActivationQuant 分别由 `kernel_qgmm_mx.h` 和 `kernel_qgmm_mx_activation_quant.h`
独立实现，不通过共享 Kernel 基类耦合。普通 QGMM 的默认覆盖模式仅使用 AIC；
`KernelGroupedMmadWithScaleMxActivationQuant` 启用 AIC:AIV=1:2 的融合后处理路径。
普通 QGMM 通过 `BlockEpilogue` 类型选择覆盖输出或 FP32 原地累加，
两种模式共用 group list 解析、Group 偏移、Tensor 组装和 BlockMmad 调用。
普通 QGMM Kernel 自己完成 group list 解析、Group 偏移和 Tensor 组装；GMMAQ 路径额外使用
Scheduler 的 MX group/block 偏移接口完成 AIC/AIV 协同。

**实现框架参考**：[Kernel 公共框架](./kernel.md)

## 特殊约束

### 计算模式
普通 QGMM 的覆盖模式和 `AtomicAdd_=true` 模式仅使用 AIC，`BlockEpilogueInplaceAdd` 模式使用 AIC:AIV=1:1。
ActivationQuant 标签由 AIC 将 float L0C 写入 UB，两个 AIV 执行 GeluTanh 与 MXFP8/MXFP4
在线量化，并通过跨核 flag 串行复用 UB。

### Block 依赖
通常与 `BlockMmad<GroupedMatmulWithScaleMx<...>, ...>` 搭配使用。

### 调度器限制
使用 `BlockSchedulerGmmSwatWithTailSplit` 负责 grouped matmul 的单核 block 分发和 tail split。
ActivationQuant 标签保持原融合核语义，不对最后一组启用 tail split，避免 64 元素 MX scale
组被更小 N 子块重叠写入。

### Scale 类型
普通 MXFP4/MXFP8 路径的 ScaleA/ScaleB 按 `fp8_e8m0_t` 解释；
原地累加的 HiFloat4 B 使用 `MxScaleType<BType>`，即 `AscendC::hif4_scale`。

### Dtype / Format 静态校验
Kernel 编译期通过 `static_assert` 校验模板组合：

- 默认覆盖模式下，`AType` / `BType` 支持同位宽的 MXFP8 组合（`fp8_e4m3fn_t`、`fp8_e5m2_t`）、MXFP4 组合（`fp4x2_e2m1_t`、`fp4x2_e1m2_t`），以及 `AType=fp8_e4m3fn_t`、`BType=fp4x2_e2m1_t` 的 MXA8W4 组合。
- `CType` 支持 `half`、`bfloat16_t`、`float`；同位宽 MXFP8/MXFP4 组合及原地累加模式的 `BiasType` 为 `float`，覆盖模式 MXA8W4 组合的 `BiasType` 支持 `half`、`bfloat16_t`、`float`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`。
- `LayoutB` 仅支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。

原生 MXA8W4 覆盖路径仅支持 9201/9202 架构，并要求以下布局：

- `LayoutA` 为 `nd_ext_layout_ptn`，对应非转置 ND A。
- `LayoutB` 为 `zn_layout_ptn`，对应转置 FRACTAL_NZ B。
- `LayoutScaleB` 为 `scaleb_dn_layout_ptn`，对应 DN E8M0 ScaleB。

NZ 权重的组间偏移按 B 的位宽与 C0 计算，FP4 按每字节两个元素换算地址偏移。

### 原地累加模式

`BlockEpilogueEmpty` 搭配默认的 `AtomicAdd_=false` 选择覆盖输出，搭配 `AtomicAdd_=true` 选择 Fixpipe 原子累加。
`Blaze::Epilogue::Block::BlockEpilogueInplaceAdd` 用于 FP8 × FP4/HiFloat4，并按组 K 长度选择累加路径。
`GroupedMatmulWithScaleMx` 保留原有三个模板参数，不增加输出模式参数。
BlockMmad 根据输出 Tensor 的 GM/UB 位置和单输出/双 UB 输出接口选择搬运路径。

#### FP8 × FP4/HiFloat4

原地累加执行 `C[g] += MatmulMx(A[g], B[g], ScaleA[g], ScaleB[g])`，约束如下：

- A 为 `fp8_e4m3fn_t`，B 为 `fp4x2_e2m1_t` 或 `hifloat4x2_t`，C 为 `float`。
- A/B/C 布局分别为 `dn_ext_layout_ptn` / `nd_ext_layout_ptn` / `nd_ext_layout_ptn`。
- `BlockEpilogue` 使用 `BlockEpilogueInplaceAdd`，`isBias=0`，`singleW=1`。
- 沿 K 轴分组，各组共享 M/N；`groupListType` 只使用 OFFSET（0）或 LENGTH（1）。
- N 和 `baseN` 为 64 的整数倍；M、总 K 和各组 K 长度可以带尾部。
- C 按 `[groupNum, M, N]` 连续存储，调用前包含待累加的初值，零长度组不修改输出。
- 地址有效性、group list 和 tiling 合法性由调用方保证；Kernel 不集中检查参数。

输入连续存储的形状为：

| 数据 | 类型 | GM 形状 |
|------|------|------|
| A | FP8 E4M3 | `[K, M]`，逻辑形状为 `[M, K]` |
| B | FP4 E2M1 / HiFloat4 | `[K, N]`，每字节打包两个 4-bit 元素 |
| ScaleA | `fp8_e8m0_t` | `[K / 64 + groupNum, M, 2]` |
| ScaleB（FP4 E2M1） | `fp8_e8m0_t` | `[K / 64 + groupNum, N, 2]` |
| ScaleB（HiFloat4） | `AscendC::hif4_scale` | `[K / 64 + groupNum, N, 1]`，每项为 32 位 |

表中 `K / 64` 为整数除法，第 `g` 组的 scale 起始行为 `floor(groupStartK / 64) + g`。
HiFloat4 的 GM、L1 和 L0ScaleB Tensor 元素类型均为 `hif4_scale`，GM 布局的 C0 为 1，
GM K 长度和切片偏移按 `MxScaleKLen<BType>` 的原生 scale 元素口径计算。
HiFloat4 的 L1/L0ScaleB 使用带 `c0_element_half` trait 的 NN 布局，
布局 K 长度按 `MxScaleKLen<BType>` 的原生 scale 元素口径计算，即 `ceil(kSpan / 64)`；
其中 `kSpan` 为对应 L1 scale 窗口或 L0 K 分块的长度。
Tensor API 根据 half-element 布局语义，将底层搬运参数换算为 E8M0 单位，
调用方不应再将布局 K 长度乘 4。普通 MXFP4/MXFP8 的 L1/L0ScaleB 仍使用 C0 为 2 的 NN 布局。
L1 缓冲字节数仍按原生 scale 数量乘 `sizeof(hif4_scale)` 计算。
Scale 搬运统一使用 `asc::te::copy`，不做 FP16 数值转换。

选择 `BlockEpilogueInplaceAdd` 时，`AtomicAdd_` 设为 `false`。每组按实际 K 长度选择路径：
`2048 <= groupK <= 16384` 使用 `L0C -> UB -> Vector Add -> GM`，其余非空组使用
Fixpipe AtomicAdd 直接累加到 GM；空组保持 C 不变。UB READY/ACK 面板序号仅随 UB 路径推进。
该 Epilogue 仅支持 9201/9202 架构的 1:1 MIX，不能以 1:2 MIX 方式启动。

接口与同步细节见 [BlockEpilogueInplaceAdd](../../epilogue/block/block_epilogue_inplace_add.md)。
入口使用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`，在 9201 上由一个 AIC 配对一个 AIV。
AIC/AIV 执行相同的 group 和 tile 调度；完整 tile 仅执行一次 MMAD，再沿 M 轴分面板搬出，
由 AIV 读取旧 C、完成 FP32 累加并写回。结果面板与旧 C 使用独立的 UB 缓冲；
Epilogue 析构时由 AIC 等待结果槽位的最终 ACK，最后的 GM 写回由正常 kernel 结束机制保证。

`epilogueParams.ubPanelM` 指定结果面板行数；默认 0 时根据 `AscendC::TOTAL_UB_SIZE` 和
`epilogueParams.baseM/baseN` 自动计算，这两个尺寸应与 `gmmParams.baseM/baseN` 一致。
显式值不得超过 `baseM`，且须为旧 C 暂存缓冲预留空间。Kernel 将两个结果 UB Tensor
通过 `GetAccumulatorTensor({rows, cols, slot})` 获取并显式传入 BlockMmad 和 epilogue。
对齐和 UB 偏移封装在 epilogue 内；固定使用双缓冲，面板行数取自 Tensor shape，
BlockMmad 只额外接收起始面板序号。READY/ACK 同步跨 tile 连续。
UB 暂存块大小根据剩余容量计算，
调用方需保证两个结果面板和两个旧 C 暂存块均能装入 UB，每块至少一行；
行跨度为 `baseN` 向上对齐到 64 个 FP32 元素。
M/N 尾部对齐为 `1/64`，不启用最后一组的额外 tail split。
共享 scale 窗口向下对齐为 `max(kAL1, kBL1)` 的整数倍，至少取一个外层窗口，并以总 K 为上限；
调用方需按实际窗口预算 L1 空间。普通覆盖模式保留现有 scale 窗口和调度行为。

#### MX A8W8 和 MX A4W4 原子累加

- A/B 支持 MX A8W8 或 MX A4W4。A 为 DN，B 为 ND，Epilogue 使用 `BlockEpilogueEmpty`。
- 按 K 分组，组表支持 `groupListType = 0`（累计 K）和 `groupListType = 1`（每组 K 长度）。ScaleA/ScaleB 沿用按 K 分组的存储约定，scale padding 仍按原始组号计算，包含空组。
- 输出为连续 FP32 ND `[G, M, N]`，其中 `G` 为组数，每组更新独立的 `[M, N]` 切片。调用者必须在启动 Kernel 前初始化输出，K 长度为零的组保留原值。

### SwiGLU MX 融合路径约束
`KernelGmmSwiGluMixMx` 路径复用 QGMM MX 的 Cube 计算并在 AIV 侧完成 SwiGLU 与 MX 输出量化。当前该路径仅支持 MXFP8 输入：

- `AType` / `BType` 仅支持 MXFP8（`fp8_e4m3fn_t`、`fp8_e5m2_t`）。
- `CType` / `BiasType` 仅支持 `float`。
- `LayoutA` 仅支持 `nd_ext_layout_ptn`。
- `LayoutB` 支持 `nd_ext_layout_ptn`、`dn_ext_layout_ptn`、`nz_layout_ptn`、`zn_layout_ptn`。
- `LayoutC` / `LayoutBias` 支持 ND 类布局，不支持 `nz_layout_ptn`、`zn_layout_ptn`。

完整的双 N 半区、NZ/ZN 双源搬运、shape、mode 和空指针契约见
[Kernel QGMM SwiGLU MX](./kernel_qgmm_swiglu_mx.md)。

## 特殊类型别名

| 别名 | 含义 |
|------|------|
| `ProblemShape` | 整体问题规模 |
| `SchedulerProblemShape` | scheduler 使用的当前 group 问题规模 |
| `BlockShape` | 单核 block 形状 |
| `SchedulerBlockShape` | scheduler 返回的单核基本块大小与 tail split 偏移 |
| `BlockCoord` | block 坐标 |

## 特殊数据结构

### GMMTiling
```cpp
struct GMMTiling {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `groupNum` | group 数量 |
| `m/n/k` | 初始问题规模 |
| `baseM/baseN/baseK` | 基础 block/L0 tile 大小 |
| `kAL1/kBL1` | A/B 的 L1 K 轴切分 |
| `scaleKAL1/scaleKBL1` | ScaleA/ScaleB 的 L1 K 轴切分 tiling 字段；在 MX 量化中两者必须一致，并作为 BlockMmad 的共享 `scaleKL1` |
| `isBias` | 是否启用 bias；原地累加模式必须为 0 |
| `dbL0C` | L0C 双缓冲模式；当前仅值 `2` 启用，其余值均视为禁用 |
| `l1BufferStage` | A/B 的 L1 缓冲级数；`3` 启用三缓冲，其他值按双缓冲处理 |
| `groupType` | GMM tiling 兼容保留字段；当前 kernel 未读取该字段，split 方向由 `LayoutA` 对应的 `TRANS_A` 编译期路径决定（`!TRANS_A` 按 M，`TRANS_A` 按 K） |
| `groupListType` | offset、length 或 sparse |
| `singleW` | `1` 表示权重/权重 scale 是单个连续 Tensor；`0` 表示按 TensorList 逐 group 解引用 |

`kAL1/kBL1/scaleKAL1/scaleKBL1` 需满足 BlockMmad 的 L1 参数约束：`kAL1` 与 `kBL1` 的较大值作为外层 K 窗口，较大值需为较小值的整数倍；tiling 需传入一致的 `scaleKAL1` 和 `scaleKBL1`。普通覆盖模式要求该值不小于外层窗口且为其整数倍，并直接用作 `scaleKL1`；原地累加模式按前述规则对齐共享窗口。

A/B、ScaleA/ScaleB 和 bias 的 L1 缓冲区需满足容量要求，且各区域互不重叠。
当单个 tile 占用完整 L0C 容量时，`dbL0C` 应设置为 `1`，使用单缓冲。

### Params
```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockEpilogueParams epilogueParams;
    GM_ADDR groupListGmAddr;
    GMMTiling gmmParams;
};
```

## 特殊成员方法

### operator() 函数
```cpp
__aicore__ inline void operator()(const Params& params)
```

功能：
- 作为 kernel 入口，内部调用 `Run(params)`

### Init 函数
```cpp
__aicore__ inline void Init(const Params& params)
```

功能：
- 初始化 GM 基地址
- 读取 grouped matmul tiling 参数
- 构造首个 group 的 `problemShape_`

### Run 函数
```cpp
__aicore__ inline void Run(const Params& params)
```

功能：
- 遍历所有 group
- 逐 group 更新 `m/n/k`
- 逐 group 更新地址偏移
- 调用 scheduler 和 BlockMmad 完成单核 block 级计算

### SetMNK 函数
```cpp
__aicore__ inline void SetMNK(uint32_t groupIdx)
```

功能：
- 从 group list 中提取当前组切分值
- 当前 QGMM MX scalar 路径按 `LayoutA` 对应的 `TRANS_A` 编译期路径更新当前组的 `problemShape_`；`groupType` 在当前 kernel 中未参与该判断

### Scheduler 的 UpdateMxGroup/GetNextMxBlock 接口
```cpp
__aicore__ inline void UpdateMxGroup(const MxGroupParams& params)
__aicore__ inline bool GetNextMxBlock(MxBlockInfo& blockInfo)
```

功能：
- 仅供 GMMAQ ActivationQuant Kernel 使用
- 调度器根据当前 group 形状计算 A/B/ScaleA/ScaleB/Bias/输出基础偏移
- 调度器返回 block 形状、输入 slice 坐标及输出/输出 scale 偏移

### ProcessSingleGroup 函数
```cpp
template <bool isLastGroupAndNeedSplit>
__aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
```

功能：
- 普通 QGMM：Kernel 根据 Scheduler 的通用 block 坐标和自身计算的 Group 偏移构造 GM Tensor，直接交给 `BlockMmad`
- GMMAQ：Kernel 使用 Scheduler 的 MX group/block 信息，AIC 将结果写 UB，AIV 执行 Epilogue

## 调用示例

### 组件组装
```cpp
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
// 默认覆盖模式；原有调用无需添加输出模式参数。
using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using QgmmKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

原地累加使用以下组件组合：

```cpp
#include "blaze/epilogue/block/block_epilogue_inplace_add.h"

using InplacePolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
using InplaceBlockMmad = Blaze::Gemm::Block::BlockMmad<
    InplacePolicy, fp8_e4m3fn_t, asc::te::dn_ext_layout_ptn,
    fp4x2_e2m1_t, asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn,
    float, asc::te::nd_ext_layout_ptn>;
using InplaceKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, InplaceBlockMmad, Blaze::Epilogue::Block::BlockEpilogueInplaceAdd, BlockScheduler>;

InplaceKernel::Params inplaceParams = {
    {m, n, k, 0},
    {aGmAddr, bGmAddr, cGmAddr, nullptr, scaleAGmAddr, scaleBGmAddr},
    {},
    groupListGmAddr,
    {groupNum, m, n, k, baseM, baseN, baseK, kAL1, kBL1,
     scaleKAL1, scaleKBL1, 0, dbL0C, l1BufferStage, 2, groupListType, 1}
};
inplaceParams.epilogueParams = {baseM, baseN, 64}; // 面板行数设为 0 时自动规划。
// 在 __global__ __aicore__ 入口中声明混合任务类型。
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
InplaceKernel inplaceKernel;
inplaceKernel(inplaceParams);
```

HiFloat4 输入将 B 类型替换为 `hifloat4x2_t`，同时提供对应的 `hif4_scale` 数据。
`groupType=2` 表达 K 分组，实际 split 方向仍由 DN A 的布局路径选择。

### 参数准备
```cpp
using Params = typename QgmmKernel::Params;

Params params = {
    {m, n, k, 0},
    {aGmAddr, bGmAddr, cGmAddr, biasGmAddr, scaleAGmAddr, scaleBGmAddr},
    {},
    groupListGmAddr,
    {groupNum, m, n, k, baseM, baseN, baseK, kAL1, kBL1, scaleKAL1, scaleKBL1,
     isBias, dbL0C, l1BufferStage, groupType, groupListType, singleW}
};
```

### Kernel 执行
```cpp
QgmmKernel kernel;
kernel(params);
```

### MX A8W8 原子累加示例

以下示例使用 MXFP8 E4M3 输入、连续存放的 B 和 ScaleB，以及按组长度编码的 INT64 组表。调用前将初始 YRef 写入 `yRefGmAddr` 指向的 FP32 GM 缓冲区。

```cpp
using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using LayoutA = asc::te::dn_ext_layout_ptn;
using LayoutND = asc::te::nd_ext_layout_ptn;
using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, true>;
using Mmad = Blaze::Gemm::Block::BlockMmad<
    Policy, fp8_e4m3fn_t, LayoutA, fp8_e4m3fn_t, LayoutND, float, LayoutND, float, LayoutND>;
using Epilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;

constexpr uint8_t NO_BIAS = 0;
constexpr uint8_t DOUBLE_BUFFER = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
constexpr int8_t GROUP_TYPE_K = 2;
constexpr uint8_t GROUP_LIST_LENGTH = 1;
constexpr uint8_t SINGLE_WEIGHT = 1;
Kernel::Params params{};
params.problemShape = {m, n, k, 0};
params.mmadParams.aGmAddr = aGmAddr;
params.mmadParams.bGmAddr = bGmAddr;
params.mmadParams.cGmAddr = yRefGmAddr;
params.mmadParams.scaleAGmAddr = scaleAGmAddr;
params.mmadParams.scaleBGmAddr = scaleBGmAddr;
params.groupListGmAddr = groupListGmAddr;
params.gmmParams = {groupNum, m, n, k, baseM, baseN, baseK, kAL1, kBL1,
                    scaleKL1, scaleKL1, NO_BIAS, DOUBLE_BUFFER, DOUBLE_BUFFER,
                    GROUP_TYPE_K, GROUP_LIST_LENGTH, SINGLE_WEIGHT};

Kernel kernel;
kernel(params);
```

## 调度流程
```text
读取首组 tiling
    -> 初始化 block scheduler
    -> 遍历 group
    -> 更新当前组的 m/n/k
    -> 更新 A/B/Scale/Bias/C 偏移（普通 QGMM 在 Kernel 内完成；GMMAQ 由 MX Scheduler 辅助完成）
    -> scheduler 分发单核 block
    -> BlockMmad 在单核 block 内执行 L0 tile 计算
    -> 普通覆盖模式的末组按调度条件执行 tail split，原地累加模式沿用基础 tile 分配
    -> 原地累加模式按所选路径使用 Fixpipe 原子写回或 AIV UB 加法及 GM 写回
```

使用 `AtomicAdd_=true` 时，AIC 在遍历组表前调用 `SetAtomicAdd<float>()`，全部组处理结束后调用 `SetAtomicNone()`。

## 适用场景
- MX 量化 grouped matmul
- group 间 `m` 或 `k` 动态变化
- 需要 sparse group list / tail split 的 grouped matmul 场景
- 需要将各 K 分组的结果累加到 FP32 YRef 的场景

## 双级 MXFP4 策略

`GroupedMatmulWithScaleMx<0,false,KernelGroupedMmadWithDualLevelScaleMx>` 使用同一个
`GemmUniversal`，与 `BlockEpilogueDualLevelScale` 组合执行。该策略要求 X 为 E2M1 ND，
weight 为 E2M1 NZ/ZN，沿 M 分组，level1 scale 为 ND，输出为 FP16/BF16。
Host 在生成 tiling 前校验逻辑 K 和 level0 段长。调用方须保证 K 大于 2 且为偶数，并将 `baseK`、`kAL1` 和 `kBL1` 都设为 level0 段长，将 `scaleKAL1` 和 `scaleKBL1` 设为同一个 level1 scale L1 K 窗口。Kernel 在初始化跨核同步前校验 L1 条带等于 level0 段长、段长满足 MX 对齐及 L0 缓冲区容量约束，且 scale 窗口长度是段长的正整数倍。
Kernel 由 `gmmParams.k` 和 `gmmParams.baseK` 计算包含尾段的段数，用于 level0 scale 偏移和 AIV 侧的 Epilogue 初始化。在 AIC 侧向 BlockMmad 传入逻辑 K 长度和完整 K 段长度。二级量化调用方在 `epilogueParams` 中提供两路 level0 scale 地址，在 `mmadParams` 中提供 bias 和输出地址。Kernel 使用这四个地址、基本块配置和段数构造 Epilogue 的初始化参数。
Kernel 将 BlockMmad 的 L0 基本块行列上限和行对齐作为 Epilogue `Init` 的模板参数，使 AIC 的 UB 写入视图和 AIV 的读取布局使用同一份定义。
Kernel 将 `DOUBLE_BUFFER_COUNT` 传给 Epilogue 初始化，并用同一数量校验 UB 视图数组。私有嵌套类 `DualLevelPartialChannel` 按该数量管理连续序号、初始 credit 与最终 drain。
在 AIC 从 Epilogue 的 `GetAccumulatorTensor` 获取UB 接收 Tensor 视图数组并传给 MMAD 的 `operator()`，
在 AIV 调用 Epilogue 的 `operator()`。BlockMmad 和 Epilogue 通过模板参数使用通道，不包含 Kernel 头文件。
单块内的微块遍历器 `DualLevelPartialTiles` 同样为 Kernel 私有嵌套类。Kernel 在
BlockMmad 和 Epilogue 的 `operator()<PartialTiles_>` 中显式指定相同类型，统一生产与消费顺序，
不将单块内部的微块遍历纳入 BlockScheduler 的多核任务分配职责。
两侧共享 partial 微块遍历规则；AIV 不经过 MMAD 的 L1/L0 循环。
Kernel 将专家与全局输出坐标转换为当前块的两侧 level0 scale、bias 和输出偏移。
Epilogue 管理 partial 布局，通道管理 ready/credit。

level1 E8M0 scale 每 32 K 生效。每个 level0 K 段产生的 FP32 partial 在 epilogue 中应用两侧
level0 FP32 scale 后跨段累加，最后加可选 bias 并转换输出。
`groupListType` 支持 0（结束位置）、1（行数）和 2（专家编号/行数）；type=2 的
输入形状为 `(E,2)`。调用方须保证专家编号有效、各组行数非负且总和等于 M。
非零行组在前，零行组在后，两类分组内部均按专家编号递增排列。
普通 MX 策略保持 AIC 执行和原有调用接口。
完整组件约束见 [双级 epilogue](../../epilogue/block/block_epilogue_dual_level_scale.md)，
组装示例见 [dual_level_grouped_matmul](../../../../examples/acl/grouped_matmul/dual_level_grouped_matmul/README.md)。
