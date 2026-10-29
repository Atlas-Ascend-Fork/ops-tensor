# Kernel Qbmm StreamK
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_streamk.h)

## 功能说明
QBMM MX StreamK Kernel 是面向 MxFP4/MxFP8 量化矩阵乘的 `GemmUniversal` 特化实现。该 Kernel 复用 MX 量化场景的 `BlockMmad`，复用非量化 Matmul StreamK 的调度器和后处理组件，通过 AIC + AIV 协同完成 K 轴切分后的中间结果归约。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
支持 MX 量化输入，A/B 矩阵可使用 MxFP4 或 MxFP8 数据类型，Scale 张量使用 `fp8_e8m0_t` 类型。

Kernel 通过 `static_assert` 要求 A/B 为同 bit-width 的 MxFP4 或 MxFP8 组合，C 为 `half`、
`bfloat16_t` 或 `float`；BlockEpilogue 输出必须与 C 一致，workspace 必须为 `float`。
`LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN，`LayoutC` 支持 `nd_ext_layout_ptn` / `dn_ext_layout_ptn`。
Kernel 不额外校验 bias 类型、`LayoutBias` 或 scale。

ND/DN 白名单保留 L0C 搬运路径支持的 C Tensor 布局；自定义 BlockEpilogue 必须与输出布局一致。
模板校验通过不等同于任意后处理组件均支持 DN 输出。

### Scale 因子要求

ScaleB 布局来自 `BlockMmad::LayoutScaleB`。原有非 tuple 的 B 布局调用仍使用 ND scale；
ScaleNZ 调用使用独立 `nn_layout_ptn`。具体 shape 和转换约定见
[MX BlockMmad 的 ScaleB 独立布局](../block/block_mmad_qbmm_mx.md#scaleb-独立布局)。

必须提供两个 Scale 因子：
- `scaleAGmAddr`：A 矩阵的 per-token scale，对应算子输入 `perTokenScale`
- `scaleBGmAddr`：B 矩阵的 per-group scale，对应算子输入 `scale`

### 计算模式
支持 AIC + AIV 双核协同：
- **AIC 核**：执行 MX 量化矩阵乘计算，DP tile 直接写 C，SK tile 写 workspace
- **AIV 核**：执行 StreamK 后处理，从 workspace 读取多个 K 切分结果，累加后写回 C

### Batch 限制
当前 StreamK 特化仅支持 `batch == 1`。多 Batch、Batch 广播和 Batch 偏移不在该 Kernel 中处理。

### Workspace 必需
必须提供 workspace 用于存储 StreamK 的 K 轴切分中间结果：
- AIC 在 SK 模式下把 partial sum 写入 workspace
- AIV 从 workspace 读取 partial sum 并执行 Add 归约
- AIV 将最终结果写回输出 GM

### BlockScheduler 限制
仅支持 `BlockSchedulerMatmulStreamK` 调度器，调度策略与非量化 StreamK 保持一致，支持 DP+SK 混合模式。

### BlockEpilogue 限制
复用非量化 `BlockEpilogueMatmulStreamK`。workspace 写入和读取必须使用同一套 stride 约定，尤其在 `ND_FIXPIPE_1_2` 模式下，workspace 的 N 维 stride 需要按 epilogue 的 Fixpipe 策略对齐。

### L0C2Out 模式
支持两种 Fixpipe 输出模式：
- **ON_THE_FLY**：workspace 按紧凑 N stride 写入和读取
- **ND_FIXPIPE_1_2**：workspace 的 N stride 按 32B 对齐，保证 AIC 写入和 AIV 读取约定一致

## 特殊模板组件

| 组件 | 要求 |
|------|------|
| ProblemShape | `Shape<m, n, k, batch>`，其中 `batch == 1` |
| BlockMmad | `BlockMmad<MatmulWithScaleMx<..., KernelQbmmMultiBlockStreamK>, ...>` |
| BlockEpilogue | `BlockEpilogueMatmulStreamK<WorkspaceType, OutType, MatmulMultiBlockWithStreamK<...>>` |
| BlockScheduler | `BlockSchedulerMatmulStreamK<ProblemShape>` |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、batch |
| BlockMmadOp | MX 量化 BlockMmad 组件 |
| BlockEpilogue | StreamK 后处理组件 |
| BlockScheduler | StreamK 调度器 |
| BlockMmadParams | `BlockMmadOp::Params`，包含 A/B/C/Bias/Scale 地址 |
| BlockEpilogueParams | `BlockEpilogue::Params`，包含 C 和 workspace 地址 |
| BlockSchedulerParams | `BlockScheduler::Params`，包含 usedCoreNum、baseM/baseN/baseK、singleCoreK、kL1 等调度参数 |

## 特殊数据结构

### Params

```cpp
struct Params {
    ProblemShape problemShape;          // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;         // MX BlockMmad 参数
    BlockEpilogueParams epilogueParams; // StreamK epilogue 参数
    BlockSchedulerParams schParams;     // StreamK scheduler 参数
    QBMMStreamKParams qbmmParams;       // QBMM StreamK 特有参数
};
```

### QBMMStreamKParams

```cpp
struct QBMMStreamKParams {
    uint32_t scaleKL1; // Scale 在 L1 上的 K 方向长度
    uint32_t dbL0C;    // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U; // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

Scale L1 约束：
- `scaleKL1 >= schParams.kL1`
- `scaleKL1 % schParams.kL1 == 0`
- `scaleKL1` 与 `schParams.kL1` 均以原始 K 轴元素数计量，不使用压缩后的 scale 元素数计量

约束原因：`BlockMmad` 使用 `scaleKL1 / kL1` 计算一个 Scale L1 buffer 可覆盖的 K window 数，并据此控制 scale buffer 复用与释放。如果 `scaleKL1 < kL1`，或 `scaleKL1` 不是 `kL1` 的整数倍，scale 搬运和同步节奏可能与 A/B 的 K window 节奏不一致。

K 轴切分与 Scale group 对齐约束：
- `schParams.singleCoreK` 必须按 MX scale group 对齐，即 `schParams.singleCoreK % 64 == 0`
- 非尾部 K split 不允许从一个 MX scale group 中间开始；每个非尾 split 的真实 K 起点应落在 64 的整数倍上
- 尾部 K split 可以小于 `singleCoreK`，但其起点仍必须满足 64 对齐，scale 读取长度按尾部真实 K 长度 `ceil(curK / 64) * 2` 计算
- 该约束由 host tiling 保证；如果外部直接实例化该 kernel，也必须提供满足上述条件的 `singleCoreK`

约束原因：MxFP4/MxFP8 的 scale 以每 64 个 K 元素为一个 group，并在 scale tensor 上以 `ceil(K / 64) * 2` 存储。StreamK 沿 K 轴切分时，scale offset 必须由真实 K 起点对应的 scale group 推导；若 `singleCoreK` 不是 64 对齐，后续 split 可能切在 scale group 中间，导致 A/B 数据与 scale group 错位。

### BlockMmadParams（QBMM MX StreamK 特有）

```cpp
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址，可选
    GM_ADDR scaleAGmAddr; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr; // B 矩阵 Scale GM 地址
};
```

说明：是否启用 Bias 由 `biasGmAddr != nullptr` 判断，不额外传递 bias 标志。

## 特殊成员方法

### 构造函数
```cpp
__aicore__ inline GemmUniversal()
```
功能：构造 QBMM MX StreamK Kernel 对象。

### 析构函数
```cpp
__aicore__ inline ~GemmUniversal()
```
功能：析构 QBMM MX StreamK Kernel 对象。

### Init函数
```cpp
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Kernel，提取问题规模、GM 地址、workspace 地址和 QBMM 参数。

执行流程：
1. 保存 `problemShape` 和 `usedCoreNum`
2. 设置 A/B/C/Bias/ScaleA/ScaleB 的 GM 地址
3. 设置 workspace GM 地址
4. 根据 `biasGmAddr` 判断是否启用 Bias

### operator函数
```cpp
__aicore__ inline void operator()(Params const& params)
```
功能：执行 QBMM MX StreamK Kernel。

公共流程：
1. 调用 `Init(params)` 初始化参数
2. 检查 `usedCoreNum` 和 batch，batch 不为 1 时直接返回
3. 创建 `BlockSchedulerMatmulStreamK`
4. 获取 L1/L0 tile 形状、MN tile 数量和 K 切分数量

AIC 核执行流程：
1. 根据 blockIdx 判断是否参与实际计算
2. 初始化 MX `BlockMmad`
3. 构建 A/B/C/Bias/ScaleA/ScaleB 的 Tensor API Layout
4. 根据当前 tile 坐标 slice GM Tensor
5. DP 模式下直接写 C
6. SK 模式下写 workspace，并在需要时预取下一轮 SK tile
7. 计算完成后通过跨核同步标志通知 AIV

AIV 核执行流程：
1. 等待 AIC 同步标志
2. 初始化 `BlockEpilogueMatmulStreamK`
3. 从 workspace 读取 K 轴切分的 partial sum
4. 执行 Add 归约和类型转换
5. 将最终结果写回 C GM

## Workspace 布局

### 写入约定
AIC 在 SK tile 上把 partial sum 写入 workspace。workspace 的基地址按 core、tile 和 K 切分编号组织，保证每个 K 切分结果互不覆盖。

### stride 约定
workspace 的 N 维 stride 需要和 epilogue 保持一致：

```cpp
auto workspaceStrideColumn0 =
    BlockEpilogue::DispatchPolicy::fixpOpti == MatMulL0C2Out::ND_FIXPIPE_1_2 ?
        CeilAlign(Get<MNK_N>(singleCoreShape), BLOCK_BYTE_SIZE) :
        Get<MNK_N>(singleCoreShape);
```

说明：
- `ON_THE_FLY` 模式使用紧凑 N stride
- `ND_FIXPIPE_1_2` 模式使用 32B 对齐后的 N stride
- AIC 写 workspace 和 AIV 读 workspace 必须使用同一个 stride，否则非 32 对齐 N 场景会出现读写错位

## DP+SK 混合策略

调度策略复用非量化 StreamK：

```cpp
tailMNTileNum = (mTileNum * nTileNum) % usedCoreNum;
tileNum = (mTileNum * nTileNum - tailMNTileNum) + tailMNTileNum * skKTileNum;
```

说明：
- **DP（Data Parallel）模式**：前 `mTileNum * nTileNum - tailMNTileNum` 个 MN tile 完整计算，结果直接写 C
- **SK（StreamK）模式**：尾部 `tailMNTileNum` 个 MN tile 沿 K 轴拆成多个 tile，partial sum 写 workspace
- **AIV 归约**：对 SK 模式的多个 K 切分结果执行 Add，输出最终 C

## 调用示例

完整可编译、可运行并带 golden 校验的示例见
[quant_batch_matmul_mx](../../../../examples/quant_batch_matmul/quant_batch_matmul_mx/README.md)，
对应 CSV 场景为 `qbmm_mx_streamk`。

### 组件组装

```cpp
using AType = fp8_e4m3fn_t;
using BType = fp8_e5m2_t;
using OutType = half;
using BiasType = float;
using WorkspaceType = float;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false, Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
using EpilogueDispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<
    WorkspaceType, OutType, EpilogueDispatchPolicy>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
using QbmmStreamKKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备

```cpp
QbmmStreamKKernel::Params params{};
params.problemShape = {m, n, k, 1};
params.mmadParams = {x1GM, x2GM, yGM, biasGM, perTokenScaleGM, scaleGM};
params.epilogueParams = {yGM, workspaceGM};
params.schParams.usedCoreNum = usedCoreNum;
params.schParams.baseM = baseM;
params.schParams.baseN = baseN;
params.schParams.baseK = baseK;
params.schParams.singleCoreK = singleCoreK;
params.schParams.kL1 = kL1;
params.schParams.isHf32 = isHf32;
params.schParams.l2CacheMode = l2CacheMode;
params.qbmmParams = {scaleKL1, dbL0C, bMustHitL2};
```

`scaleKL1` 必须满足 `scaleKL1 >= kL1` 且 `scaleKL1 % kL1 == 0`；
`singleCoreK` 必须按 64 个原始 K 元素对齐。`perTokenScaleGM` 和 `scaleGM` 分别指向
A per-token scale 与 B per-group scale。

### Kernel 执行

```cpp
QbmmStreamKKernel qbmm;
qbmm(params);
```

## 数据流

### 存储层次

```text
GM(A/B/ScaleA/ScaleB/Bias)
    ↓
Tensor API Layout + BlockSchedulerMatmulStreamK
    ↓
BlockMmad<MatmulWithScaleMx<..., KernelQbmmMultiBlockStreamK>>
    ↓
DP tile: L0C → GM(C)
SK tile: L0C → Workspace(float partial sum)
    ↓
BlockEpilogueMatmulStreamK(AIV)
    ↓
GM(C)
```

### AIC/AIV 协同流程

```text
AIC: 计算 DP tile → 直接写 C
AIC: 计算 SK tile → 写 workspace
AIC: CrossCoreSetFlag 通知 AIV
    ↓
AIV: CrossCoreWaitFlag 等待 AIC 完成
AIV: 从 workspace 读取多个 K 分片
AIV: Add 归约 + Cast/ReLU 等后处理
AIV: 写回 C
```
