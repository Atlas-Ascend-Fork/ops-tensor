# Kernel Qbmm Mx
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mx.h)

## 功能说明
MX 量化 Batch Matmul Kernel，仅支持 AIC 计算，支持 MxFP4/MxFP8 量化格式。集成 Scale 因子（x1Scale、x2Scale）处理、多 Batch 维度支持、L2 Cache 动态配置，适用于量化推理场景。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
支持以下量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`（4-bit 浮点）
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`（8-bit 浮点）

Kernel 通过 `static_assert` 要求 A/B 为同 bit-width 的 MxFP4 或 MxFP8 组合，C 为 `half`、
`bfloat16_t` 或 `float`，并保留原有 `BiasType = float` 约束。`LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN，
`LayoutC` 支持 `nd_ext_layout_ptn` / `dn_ext_layout_ptn`；本次不增加 `LayoutBias` 或 scale 校验。

ND/DN 白名单保留 L0C 搬运路径支持的 C Tensor 布局；自定义 BlockEpilogue 必须与输出布局一致。
模板校验通过不等同于任意后处理组件均支持 DN 输出。

### Scale 因子要求

ScaleB 布局来自 `BlockMmad::LayoutScaleB`。原有非 tuple 的 B 布局调用仍使用 ND scale；
ScaleNZ 调用使用独立 `nn_layout_ptn`。具体 shape 和转换约定见
[MX BlockMmad 的 ScaleB 独立布局](../block/block_mmad_qbmm_mx.md#scaleb-独立布局)。

ScaleNZ 的 batch stride 包含 N 轴补齐，N 对齐使用 `SCALE_N_ALIGN`；ND scale 的 N 对齐为 1。

ScaleNZ 的动态 N 尾块起点按 `SCALE_N_ALIGN` 对齐，避免从 NZ 的 N0 分组中间搬运。

必须提供两个 Scale 因子：
- `scaleAGmAddr`：A 矩阵的 缩放因子（`fp8_e8m0_t` 类型）
- `scaleBGmAddr`：B 矩阵的 缩放因子（`fp8_e8m0_t` 类型）

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。

### BlockScheduler 限制
仅支持 `BlockSchedulerQbmm` 调度器，支持多 Batch 维度切分。

### L2 Cache 动态配置
根据 tile 形状动态启用/禁用 L2 Cache：
- 大 tile（`curBaseM >= problemShape.m`）：禁用 L2 Cache
- 小 tile：启用 L2 Cache

### Batch 维度限制
支持 4 维 Batch（batchA1/A2/A3/A4、batchB1/B2/B3/B4、batchC1/C2/C3/C4），需满足广播规则。x1Scale/x2Scale 的 Batch 维度须分别与 A 矩阵/B 矩阵一致。

### Atomic Add 模式
可选 Atomic Add 模式（`IS_ATOMIC_ADD = true`），用于多核并行累加场景。

## 特殊成员方法

### 构造函数
```
__aicore__ inline GemmUniversal()
```
功能：构造 GemmUniversal 对象。

### 析构函数
```
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal 对象。

### 特殊模板参数

```
template <
    class ProblemShape,      // 问题形状类型
    class BlockMmad,         // BlockMmadMX 组件
    class BlockEpilogue,     // 后处理组件（通常为 void）
    class BlockScheduler>    // BlockSchedulerQbmm 调度器
```

### 模板参数说明
| 参数 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、b（batch） |
| BlockMmad | BlockMmadMX 组件，基于 `MatmulWithScaleMx` 调度策略 |
| BlockEpilogue | 后处理组件（通常不使用，传 void） |
| BlockScheduler | BlockSchedulerQbmm 调度器 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| WEIGHT_NZ | B 矩阵是否为 NZ 格式（由 LayoutB 判断） |
| TRANS_A | A 矩阵是否转置（由 LayoutA 判断） |
| TRANS_B | B 矩阵是否转置（由 LayoutB 判断） |
| IS_ATOMIC_ADD | 是否启用 Atomic Add 模式（继承自 BlockMmad::DispatchPolicy） |
| C0_SIZE | C0 对齐大小（FP4: 64，FP8: 32） |
| SCALE_C0 | Scale C0 对齐大小（固定为 2，定义于 `common_utils.h`） |
| MakeLayoutScaleA | ScaleA Layout 构建器（根据 TRANS_A 选择） |
| MakeLayoutScaleB | ScaleB Layout 构建器（根据 TRANS_B 选择） |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // mmad 参数（包含 GM 地址）
    L1Params l1Params;              // L1 参数（kL1, scaleKL1, l1BufNum）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // QBMM 特有参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4;  // A 矩阵/x1Scale 的 Batch 维度
    uint32_t batchB1, batchB2, batchB3, batchB4;  // B 矩阵/x2Scale 的 Batch 维度
    uint32_t batchC1, batchC2, batchC3, batchC4;  // C 矩阵 Batch 维度
    uint32_t biasThreeDim;                        // Bias 是否为 3 维
    uint32_t baseM, baseN, baseK;                 // L0 tile 形状
    uint32_t isBias;                              // 是否启用 bias
    uint32_t dbL0C;                               // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U;                     // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

### BlockMmadParams（MX 特有）
```
struct Params {
    GM_ADDR aGmAddr;             // A 矩阵 GM 地址
    GM_ADDR bGmAddr;             // B 矩阵 GM 地址
    GM_ADDR cGmAddr;             // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;          // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr;        // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr;        // B 矩阵 Scale GM 地址
};
```

## 特殊成员方法

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能：初始化 Kernel，提取问题规模、GM 地址、Batch 参数。
执行流程：
1. AIV 检查：如果是 AIV 核则直接返回
2. 设置 Bias 标志：根据 `qbmmParams.isBias` 判断
3. 设置 BiasThreeDim 标志：根据 `qbmmParams.biasThreeDim` 判断
4. 调用 `ResetGmAddr` 设置 GM 地址

### Run函数
```
__aicore__ inline void Run(const Params& params)
```
功能：执行量化 Batch Matmul Kernel 计算。
执行流程：
1. Atomic Add 配置：如果 `IS_ATOMIC_ADD = true`，调用 `SetAtomicAdd<float>`
2. 调用 `Init(params)` 设置参数
3. 创建 BlockScheduler 实例
4. 初始化 BlockMmadMX 组件
5. 判断 Batch 数量：
   - 单 Batch（`b == 1`）：调用 `ProcessSingleBatch`
   - 多 Batch：调用 `ProcessWithBatch`
6. 清理 Atomic Add：如果启用，调用 `SetAtomicNone`

### ProcessSingleBatch函数
```
__aicore__ inline void ProcessSingleBatch(
    const Params& params, BlockScheduler& bs,
    uint64_t batchCnt, bool isTailRound)
```
功能：处理单个 Batch 的矩阵乘计算。
执行流程：
1. 构建 Layout：A、B、ScaleA、ScaleB、Bias、C
2. 创建 GM Tensor
3. 动态配置 L2 Cache
4. Tile 循环处理：
   - 获取 tile 坐标和形状
   - Slice GM Tensor 到当前 tile
   - 调用 BlockMmadMX 执行量化矩阵乘

### ProcessWithBatch函数
```
__aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs)
```
功能：处理多 Batch 的矩阵乘计算。
执行流程：
1. 计算 Batch 广播倍数：multiA1C1、multiB1C1 等
2. 4 维 Batch 循环（batchC1/C2/C3/C4）：
   - 更新 Batch 偏移：`batchCOffset_`、`batchAOffset_`、`batchBOffset_`，x1Scale/x2Scale 分别复用 `batchAOffset_`/`batchBOffset_`
   - 调用 `AddBatchOffset` 更新 GM 地址
   - 调用 `ProcessSingleBatch` 处理当前 Batch

### SetL2Cache函数
```
template <typename TensorB, typename TensorC>
__aicore__ inline void SetL2Cache(
    const ProblemShape& problemShape,
    uint64_t baseM, uint64_t baseN,
    TensorB& gmB, TensorC& gmC)
```
功能：动态配置 L2 Cache。
说明：
- 同 Batch 且 M tile 覆盖完整 M 维时，根据 B 的布局和对齐情况配置 B 的 L2 Cache
- 当前实现不再配置 ScaleB 的 L2 Cache hint，ScaleB 使用 Tensor API 默认 Cache 策略
- Atomic Add 模式：禁用 C 的 L2 Cache

## 调用示例

### 组件组装
```
// 定义量化数据类型
using AType = fp4x2_e2m1_t;  // 或 fp8_e5m2_t
using BType = fp4x2_e2m1_t;  // 或 fp8_e5m2_t
using CType = float;
using BiasType = float;
using ScaleType = fp8_e8m0_t;

// 定义 Layout
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;

// 定义调度策略
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<A_FULL_LOAD_MODE>;

// 定义 BlockMmadMX
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

// 定义 BlockScheduler
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQbmm<ProblemShape>;

// 定义 Kernel
using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, void, BlockScheduler>;
```

### 参数准备
```
using Params = typename QBMMKernel::Params;
Params params = {
    {m, n, k, batch},               // problem shape
    {aGM, bGM, cGM, biasGM, scaleAGM, scaleBGM}, // mmad params
    {kL1, scaleKL1, l1BufNum},       // L1 params
    {baseM, baseN, baseK, mTailTile, nTailTile, ...}, // scheduler params
    {batchA1, batchA2, ... batchC4, biasThreeDim, baseM, baseN, baseK, isBias, dbL0C,
     bMustHitL2} // QBMM tiling
};
```

### Kernel 执行
```
QBMMKernel qbmm;
qbmm(params);  // 或 qbmm.Run(params);
```

## 数据流

### 存储层次
```
GM (A/B/ScaleA/ScaleB/Bias) → L1 (量化数据 + Scale) → L0A/L0B → L0C → GM (C)
```

### Batch 处理流程
```
Batch 循环（4 维）
    ↓
更新 Batch 偏移（batchA/B/COffset）
    ↓
重置 GM 地址（AddBatchOffset）
    ↓
处理单个 Batch（ProcessSingleBatch）
    ↓
Tile 循环 → BlockMmadMX 执行
```

### 量化计算流程
```
加载量化数据 (A/B) + Scale (ScaleA/ScaleB)
    ↓
Dequantize（自动在 Mmad 中完成）
    ↓
Mmad 计算（MX 模式：MmadTraitMX）
    ↓
输出 float 结果
```

## 性能优化建议

### L1 缓冲配置
- `l1BufNum`：建议 2、3 或 4，平衡 L1 容量和流水线并行度
- `kL1`：建议对齐到 `MXFP_DIVISOR_SIZE`（64）

### Scale KL1 配置
- `scaleKL1`：建议为 `kL1` 的整数倍（如 2×kL1）
- Scale 数据复用：Scale 常驻 L1，减少搬运开销

### 全载模式选择
- **非全载模式**：每次迭代重新加载 A/B 块
- **A 全载模式**：A 矩阵常驻 L1，适用于大 K、小 M 场景

### Batch 维度设计
- Batch 维度需满足广播规则：`batchA = batchC × multiA`、`batchB = batchC × multiB`
- 4 维 Batch 灵活支持多种广播场景

### L2 Cache 配置
- 大 tile 场景自动禁用 L2 Cache，避免缓存污染
- Atomic Add 模式自动禁用 C 的 L2 Cache

## 适用场景

- **量化推理**：MxFP4/MxFP8 量化矩阵乘
- **Batch Matmul**：多 Batch 维度支持
- **Scale 因子处理**：per-token 和 per-group scale
- **广播 Batch**：A/B/C 不同 Batch 维度的广播计算
