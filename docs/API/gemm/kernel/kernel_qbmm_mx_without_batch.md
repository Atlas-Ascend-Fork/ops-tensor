# Kernel Qbmm Mx Without Batch

> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h)

## 功能说明

该 Kernel 面向 Batch 为 1 的 MX 量化矩阵乘场景，仅在 AIC 上执行，支持 MxFP4/MxFP8
量化格式。Kernel 由 `BlockMmadMX` 和 `BlockSchedulerQbmm` 组装；由于 Batch 固定为 1，调度过程
不执行多 Batch 地址换算和广播循环。

**相关实现**：[Kernel Qbmm Mx](./kernel_qbmm_mx.md)

## 特殊约束

### 量化格式支持

支持以下量化数据类型：

- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`

A/B、C dtype 和 A/B/C layout 的编译期 `static_assert` 约束为：A/B 必须为
同 bit-width 的 MxFP4 或 MxFP8 组合，C 支持 `half` / `bfloat16_t` / `float`；
`LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN，`LayoutC` 支持 `nd_ext_layout_ptn` / `dn_ext_layout_ptn`。
Kernel 不额外校验 bias 类型、`LayoutBias` 或 scale。

ND/DN 白名单保留 L0C 搬运路径支持的 C Tensor 布局；自定义 BlockEpilogue 必须与输出布局一致。
模板校验通过不等同于任意后处理组件均支持 DN 输出。

### Scale 因子要求

ScaleB 布局来自 `BlockMmad::LayoutScaleB`。原有非 tuple 的 B 布局调用仍使用 ND scale；
ScaleNZ 调用使用独立 `nn_layout_ptn`。具体 shape 和转换约定见
[MX BlockMmad 的 ScaleB 独立布局](../block/block_mmad_qbmm_mx.md#scaleb-独立布局)。

ScaleNZ 的动态 N 尾块起点按 `SCALE_N_ALIGN` 对齐，避免从 NZ 的 N0 分组中间搬运。


必须提供两个 Scale 因子：

- `scaleAGmAddr`：A 矩阵的 per-token scale（`fp8_e8m0_t` 类型）
- `scaleBGmAddr`：B 矩阵的 per-group scale（`fp8_e8m0_t` 类型）

### Batch 限制

该类只处理单 Batch 输入，不包含 `batchA1`、`batchB1`、`batchC1` 等 Batch 广播参数。需要多 Batch 或广播场景时使用 `kernel_qbmm_mx.h` 中的 QBMM MX Kernel。

### 计算模式

仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。

## 特殊数据结构

### Params

```cpp
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // BlockMmad 参数（包含 GM 地址）
    L1Params l1Params;              // L1 参数（kL1, scaleKL1, l1BufNum）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // Without Batch QBMM 特有参数
};
```

### QBMMTiling

```cpp
struct QBMMTiling {
    uint32_t baseM;   // L0 tile M 维度
    uint32_t baseN;   // L0 tile N 维度
    uint32_t baseK;   // L0 tile K 维度
    uint32_t isBias;  // 是否启用 bias
    uint32_t dbL0C;   // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U; // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

### BlockMmadParams

```cpp
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr; // B 矩阵 Scale GM 地址
};
```

## 公共接口

### operator() 函数

```cpp
__aicore__ inline void operator()(const Params& params)
```

该方法是 Kernel 的公开调用入口，内部调用私有的 `Run(params)`。外部代码应通过 `kernel(params)` 执行计算。

## 内部实现

### Run 函数

```cpp
__aicore__ inline void Run(const Params& params)
```

`Run` 是私有方法，用于执行单 Batch MX 量化矩阵乘，仅供 `operator()` 内部调用。

执行流程：

1. AIV 核直接返回
2. 配置 Atomic Add，并在该模式下禁用 C 的 L2 Cache（可选）
3. 绑定 GM 地址并创建 BlockScheduler
4. 初始化 BlockMmadMX
5. 调用 `Process` 创建 GM Tensor，并按 tile 通过 `ProcessOneBlock` 调用 BlockMmadMX
6. 清理 Atomic Add（可选）

### SetBL2Cache 函数

```cpp
template <typename TensorB>
__aicore__ inline void SetBL2Cache(
    const ProblemShape& problemShape,
    uint64_t currentBasicBlockM,
    uint64_t currentBasicBlockN,
    uint32_t bMustHitL2,
    TensorB& gmB)
```

`SetBL2Cache` 是私有方法，用于动态配置 B 矩阵的 L2 Cache hint。

说明：

- `bMustHitL2` 非 0 时，B 的 L2 Cache hint 保持为 `NORMAL`
- `bMustHitL2` 为 0、M tile 覆盖完整 M 维，且 B 已转置或当前 N tile 满足相应量化格式的对齐要求时，将 B 的 L2 Cache hint 设置为 `DISABLE`
- 当前实现不配置 ScaleB 的 L2 Cache hint，ScaleB 使用 Tensor API 默认 Cache 策略

## 调用示例

完整可编译、可运行并带 golden 校验的示例见
[quant_batch_matmul_mx](../../../../examples/quant_batch_matmul/quant_batch_matmul_mx/README.md)，
对应 CSV 场景为 `qbmm_mx_without_batch`。

```cpp
using AType = fp8_e4m3fn_t;
using BType = fp8_e5m2_t;
using CType = float;
using BiasType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

constexpr uint64_t FULL_LOAD_MODE = Blaze::Gemm::NONE_FULL_LOAD_MODE;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

QBMMKernel::Params params{};
params.problemShape = {m, n, k, 1};
params.mmadParams = {x1GM, x2GM, yGM, biasGM, perTokenScaleGM, scaleGM};
params.l1Params = {kL1, scaleKL1, l1BufNum};
params.schParams = {baseM, baseN, mTailTile, nTailTile,
                    mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain};
params.qbmmParams = {baseM, baseN, baseK, isBias, dbL0C, bMustHitL2};

QBMMKernel kernel;
kernel(params);
```

其中 `problemShape` 的 Batch 必须为 1；`perTokenScaleGM` 和 `scaleGM` 分别指向
`fp8_e8m0_t` 的 A per-token scale 与 B per-group scale。

## 适用场景

- Batch 固定为 1 且无需广播的 MX 量化推理
- 希望减少多 Batch 分支和地址偏移开销的 QBMM MX 场景
