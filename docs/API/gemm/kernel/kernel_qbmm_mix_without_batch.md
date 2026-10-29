# Kernel Qbmm Mix Without Batch
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h)

## 功能说明
该 Kernel 面向 Batch 为 1 的 A8W8 MIX 量化矩阵乘场景，与
[kernel_qbmm_mix](./kernel_qbmm_mix.md) 使用相同的计算流程，但不执行 4 维 Batch 广播和地址换算。
**AIC（cube）+ AIV（vector）双核协同**：AIC 完成 int32 矩阵乘，并通过 Fixpipe（NoQuant）将
L0C 结果搬入 UB；AIV 执行反量化和 Bias 计算，输出 bf16/fp16/fp32。输入支持 int8 的
per-token、per-channel 和 per-tensor 量化模式，以及 WeightNz（FRACTAL_NZ）格式。

Bias 的编译期 `BiasType` 和运行时 `Params::biasDtype` 是不同的约束：实际 bias 类型仍由 epilogue
在运行时分派。Kernel 不额外校验 bias 或 scale，也不约束占位 `LayoutC` / `LayoutBias` 标签。
实际输出仍由 Kernel 和 epilogue 构造的 ND 布局决定，不由这些标签选择。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)
**配套组件**：[block_mmad_a8w8_mix](../block/block_mmad_a8w8_mix.md) + [block_epilogue_dequant](../../epilogue/block/block_epilogue_dequant.md)

## 与带 Batch 版本的差异
| 维度 | kernel_qbmm_mix | kernel_qbmm_mix_without_batch |
|------|-----------------|-------------------------------|
| 类名 | `GemmUniversal<...>`（SFINAE 特化） | `GemmUniversal<...>`（`KernelMmadWithScaleMixWithoutBatch` 特化） |
| Batch | 4 维 Batch 广播并跨 Batch 维护尾块状态 | 仅单 Batch，无 Batch 偏移逻辑 |
| 尾块切分 | 通过 `needUpdateTail_` 和 `restBatch` 跨 Batch 更新 | 单轮判断即可 |
| QBMMTiling | batchA/B/C 等 12 个字段 | 仅包含 B 的 L2 Cache 控制字段 |
| 偏移计算 | 含 `batchCOffset_` | `mPos * n + nPos` |

## 特殊约束
- AIC + AIV 双核（同带 Batch 版本）。
- 仅支持 `BlockSchedulerQbmm`，单 Batch；支持尾块切分（mTailTile / nTailTile）。
- A/B、输出 dtype、BlockEpilogue L0C 类型一致性和 A/B layout 的编译期 `static_assert` 约束同
  [kernel_qbmm_mix](./kernel_qbmm_mix.md)。

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;       // 问题 shape (m, n, k)
    BlockMmadParams mmParams;        // mmad 参数（A/B/C GM 地址）
    BlockSchedulerParams schParams;  // scheduler 参数（含 mTailTile / nTailTile）
    EpilogueParams epilogueParams;   // dequant epilogue 参数
    QBMMTiling qbmmParams;           // QBMM 特有 tiling
};
```

### QBMMTiling

```
struct QBMMTiling {
    uint32_t bMustHitL2 = 1U; // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

## 特殊成员方法

### operator函数
```
__aicore__ inline void operator()(const Params& params)
```
执行流程：
1. 构造 BlockScheduler。
2. AIC：用 `{baseM, baseN, baseK}` 与 `kAL1/kBL1/nBufferNum/dbL0C` 初始化 BlockMmad。
3. AIV：用 `epilogueParams` 初始化 BlockEpilogueDequant。
4. 调用 `Run(params, bs)`。

### Run函数
```
__aicore__ inline void Run(const Params& params, BlockScheduler& bs)
```
执行流程：
1. 构建 A/B 的 GM Tensor。
2. 尾块判断：`(GetEndBlockIdx()+1) * mTailTile * nTailTile <= GetBlockNum()` 时 `UpdateTailTile`。
3. Tile 循环：
   - **AIC**：必要时 `WaitForVector()`；Slice A/B；按 `CeilAlign(curN, L0C_ALIGN)` 对齐 UB 行距；BlockMmad 写 L0C→UB；`NotifyVector()`。
   - **AIV**：`WaitForCube()`；调用 epilogue（偏移 `scale=nPos, ptScale=mPos, bias=nPos, C=mPos*n+nPos`）；`NotifyCube()`。
4. 收尾：AIC 若有 tile，`WaitForVector()`。

### AIC<->AIV 同步
同 [kernel_qbmm_mix](./kernel_qbmm_mix.md)：NotifyVector / WaitForVector（PIPE_FIX，flag 0 与 0+16），NotifyCube / WaitForCube（PIPE_V，flag 0）。

## 调用示例

完整可编译、可运行并带 golden 校验的示例见
[quant_batch_matmul_mix](../../../../examples/acl/quant_batch_matmul/quant_batch_matmul_mix/README.md)，
对应 CSV 场景为 `qbmm_mix_without_batch`。

以下示例与带 Batch 版本使用相同的 MMAD、Scheduler 和 Epilogue 组件，但通过
`KernelMmadWithScaleMixWithoutBatch` 选择单 Batch 特化，并省略所有 Batch 广播参数。

```cpp
using AType = int8_t;
using BType = int8_t;
using OutType = bfloat16_t;
using BiasType = int32_t;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

constexpr uint64_t FULL_LOAD_MODE = Blaze::Gemm::NONE_FULL_LOAD_MODE;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<
    FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB,
    int32_t, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<
    OutType, BiasType, float, float, int32_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

QBMMKernel::Params params{};
params.problemShape = {m, n, k, 1};

params.mmParams.aGmAddr = x1GM;
params.mmParams.bGmAddr = x2GM;
params.mmParams.problemShape = params.problemShape;
params.mmParams.l0TileShape = {baseM, baseN, baseK, 0};
params.mmParams.kAL1 = kAL1;
params.mmParams.kBL1 = kBL1;
params.mmParams.l1BufferNum = nBufferNum;
params.mmParams.enableL0CPingPong = dbL0C > 1U;

params.schParams = {baseM, baseN, mTailTile, nTailTile,
                    mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain};
params.qbmmParams.bMustHitL2 = bMustHitL2;

params.epilogueParams.x2ScaleGmAddr = scaleGM;
params.epilogueParams.x1ScaleGmAddr = perTokenScaleGM;
params.epilogueParams.biasGmAddr = biasGM;
params.epilogueParams.outGmAddr = yGM;
params.epilogueParams.m = m;
params.epilogueParams.n = n;
params.epilogueParams.baseM = baseM;
params.epilogueParams.baseN = baseN;
params.epilogueParams.x1QuantMode = x1QuantMode;
params.epilogueParams.x2QuantMode = x2QuantMode;
params.epilogueParams.isBias = isBias != 0U;
params.epilogueParams.biasDtype = biasDtype;

QBMMKernel kernel;
kernel(params);
```

## 适用场景
- 单 Batch int8 量化 Matmul。
- 无需 Batch 广播、追求更轻调度路径的量化推理与 WeightNz 场景。
