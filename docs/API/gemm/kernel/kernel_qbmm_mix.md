# Kernel Qbmm Mix
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mix.h)

## 功能说明
该 Kernel 面向支持 Batch 广播的 MIX 量化矩阵乘场景，采用 AIC（cube）与 AIV（vector）双核协同：
AIC 完成矩阵乘，并通过 Fixpipe（NoQuant）将 L0C 中的 int32 累加结果搬入 UB；AIV 执行反量化和
Bias 计算，输出 bf16/fp16/fp32。Kernel 支持多 Batch 和尾块切分，调度行为与
`QuantBmmPertokenRegbaseKernel` / `AL1FullLoad` 保持一致。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)
**配套组件**：[block_mmad_a8w8_mix](../block/block_mmad_a8w8_mix.md)（AIC 矩阵乘）+ [block_epilogue_dequant](../../epilogue/block/block_epilogue_dequant.md)（AIV 反量化）

## 特殊约束

### 调度策略限制
仅匹配 `BlockMmad::DispatchPolicy::ScheduleType == KernelMmadWithScaleMix` 的特化（见 [dispatch_policy](../../../../include/blaze/gemm/policy/dispatch_policy.h) 的 `MatmulWithScaleMix`）。该 Kernel 通过模板 SFINAE（`enable_if_t<is_same_v<KernelMmadWithScaleMix, ScheduleType>>`）与其它 `GemmUniversal` 特化区分。

### 计算模式
AIC + AIV 双核：
- `ASCEND_IS_AIC`：执行 cube 矩阵乘，L0C→UB（fixpipe NoQuant），不做 scale/bias。
- `ASCEND_IS_AIV`：执行 dequant 向量后处理，写最终结果到 GM。

### 量化格式支持

Kernel 在模板实例化时通过 `static_assert` 校验以下组合：

- A/B 支持 `int8_t` 同型、`hifloat8_t` 同型，以及 `fp8_e4m3fn_t` / `fp8_e5m2_t` 的 FP8 组合；
- BlockEpilogue 输出支持 `half`、`bfloat16_t` 和 `float`；
- `LayoutA` 支持 ND/DN，`LayoutB` 支持 ND/DN/NZ/ZN；
- BlockEpilogue 的 L0C 类型必须与 BlockMmad 一致。

dtype/format 校验不包含 bias 类型、`LayoutBias` 或 scale 类型约束，也不按 A/B 类型限制
bias/scale 组合。`BlockEpilogueDequant` 仍根据 `Params::biasDtype` 分派实际 bias 搬运；
scale 的有效类型和量化模式由原有实现及调用方保证。

`BlockMmad` 的 `LayoutC` 是占位标签，不参与实际输出布局选择，因此不对其增加编译期断言。
真实的 L0C→UB 和 epilogue 写回仍使用实现中构造的 ND 布局；这不表示支持 DN 物理输出。

x2Scale 支持 per-channel / per-tensor，x1Scale 支持 per-token / per-tensor（可选）。不在上述范围内的
dtype 或 format 组合会在 Kernel 入口产生明确的编译错误。

### BlockScheduler 限制
仅支持 `BlockSchedulerQbmm`，支持 4 维 Batch 切分与尾块切分（mTailTile / nTailTile）。

### Batch 维度限制
支持 4 维 Batch（batchA1/A2/A3/A4、batchB1/B2/B3/B4、batchC1/C2/C3/C4），需满足广播规则 `batchA = batchC × multiA`、`batchB = batchC × multiB`。`batchC1..C4` 必须 ≥ 1（作为循环上界与除数）；Kernel 内含除零防御卫语句，畸形 0 输入会直接返回。

## 特殊成员方法

### 构造/析构函数
```
__aicore__ inline GemmUniversal()
__aicore__ inline ~GemmUniversal()
```

### 特殊模板参数
```
template <
    class ProblemShape,      // 问题形状类型 (m, n, k, batch)
    class BlockMmad,         // BlockMmad（A8W8 MIX），ScheduleType 必须为 KernelMmadWithScaleMix
    class BlockEpilogue,     // BlockEpilogueDequant（AIV 反量化后处理）
    class BlockScheduler>    // BlockSchedulerQbmm 调度器
```

### 模板参数说明
| 参数 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、b（batch） |
| BlockMmad | A8W8 MIX BlockMmad，基于 `MatmulWithScaleMix` 调度策略 |
| BlockEpilogue | `BlockEpilogueDequant`，AIV 侧反量化后处理 |
| BlockScheduler | BlockSchedulerQbmm 调度器 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| WEIGHT_NZ | B 矩阵是否为 NZ 格式（继承自 BlockMmad） |
| TRANS_A / TRANS_B | A/B 是否转置（继承自 BlockMmad） |
| IS_ATOMIC_ADD | 是否启用 Atomic Add（继承自 BlockMmad::DispatchPolicy） |
| C0_SIZE | C0 对齐大小（int8: 32） |
| MakeLayoutA / MakeLayoutB | A/B 的 FrameLayout 构建器 |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;       // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;      // mmad 参数（A/B/C GM 地址）
    BlockSchedulerParams schParams;  // scheduler 参数（含 mTailTile / nTailTile）
    QBMMTiling qbmmParams;           // QBMM 特有 tiling
    EpilogueParams epilogueParams;   // dequant epilogue 参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t batchA1..A4, batchB1..B4, batchC1..C4;  // 4 维 Batch
    uint32_t biasThreeDim;     // bias 是否为 3 维（按 batch 偏移）
    uint32_t x1QuantMode;      // x1（per-token / per-tensor）量化模式
    uint32_t x2QuantMode;      // x2（per-channel / per-tensor）量化模式
    uint32_t kAL1, kBL1;       // A/B 的 L1 K 轴切分
    uint32_t nBufferNum;       // L1 缓冲数量
    uint32_t baseM, baseN, baseK;  // L0 tile 形状
    uint32_t isBias;           // 是否启用 bias
    uint32_t dbL0C;            // L0C 双缓冲标志（>1 启用 ping-pong）
    uint32_t bMustHitL2 = 1U;  // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

## 特殊成员方法

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能：设置 bias 三维标志；AIC 侧记录 A/B 的 GM 基址。

### Run函数
```
__aicore__ inline void Run(const Params& params)
```
执行流程：
1. `Init(params)` 设置标志与 GM 基址。
2. 构造 BlockScheduler。
3. AIC：用 `{baseM, baseN, baseK}` 与 `kAL1/kBL1/nBufferNum/dbL0C` 初始化 BlockMmad。
4. AIV：用 `epilogueParams` 初始化 BlockEpilogueDequant。
5. 单 Batch（`b == 1`）：`AddBatchOffset` + `ProcessSingleBatch`；多 Batch：`ProcessWithBatch`。

### ProcessWithBatch函数
```
__aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs)
```
功能：4 维 Batch 循环，逐 Batch 更新 A/B/C 偏移并调用 `ProcessSingleBatch`。
说明：进入除法前对 `batchC1..C4 == 0` 做防御（除零即返回）；尾块更新状态
`needUpdateTail_` 会跨 Batch 保持，并计入剩余 Batch 的 tile 数（`restBatch * GetTotalCnt()`），
保证多 Batch + tail-split 不会错位。

### ProcessSingleBatch函数
```
__aicore__ inline void ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
```
执行流程（每个 tile）：
1. 取 tile 坐标 (mPos, nPos) 与形状 (curM, curN)。
2. **AIC**：必要时 `WaitForVector()`；Slice A/B 到当前 tile；按 `CeilAlign(curN, L0C_ALIGN)` 对齐 UB 行距（与 AIV 读取行距一致，避免 N-tail 错位）；调用 BlockMmad 写 L0C→UB；`NotifyVector()`。
3. **AIV**：`WaitForCube()`；计算 scale/ptScale/bias/C 偏移（bias 三维时叠加 `batchCOffset_ * n`）；调用 BlockEpilogueDequant；`NotifyCube()`。

### AIC<->AIV 同步
| 方法 | 作用 |
|------|------|
| NotifyVector / WaitForVector | AIC 通知 / 等待 AIV（PIPE_FIX，flag 0 与 0+16） |
| NotifyCube / WaitForCube | AIV 通知 / 等待 AIC（PIPE_V，flag 0） |

## 调用示例

完整可编译、可运行并带 golden 校验的示例见
[quant_batch_matmul_mix](../../../../examples/acl/quant_batch_matmul/quant_batch_matmul_mix/README.md)，
对应 CSV 场景为 `qbmm_mix`。

以下示例以单路 `int8` 输入、`bfloat16_t` 输出为例。GM 地址和切分参数由上层 host tiling
计算后传入 Kernel；示例显式填写全部组件参数，便于区分 MMAD、Scheduler、QBMM 和 Epilogue
各自负责的字段。

### 组件组装

```cpp
using AType = int8_t;
using BType = int8_t;
using OutType = bfloat16_t;
using BiasType = int32_t;
using X2ScaleType = float;
using X1ScaleType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

constexpr uint64_t FULL_LOAD_MODE = Blaze::Gemm::NONE_FULL_LOAD_MODE;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<FULL_LOAD_MODE, false>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB,
    int32_t, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<
    OutType, BiasType, X2ScaleType, X1ScaleType, int32_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备与 Kernel 执行

```cpp
QBMMKernel::Params params{};
params.problemShape = {m, n, k, batch};

params.mmadParams.aGmAddr = x1GM;
params.mmadParams.bGmAddr = x2GM;
params.mmadParams.problemShape = params.problemShape;
params.mmadParams.l0TileShape = {baseM, baseN, baseK, 0};
params.mmadParams.kAL1 = kAL1;
params.mmadParams.kBL1 = kBL1;
params.mmadParams.l1BufferNum = nBufferNum;
params.mmadParams.enableL0CPingPong = dbL0C > 1U;

params.schParams = {baseM, baseN, mTailTile, nTailTile,
                    mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain};

params.qbmmParams.batchA1 = batchA1;
params.qbmmParams.batchA2 = batchA2;
params.qbmmParams.batchA3 = batchA3;
params.qbmmParams.batchA4 = batchA4;
params.qbmmParams.batchB1 = batchB1;
params.qbmmParams.batchB2 = batchB2;
params.qbmmParams.batchB3 = batchB3;
params.qbmmParams.batchB4 = batchB4;
params.qbmmParams.batchC1 = batchC1;
params.qbmmParams.batchC2 = batchC2;
params.qbmmParams.batchC3 = batchC3;
params.qbmmParams.batchC4 = batchC4;
params.qbmmParams.biasThreeDim = biasThreeDim;
params.qbmmParams.x1QuantMode = x1QuantMode;
params.qbmmParams.x2QuantMode = x2QuantMode;
params.qbmmParams.kAL1 = kAL1;
params.qbmmParams.kBL1 = kBL1;
params.qbmmParams.nBufferNum = nBufferNum;
params.qbmmParams.baseM = baseM;
params.qbmmParams.baseN = baseN;
params.qbmmParams.baseK = baseK;
params.qbmmParams.isBias = isBias;
params.qbmmParams.dbL0C = dbL0C;
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

## 数据流
```
AIC:  GM(int8 A/B) → L1 → L0A/L0B → L0C(int32) --fixpipe NoQuant--> UB
                                   │  (Sync::NotifyVector)
AIV:  UB(int32) × x2Scale [× x1Scale] + bias --VF dequant--> GM(bf16/fp16/fp32)
```

## 性能优化建议
- `nBufferNum`：2 或 4，平衡 L1 容量与流水线并行度。
- `dbL0C > 1`：启用 L0C ping-pong，重叠 AIC 计算与 AIV 后处理。
- A 全载模式（`A_FULL_LOAD_MODE`）：A 常驻 L1，适用于大 K、小 M。
- 多 Batch 与尾块场景由 `needUpdateTail_` 和 `restBatch` 共同维护调度状态；修改其中一条
  处理路径时，需要同步检查另一条路径的状态更新逻辑。

## 适用场景
- 支持多 Batch 的 int8 量化 Matmul（per-token / per-channel / per-tensor）。
- WeightNz（FRACTAL_NZ）权重布局下的量化 MatMul。
- 带 bias（bf16/fp16/fp32）的量化推理。
