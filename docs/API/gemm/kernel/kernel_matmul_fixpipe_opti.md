# Kernel Matmul Fixpipe Opti
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_fixpipe_opti.h)

## 功能说明
Fixpipe 优化的非全载矩阵乘 Kernel，AIC+AIV 双核并行。AIC 做矩阵乘 + L0C→UB，AIV 做 epilogue（UB→GM）。B 矩阵非全载时使用此 Kernel，B 全载 + fixpipe 场景请使用 [KernelMatmulBL1FullLoad](./kernel_matmul_bl1_full_load.md)。

根据 `FULL_LOAD_MODE` 选择 calc BlockMmad：
- `B_FULL_LOAD_MODE` → 路由到 `BlockMmadBL1FullLoad`（B 全载）
- 其他 → 路由到 `BlockMmadFixpipeOpti`（A/B 均流水）

**继承自**：GemmUniversal 基础模板，按 `KernelMmadMultiBlockFixpipeOpti` 调度策略特化。

**适用场景**：小 K 的 fixpipe1v1（fp16）和 fixpipe1v2（fp32）场景，大 K 场景应使用 B 全载 + fixpipe 的 `KernelMatmulBL1FullLoad`。

## 与相关 Kernel 对比

| 特性 | KernelMatmulFixpipeOpti | KernelMatmulBL1FullLoad |
|------|------------------------|------------------------|
| B 矩阵 | 每 K 迭代流水搬入 | 一次搬入，L1 常驻 |
| AIC 侧 BlockMmad | `BlockMmadFixpipeOpti`（非全载） / `BlockMmadBL1FullLoad`（B 全载） | `BlockMmadBL1FullLoad` |
| BlockEpilogue | `BlockEpilogueFixpipe` | `BlockEpilogueEmpty` / `BlockEpilogueFixpipe` |
| 输出模式 | 固定 Fixpipe | ON_THE_FLY + Fixpipe |
| AIV 参与 | 是 | 仅 Fixpipe 模式 |

## 调度策略

```cpp
// nn 仓 mat_mul_fixpipe.h 中的条件选择：
using DispatchPolicy = conditional_t<
    (FULL_LOAD_MODE == B_FULL_LOAD_MODE),
    MatmulMultiBlockBFullLoad<L0C2OUT_MODEL, 0, KernelMmadMultiBlockBFullLoad>,     // B 全载
    MatmulMultiBlockFixpipeOpti<L0C2OUT_MODEL, 0, KernelMmadMultiBlockFixpipeOpti>>; // 非全载
```

## 特殊约束

### AIC-AIV 同步
```cpp
构造:  AIV SetFlag(AIV_SYNC_AIC_FLAG) × 2    // 预置 UB 空闲信号
析构:  AIC WaitFlag(AIV_SYNC_AIC_FLAG) × 4    // 等待所有 epilogue 完成
```

### splitM 子块
```cpp
if ASCEND_IS_AIV {
    if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
        return;  // 非 splitM 时 AIV 子块 1 不参与
    }
    curBlockIdx /= AscendC::GetTaskRation();  // 双核平分 block
}
```

### Calc BlockMmad 路由
```cpp
if ASCEND_IS_AIC {
    if constexpr (FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
        blockMmad(gmBlockA, gmB, gmBias, ubLocal, validBlockShape);      // B 常驻 L1
    } else {
        blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubLocal, validBlockShape); // A/B 均流水
    }
}
```

## Params 参数结构

```cpp
struct Params {
    ProblemShape problemShape;          // (m, n, k, batch)
    BlockMmadParams mmadParams;         // BlockMmad 参数（含 splitM、ubDB）
    BlockEpilogueParams epilogueParams; // BlockEpilogueFixpipe 参数 {cGM}
    BlockSchedulerParams schParams;     // BlockScheduler 参数
};
```

## 使用示例

```cpp
// Fixpipe1v1 fp16，非 B 全载
using DispatchPolicy = MatmulMultiBlockFixpipeOpti<ND_FIXPIPE_1_1, 0>;
using BlockMmad = BlockMmad<DispatchPolicy, half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn,
                             half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn>;
using BlockEpilogue = BlockEpilogueFixpipe<half, half, DispatchPolicy>;
using MatmulKernel = GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

Params params = {
    {m, n, k, batch},           // ProblemShape
    {aGM, bGM, cGM, ..., 0, 1}, // mmadParams（splitM=0, ubDB=1）
    {cGM},                       // epilogueParams
    {...}                        // schParams
};
```

## 组件组装

```text
KernelMatmulFixpipeOpti
    → BlockScheduler (MatmulBasic)
    → BlockMmad (FixpipeOpti 或 BL1FullLoad)
    → BlockEpilogueFixpipe
```

详见：
- [BlockMmadMatmulFixpipeOpti](../block/block_mmad_matmul_fixpipe_opti.md)
- [BlockMmadMatmulBL1FullLoad](../block/block_mmad_matmul_bl1_full_load.md)
- [BlockEpilogueFixpipe](../../epilogue/block/block_epilogue_fixpipe.md)
