# Kernel Matmul BL1 Full Load
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_matmul_bl1_full_load.h)

## 功能说明
B 矩阵 L1 全载 Kernel，B 矩阵常驻 L1，A 矩阵流水搬入。支持两种输出模式：
- **ON_THE_FLY**（`L0C2OUT_MODEL = 0`）：仅 AIC 计算，L0C 直销 GM，无 workspace
- **Fixpipe**（`L0C2OUT_MODEL != 0`）：AIC+AIV 双核并行，AIC 做 L0C→UB，AIV 做 UB→GM，需要 `BlockEpilogueFixpipe`

**继承自**：GemmUniversal 基础模板，按 `KernelMmadMultiBlockBFullLoad` 调度策略特化。

**适用场景**：大 K 或大 N 的矩阵乘，B 矩阵一次搬入 L1 避免重复搬运。当同时启用 fixpipe 输出模式时，支持 fixpipe1v1（fp16）和 fixpipe1v2（fp32）场景。

## 与 KernelMatmulBasic 的区别

| 特性 | KernelMatmulBasic | KernelMatmulBL1FullLoad |
|------|-------------------|------------------------|
| B 矩阵驻留 | 每次 K 迭代重新搬 B | 一次搬入，常驻 L1 |
| B L1 缓冲 | 双缓冲/多缓冲 | 单缓冲（固定 slot 0） |
| A L1 缓冲 | 与 B 共用多缓冲 | 独立多缓冲（l1Stages） |
| 输出模式 | 仅 L0C→GM | L0C→GM（ON_THE_FLY）或 L0C→UB→GM（Fixpipe） |
| AIC-AIV 支持 | 仅 AIC | ON_THE_FLY: 仅 AIC；Fixpipe: AIC+AIV |

## 调度策略

使用 `MatmulMultiBlockBFullLoad<L0C2OUT_MODEL, FUSED_OP_TYPE>` 调度策略：
```cpp
using DispatchPolicy = MatmulMultiBlockBFullLoad<L0C2OUT_MODEL, FUSED_OP_TYPE,
                                                  KernelMmadMultiBlockBFullLoad>;
```

### FULL_LOAD_MODE
固定为 `B_FULL_LOAD_MODE`（值 2），由 `MatmulMultiBlockBFullLoad` 内联定义。

### L0C2OUT_MODEL
| 值 | 含义 | BlockEpilogue |
|----|------|---------------|
| `ON_THE_FLY` (0) | L0C 直销 GM | `BlockEpilogueEmpty` |
| `ND_FIXPIPE_1_1` (1) | Fixpipe1v1 fp16 | `BlockEpilogueFixpipe` |
| `ND_FIXPIPE_1_2` (2) | Fixpipe1v2 fp32 | `BlockEpilogueFixpipe` |

## 特殊约束

### ON_THE_FLY 模式
仅 AIC 执行，AIV 直接返回：
```cpp
if constexpr (L0C2OUT_MODEL == ON_THE_FLY) {
    if ASCEND_IS_AIV { return; }
}
```

### Fixpipe 模式
- AIC 和 AIV 同时参与，共享整个 UB（fixpipe 通道）
- AIC 做矩阵乘 + L0C→UB（`CopyL0C2UB`）
- AIV 做 epilogue（UB→GM，`DataCopyPad`）
- 需要 AIC-AIV 跨核同步（`Gemm::Sync` 块内同步标志）
- epilogue 在 AIV 侧处理，不改写 AIC 侧逻辑

### 同步机制
```
构造:  AIV SetFlag(4), SetFlag(5)           // UB 初始空闲
循环:  AIC WaitFlag(4+slot)                 // 等待 UB 空闲
       AIC L0C→UB (CopyOutFromL0C2UB)
       AIC SetFlag(6+slot)                  // 通知 AIV 数据就绪
                                         AIV WaitFlag(6+slot)   // 等待数据
                                         AIV UB→GM (DataCopyPad)
                                         AIV SetFlag(4+slot)    // UB 空闲
析构:  AIC WaitFlag(4), WaitFlag(5), ...    // 等待所有 epilogue 完成
```

## Params 参数结构

### 结构定义
```cpp
struct Params {
    ProblemShape problemShape;          // 问题规模 (m, n, k, batch)
    BlockMmadParams mmadParams;         // BlockMmadBL1FullLoad 参数
    BlockEpilogueParams epilogueParams; // BlockEpilogue 参数
    BlockSchedulerParams schParams;     // BlockScheduler 参数
};
```

### BlockMmadParams 关键参数
| 参数 | 类型 | 说明 |
|------|------|------|
| aGmAddr / bGmAddr / cGmAddr / biasGmAddr | GM_ADDR | A, B, C, Bias 的 GM 基址 |
| oriK | uint64_t | K 维度原始大小 |
| mL1 / nL1 / kL1 | uint64_t | L1 tile 尺寸 |
| mL0 / nL0 / kL0 | uint32_t | L0 tile 尺寸 |
| l1Stages | uint32_t | A 矩阵 L1 缓冲级数 |
| l0cStages | uint16_t | L0C 缓冲级数 |
| splitM | uint64_t | fp32 时启用（=1）|
| ubDB | uint8_t | UB double buffer 开关 |

### BlockEpilogueParams
ON_THE_FLY 模式使用 `BlockEpilogueEmpty::Params{}`（空），Fixpipe 模式使用 `BlockEpilogueFixpipe::Params{cGM}`。

## 使用示例

### ON_THE_FLY 模式（B 全载 + 直销）
```cpp
using DispatchPolicy = MatmulMultiBlockBFullLoad<ON_THE_FLY, 0>;
using BlockMmad = BlockMmad<DispatchPolicy, ...>;
using BlockEpilogue = BlockEpilogueEmpty;
using MatmulKernel = GemmUniversal<..., BlockMmad, BlockEpilogue, BlockScheduler>;

Params params = {
    {m, n, k, batch},     // ProblemShape
    {aGM, bGM, cGM, ...}, // BlockMmadParams
    {},                    // BlockEpilogueParams（空）
    {...}                  // BlockSchedulerParams
};
MatmulKernel mm;
mm(params);
```

### Fixpipe 模式（B 全载 + fixpipe1v1/1v2）
```cpp
using DispatchPolicy = MatmulMultiBlockBFullLoad<ND_FIXPIPE_1_1, 0>;
using BlockMmad = BlockMmad<DispatchPolicy, ...>;
using BlockEpilogue = BlockEpilogueFixpipe<OutType, OutType, DispatchPolicy>;
using MatmulKernel = GemmUniversal<..., BlockMmad, BlockEpilogue, BlockScheduler>;

// fp32 元素为 4 字节，UB 半区容量（TOTAL_UB / sizeof(OutType) / 2）只有 fp16 的一半，
// 大 tile 无法放进单个 AIV 子块的 UB 半区，因此启用 splitM 沿 M 拆分，
// 两个 AIV 子块各拷出并写回一半 M 行。真正的 UB 双缓冲开关是 ubDB。
static constexpr bool enableSplitM = IsSameType<OutType, float>::value;
Params params = {
    {m, n, k, batch},     // ProblemShape
    {aGM, bGM, cGM, ..., enableSplitM, ubDB}, // BlockMmadParams（含 splitM/ubDB）
    {cGM},                 // BlockEpilogueParams（传入 cGM 供 epilogue 写回）
    {...}                  // BlockSchedulerParams
};
MatmulKernel mm;
mm(params);
```

## 组件组装

```text
KernelMatmulBL1FullLoad
    → BlockScheduler (MatmulBasic)
    → BlockMmad (BL1FullLoad)
    → BlockEpilogue (Empty 或 Fixpipe)
```

详见：
- [BlockMmadMatmulBL1FullLoad](../block/block_mmad_matmul_bl1_full_load.md)
- [BlockEpilogueFixpipe](../../epilogue/block/block_epilogue_fixpipe.md)（Fixpipe 模式）
