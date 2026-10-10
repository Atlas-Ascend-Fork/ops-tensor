# BlockMmad Matmul Fixpipe Opti
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h)

## 功能说明
Fixpipe 优化的非全载矩阵乘 Block，A 和 B 均通过 L1 流水缓冲搬入，输出固定使用 Fixpipe（L0C→UB）。与 [BlockMmadMatmulBL1FullLoad](./block_mmad_matmul_bl1_full_load.md) 的区别是 B 不做全载，A/B 各自有独立的 L1 流水缓冲。

使用 `BufferManager` 统一管理 L1、L0、BT 缓冲和事件同步。基于 Tensor API 实现 L0/L1 数据搬运和 Mmad 计算。

**特化自**：[block_mmad.md](./block_mmad.md) 公共模板，按 `MatmulMultiBlockFixpipeOpti` 调度策略特化。

## 调度策略

```cpp
template <uint64_t L0C2OUT_MODEL, uint64_t FUSED_OP_TYPE>
using DispatchPolicy = MatmulMultiBlockFixpipeOpti<L0C2OUT_MODEL, FUSED_OP_TYPE,
                                                    KernelMmadMultiBlockFixpipeOpti>;
```

### FULL_LOAD_MODE
固定为 `NONE_FULL_LOAD_MODE`（值 0），A 和 B 均不做全载。

### L0C2OUT_MODEL
| 值 | 含义 |
|----|------|
| `ND_FIXPIPE_1_1` (1) | Fixpipe1v1 fp16 场景 |
| `ND_FIXPIPE_1_2` (2) | Fixpipe1v2 fp32 场景 |

> 注意：此 Block 不支持 `ON_THE_FLY` 直销模式，输出始终为 L0C→UB→GM。ON_THE_FLY 场景应使用 `BlockMmadMatmulBasic` 或 `BlockMmadMatmulBL1FullLoad`。

## 特殊架构

### L1 缓冲区布局

```
L1 总空间 (QUADRUPLE_BUFFER_COUNT 等分)
├── slot[0]  ├─ A L1[i]  ├─ B L1[i]  └─ Bias L1[i]
├── slot[1]  ├─ A L1[i]  ├─ B L1[i]  └─ Bias L1[i]
├── slot[...] ...
└── slot[l1Stages-1]
```

每个 slot 内 A/B/Bias 按需分配。A 和 B 各有 l1Stages 个缓冲级用于流水。

### 对比 BL1 Full Load

| 特性 | BlockMmadFixpipeOpti | BlockMmadBL1FullLoad |
|------|---------------------|---------------------|
| B 矩阵 | 每 K 迭代重新搬入 L1 | 一次搬入，常驻 L1 |
| B L1 缓冲 | l1Stages 级 | 1 级 |
| A L1 缓冲 | l1Stages 级 | l1Stages 级 |
| 输出模式 | 仅 Fixpipe | ON_THE_FLY + Fixpipe |
| 适用场景 | 小 K 场景 | 大 K/N 场景 |

### 数据流

```
GM(A) ──(每K批)──→ L1(A) ──→ L0A ──┐               AIC │ AIV
GM(B) ──(每K批)──→ L1(B) ──→ L0B ──→  Mmad  ──→ L0C ──→ UB ═══→ GM(C)
                                          ↑                     ↑
                                     AIC-AIV Sync         BlockEpilogueFixpipe
```

- AIC 侧：Mmad 计算 → `WaitFlag(AIV→UB_free)` → `CopyL0C2UB` → `SetFlag(UB_data_ready)`
- AIV 侧（epilogue）：`WaitFlag(UB_data_ready)` → `DataCopyPad(UB→GM)` → `SetFlag(UB_free)`

## Params 参数结构

```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};
    GM_ADDR bGmAddr{nullptr};
    GM_ADDR cGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    GM_ADDR groupListGmAddr{nullptr};
    GM_ADDR workspaceGmAddr{nullptr};
    uint64_t oriK{0};
    uint64_t mL1{0};
    uint64_t nL1{0};
    uint64_t kL1{0};
    uint32_t mL0{0};
    uint32_t nL0{0};
    uint32_t kL0{0};
    uint32_t l1Stages{1};
    uint16_t l0cStages{1};
    uint64_t splitM{0};
    uint8_t ubDB{1};
};
```

参数含义与 [BlockMmadMatmulBL1FullLoad Params](./block_mmad_matmul_bl1_full_load.md#params-参数结构) 一致，不再赘述。

## 核心方法

### Init
```cpp
__aicore__ inline void Init(const Params& params)
```
初始化 tile 尺寸、L1 缓冲区布局（A/B 各 l1Stages 级 + BT 缓冲）。使用 `QUADRUPLE_BUFFER_COUNT` 均分 L1 总空间。

### operator()
```cpp
template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias,
                                  TensorC& tensorC, TupleShape& tileShape)
```
执行一轮 tile 计算：
1. 按 N 方向迭代（baseN 为单位）
2. 按 K 方向迭代，每轮同时搬入 A 和 B tile 到 L1
3. 分割 K→L0，执行 L1→L0 搬运 + Mmad 计算
4. 每轮 N 迭代结束时：`Sync::WaitForVector`（等 AIV 空闲）→ `CopyOutFromL0C2UB` → `Sync::NotifyVector`（通知 AIV 就绪）

tensorC 应为 UB tensor，AIC 写入后由 AIV epilogue 搬出。

## 跨核同步标志

与 [BlockMmadMatmulBL1FullLoad](./block_mmad_matmul_bl1_full_load.md) 使用相同的同步协议：

| 常量 | 值 | 方向 | 含义 |
|------|-----|------|------|
| `AIV_SYNC_AIC_FLAG` | 4 | AIV→AIC | UB 槽位空闲 |
| `AIC_SYNC_AIV_FLAG` | 6 | AIC→AIV | UB 数据就绪 |
| `FLAG_ID_MAX` | 16 | splitM 偏移 | splitM 辅助同步 |

## 使用示例

```cpp
// Fixpipe1v1 fp16 场景
using DispatchPolicy = MatmulMultiBlockFixpipeOpti<ND_FIXPIPE_1_1, 0>;
using BlockMmadType = BlockMmad<DispatchPolicy, half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn,
                                 half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn>;
BlockMmadType blockMmad;
blockMmad.Init({.aGmAddr = aGM, .bGmAddr = bGM, .cGmAddr = cGM,
                .oriK = k, .mL1 = 128, .nL1 = 256, .kL1 = 512,
                .mL0 = 16, .nL0 = 16, .kL0 = 16, .l1Stages = 2,
                .splitM = 0, .ubDB = 1});
// AIC 侧：blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubLocal, validBlockShape);
// AIV 侧：epilogueOp(validBlockShape, offsetC, splitM, baseM, baseN, ubDB);
```

## 相关文档

- [BlockMmadMatmulBasic](./block_mmad_matmul_basic.md) — ON_THE_FLY 直销的 BlockMmad
- [BlockMmadMatmulBL1FullLoad](./block_mmad_matmul_bl1_full_load.md) — B 全载 BlockMmad，同时支持 ON_THE_FLY 和 Fixpipe
- [BlockEpilogueFixpipe](../../epilogue/block/block_epilogue_fixpipe.md) — Fixpipe 后处理
