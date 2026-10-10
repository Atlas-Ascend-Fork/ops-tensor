# BlockMmad Matmul BL1 Full Load
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_bl1_full_load.h)

## 功能说明
B 矩阵 L1 全载矩阵乘 Block，B 一次搬入 L1 常驻，A 流水搬入。支持两种输出模式：
- **ON_THE_FLY**：L0C 直销 GM
- **Fixpipe**：L0C→UB（AIC 侧），带 AIC-AIV 跨核同步

使用 `BufferManager` 统一管理 L1、L0、BT 缓冲和事件同步。基于 Tensor API 实现 L0/L1 数据搬运和 Mmad 计算。

**特化自**：[block_mmad.md](./block_mmad.md) 公共模板，按 `MatmulMultiBlockBFullLoad` 调度策略特化。

## 调度策略

```cpp
template <uint64_t L0C2OUT_MODEL, uint64_t FUSED_OP_TYPE, class KernelSchedule>
using DispatchPolicy = MatmulMultiBlockBFullLoad<L0C2OUT_MODEL, FUSED_OP_TYPE, KernelMmadMultiBlockBFullLoad>;
```

### L0C2OUT_MODEL
| 值 | 含义 | 输出行为 |
|----|------|---------|
| `ON_THE_FLY` (0) | 直销 GM | `CopyL0C2GM` + `FixpipeParams{FINAL_ACCUMULATION}` |
| `ND_FIXPIPE_1_1` (1) | Fixpipe fp16 | `CopyL0C2UB` + `Sync::WaitForVector/NotifyVector` |
| `ND_FIXPIPE_1_2` (2) | Fixpipe fp32 | `CopyL0C2UB` + `Sync::WaitForVector/NotifyVector` + `splitM` |

## 特殊架构

### B 全载缓冲区布局

```
L1 总空间
├── A L1 缓冲 × l1Stages（流水缓冲）
├── B L1 缓冲 × 1（单缓冲，一次搬入常驻）
├── Bias L1 缓冲 × 1（单缓冲，与 B 同时搬入）
└── BT 缓冲（Bias L0 中转，baseN × sizeof(float)）
```

关键差异 vs [BlockMmadMatmulBasic](./block_mmad_matmul_basic.md)：B 仅分配 1 个 L1 slot，A 保留 l1Stages 个 slot 做流水。

### ON_THE_FLY 数据流

```
GM(B) ──(一次)──→ L1(B) ──────────→ L0B ──┐
GM(A) ──(每K批)──→ L1(A) ──→ L0A ──→  Mmad  ──→ L0C ──→ GM(C)
```

### Fixpipe 数据流

```
GM(B) ──(一次)──→ L1(B) ──────────→ L0B ──┐               AIC │ AIV
GM(A) ──(每K批)──→ L1(A) ──→ L0A ──→  Mmad  ──→ L0C ──→ UB ═══→ GM(C)
                                          ↑                     ↑
                                    AIC-AIV Sync         BlockEpilogueFixpipe
```

- AIC 侧：Mmad 计算 → `WaitFlag(AIV→UB_free)` → `CopyL0C2UB` → `SetFlag(UB_data_ready)`
- AIV 侧（epilogue）：`WaitFlag(UB_data_ready)` → `DataCopyPad(UB→GM)` → `SetFlag(UB_free)`

## Params 参数结构

```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};         // A 矩阵 GM 基址
    GM_ADDR bGmAddr{nullptr};         // B 矩阵 GM 基址
    GM_ADDR cGmAddr{nullptr};         // C 矩阵 GM 基址
    GM_ADDR biasGmAddr{nullptr};      // Bias GM 基址
    GM_ADDR groupListGmAddr{nullptr}; // 预留（group list）
    GM_ADDR workspaceGmAddr{nullptr}; // 预留（workspace）
    uint64_t oriK{0};                 // K 维度原始尺寸（未对齐）
    uint64_t mL1{0};                  // M 方向 L1 tile 尺寸
    uint64_t nL1{0};                  // N 方向 L1 tile 尺寸
    uint64_t kL1{0};                  // K 方向 L1 tile 尺寸
    uint32_t mL0{0};                  // M 方向 L0 tile 尺寸（baseM）
    uint32_t nL0{0};                  // N 方向 L0 tile 尺寸（baseN）
    uint32_t kL0{0};                  // K 方向 L0 tile 尺寸（baseK）
    uint32_t l1Stages{1};             // A 矩阵 L1 流水级数
    uint16_t l0cStages{1};            // L0C 缓冲级数
    uint64_t splitM{0};               // fp32 fixpipe 时 splitM=1
    uint8_t ubDB{1};                  // UB double buffer 开关
};
```

### 参数说明

| 参数 | 适用模式 | 说明 |
|------|---------|------|
| `splitM` | Fixpipe fp32 | 启用 M 方向分半处理，分两半写入 UB |
| `ubDB` | Fixpipe | 值为 1 时 ping-pong 禁用；>1 时启用 UB ping-pong，跨 tile 交替使用 slot 0/1 实现 loc2ub 与 ub2gm 并行 |
| `l1Stages` | 通用 | A 矩阵 L1 缓冲数；B 矩阵固定 1 stage |

## 核心方法

### Init
```cpp
__aicore__ inline void Init(const Params& params)
```
初始化 tile 尺寸、缓冲区布局（A 多缓冲 + B 单缓冲 + BT 缓冲）、B 加载状态标志。

### operator()
```cpp
template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias,
                                  TensorC& tensorC, TupleShape& tileShape)
```
执行一轮 tile 计算：
1. 首次进入时加载 B + Bias 到 L1（`isBL1Loaded_` 标志控制）
2. 按 K 方向迭代，每轮搬入 A tile 到 L1
3. 分割 K→L0，执行 L1→L0 搬运 + Mmad 计算
4. 按 N 方向迭代，每轮调用 `CopyOutFromL0C`
   - ON_THE_FLY: L0C→GM
   - Fixpipe: WaitFlag → L0C→UB → SetFlag

### CopyOutFromL0C
```cpp
template <typename TensorC, typename TensorL0C>
__aicore__ inline void CopyOutFromL0C(TensorC& tensorC, TensorL0C& tensorL0C,
                                      uint64_t tileN, uint64_t curM, uint64_t iterN)
```
ON_THE_FLY 模式：`Copy(copyL0C2GM.with(FixpipeParams{FINAL_ACCUMULATION}), gmBlockC, tensorL0C)`
Fixpipe 模式：`Sync::WaitForVector`（等 AIV 空闲）→ `CopyOutFromL0C2UB` → `Sync::NotifyVector`（通知 AIV 就绪）

### CopyOutFromL0C2UB
```cpp
template <typename TensorUB, typename TensorL0C>
__aicore__ inline void CopyOutFromL0C2UB(TensorUB& tensorC, TensorL0C& tensorL0C,
                                         uint64_t tileN, uint64_t curM, uint16_t slotIdx)
```
L0C→UB 拷贝，对齐到 `c0_element`。splitM 时使用 `CopyL0C2UBTraitSplitM`。

## 使用示例

```cpp
// ON_THE_FLY 模式
using DispatchPolicy = MatmulMultiBlockBFullLoad<ON_THE_FLY, 0>;
using BlockMmadType = BlockMmad<DispatchPolicy, half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn,
                                 half, asc::te::nd_ext_layout_ptn, half, asc::te::nd_ext_layout_ptn>;
BlockMmadType blockMmad;
blockMmad.Init({.aGmAddr = aGM, .bGmAddr = bGM, .cGmAddr = cGM,
                .oriK = k, .mL1 = 128, .nL1 = 256, .kL1 = 512,
                .mL0 = 16, .nL0 = 16, .kL0 = 16, .l1Stages = 2});
blockMmad(gmBlockA, gmB, gmBias, gmBlockC, tileShape);

// Fixpipe 模式
using DispatchPolicy = MatmulMultiBlockBFullLoad<ND_FIXPIPE_1_1, 0>;
// ... 组装 BlockMmad<DispatchPolicy, ...>
blockMmad.Init({..., .splitM = 0, .ubDB = 1});
// AIC 侧调用 blockMmad(gmBlockA, gmB, gmBias, ubLocal, validBlockShape);
// AIV 侧调用 epilogueOp(validBlockShape, offsetC, splitM, baseM, baseN, ubDB);
```

## 跨核同步标志

| 常量 | 值 | 方向 | 含义 |
|------|-----|------|------|
| `AIV_SYNC_AIC_FLAG` | 4 | AIV→AIC | UB 槽位空闲，可写入下一轮 |
| `AIC_SYNC_AIV_FLAG` | 6 | AIC→AIV | UB 数据就绪，可开始 GM 搬出 |
| `FLAG_ID_MAX` | 16 | splitM 偏移 | splitM 场景使用 `flag_id + 16` 做辅助同步 |
