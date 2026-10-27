# Block Mmad Flat Quant
> [代码位置](../../../../include/blaze/attention/block/block_mmad_flat_quant.h)

## 功能说明

FlatQuant 双矩阵乘 Block，在 AIC 侧完成两阶段矩阵乘计算：
- **Phase 1**：`A × P2 → L0C → Fixpipe → L1`（中间结果暂存 L1）
- **Phase 2**：`P1 × (中间结果) → L0C → Fixpipe → UB`（最终结果写入 UB，供 AIV 消费）

使用 `BufferManager` 管理 L1/L0/L0C 缓冲，支持 L0 PingPong 和 L0C PingPong 双缓冲。P1/P2 矩阵在首轮加载到 L1 后复用，避免重复搬运。

**继承自**：[BlockMmad 基础框架](./block_mmad.md)（显式特化实现，匹配 `BlockFlatQuant`）

## 特殊约束

### 双矩阵乘两阶段计算
FlatQuant BlockMmad 执行两阶段矩阵乘，中间结果通过 L1 暂存：
1. **Phase 1**：A（GM→L1）× P2（GM→L1，首轮加载）→ L0C → Fixpipe L0C→L1（作为 Phase 2 的输入 B）
2. **Phase 2**：P1（GM→L1，首轮加载）× Phase1 结果（L1）→ L0C → Fixpipe L0C→UB

### 计算模式
仅在 AIC 核执行，AIV 核不参与 BlockMmad 计算。

### 输出目标
结果通过 Fixpipe 写入 UB，供 AIV 侧的 BlockEpilogueFlatQuant 读取并执行 MX 量化。

### P1/P2 复用
- `isFirstRound = true` 时，P1 和 P2 从 GM 加载到 L1
- 后续轮次复用 L1 中的 P1/P2，不再从 GM 搬运

### hasP2 控制
`hasP2 = true` 时执行完整双矩阵乘（Phase 1 + Phase 2）；`hasP2 = false` 时跳过 Phase 1，仅执行 Phase 2（P1 直接与 A 计算）。

### L0C PingPong
当 `alignedML0 * alignedNL0 <= HALF_L0C_SIZE` 时启用 L0C PingPong 双缓冲，否则退化为单缓冲。

## 模板参数

| 参数 | 说明 |
|------|------|
| DispatchPolicy_ | 调度策略类型，固定为 `Attention::BlockFlatQuant<>` |
| AType_ | 提供嵌套类型 `T` 的矩阵乘类型描述，如 `MatmulType<bfloat16_t>` |
| LayoutA_ | A 矩阵布局类型 |
| BType_ | B/P1/P2 的矩阵乘类型描述，须提供嵌套类型 `T` |
| LayoutB_ | B/P1/P2 矩阵布局类型 |
| OutType_ | Out 矩阵输出类型描述，须提供嵌套类型 `T` |
| LayoutC_ | C 矩阵布局类型 |
| CType_ | C 矩阵数据类型描述，须提供嵌套类型 `T` |
| LayoutOut_ | Out 矩阵布局类型 |

## 类型别名

| 类型 | 说明 |
|------|------|
| AType / BType / CType / OutType | 数据类型（继承自模板参数） |
| A_T / B_T / C_T / Out_T | 底层数据类型（`::T`） |
| L0cType | L0C 累加类型，固定为 `float` |
| DispatchPolicy | `Attention::BlockFlatQuant<>` |
| TupleShape | `Shape<int64_t, int64_t, int64_t, int64_t>` |

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| HALF_L0C_SIZE | L0C 半缓冲大小（元素数），`TOTAL_L0C_SIZE / 2 / sizeof(float)` |
| HALF_L0C_SIZE_BYTES | L0C 半缓冲大小（字节），`TOTAL_L0C_SIZE / 2` |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};       // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};       // P1 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};       // P2 矩阵 GM 地址
    TupleShape problemShape{};      // 问题规模 (m, n, k, batch)
    TupleShape tileL1{};            // L1 tile 形状 (mL1, nL1, kL1, iterBatch)
    TupleShape tileL0{};            // L0 tile 形状 (0, 0, baseK, 0)
    bool hasP2{true};               // 是否执行 Phase 1（A×P2）
};
```

### 参数详解

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| aGmAddr | GM_ADDR | A 矩阵 GM 地址 | aGM |
| bGmAddr | GM_ADDR | P1 矩阵 GM 地址 | p1GM |
| cGmAddr | GM_ADDR | P2 矩阵 GM 地址 | p2GM |
| problemShape | TupleShape | 问题规模 (m, n, k, batch) | {128, 128, 64, 1} |
| tileL1 | TupleShape | L1 tile (mL1, nL1, kL1, iterBatch) | {128, 128, 128, 2} |
| tileL0 | TupleShape | L0 tile，仅 baseK 有效 | {0, 0, 64, 0} |
| hasP2 | bool | 是否执行 Phase 1 | true |

## 公共成员方法（Public API）

### 构造函数
```cpp
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmadFlatQuant 对象。

### 析构函数
```cpp
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmadFlatQuant 对象。

### Init函数
```cpp
__aicore__ inline void Init(const Params& params)
```
功能：初始化 BlockMmad 组件。

执行流程：
1. 设置问题规模：m_, n_, k_
2. 设置 L1/L0 形状：mL1_, nL1_, kL1_, iterBatch_, baseK_
3. 计算 A L1 单缓冲大小和偏移
4. 计算 P2 L1 偏移（bl1OffsetP2_）
5. 初始化 PingPong 计数器和 BufferManager（L0/L0C）

### operator函数
```cpp
template <typename TensorA, typename TensorP1, typename TensorP2>
__aicore__ inline void operator()(
    TensorA gmA,           // A 矩阵 GM Tensor（已 Slice 到当前 block）
    TensorP1 gmP1,         // P1 矩阵 GM Tensor
    TensorP2 gmP2,         // P2 矩阵 GM Tensor
    TupleShape tileShape,  // Tile 形状 (mL1, nL1, kL1, iterBatch)
    bool isFirstRound)     // 是否首轮（控制 P1/P2 加载）
```
功能：执行双矩阵乘计算。

执行流程：
```
Phase 1 (hasP2=true):
  A(GM→L1) + P2(GM→L1, 首轮)
      ↓
  K 轴循环: A(L1) × P2(L1) → L0A/L0B → L0C (Mmad)
      ↓
  Fixpipe L0C → L1 (temp 结果)
      ↓
Phase 2:
  P1(GM→L1, 首轮) + temp(L1)
      ↓
  Batch 循环:
    K 轴循环: P1(L1) × temp(L1) → L0A/L0B → L0C (Mmad)
      ↓
  Fixpipe L0C → UB (bf16, 供 AIV 消费)
```

## 私有方法

### CopyGM2L1
```cpp
template <typename TensorDst, typename TensorSrc>
__aicore__ inline void CopyGM2L1(TensorDst& dst, const TensorSrc& src, const Gemm::BufferSlot& slot)
```
功能：GM→L1 数据搬运，使用 `CopyGM2L1` Copy 操作，通过 slot 锁管理 MTE2 流水同步。

### CopyL1ToL0
```cpp
template <typename L1Tensors, typename SlotsTuple, typename L0Shape>
__aicore__ inline auto CopyL1ToL0(const L1Tensors& l1Tensors, const L0Shape& l0Shape,
                                  const Gemm::BufferSlot& l0Slot, const SlotsTuple& slotsTuple)
```
功能：L1→L0A/L0B 数据搬运，A 矩阵使用 nz_layout_ptn，B 矩阵使用 zn_layout_ptn。返回 L0A/L0B Tensor tuple。

参数说明：
| 参数 | 说明 |
|------|------|
| l1Tensors | (tensorAL1, tensorBL1) tuple |
| l0Shape | (curML0, curNL0, curK0, kOffset, bRowOffset) |
| l0Slot | L0 缓冲 slot |
| slotsTuple | (aL1Slot, bL1Slot) tuple |

### Mmad
```cpp
template <typename TensorL0C, typename L0Tensors, typename SlotsTuple, typename MnkShape>
__aicore__ inline void Mmad(TensorL0C& tensorL0C, const L0Tensors& l0Tensors, const MnkShape& mnkShape,
                            uint8_t unitFlag, bool cmatrixInitVal, const SlotsTuple& slots)
```
功能：执行 Mmad 矩阵乘计算 `C = A × B`，通过 slot 锁管理 M 流水同步。

### FixpipeL0CToL1
```cpp
template <typename TensorsTuple, typename SlotsTuple>
__aicore__ inline void FixpipeL0CToL1(const TensorsTuple& tensors, const SlotsTuple& slots)
```
功能：L0C→L1 Fixpipe 搬运（Phase 1 中间结果写回 L1）。

### FixpipeL0CToUB
```cpp
template <typename TensorsTuple>
__aicore__ inline void FixpipeL0CToUB(const TensorsTuple& tensors, const Gemm::BufferSlot& l0cSlot)
```
功能：L0C→UB Fixpipe 搬运（Phase 2 最终结果写入 UB，供 AIV 消费），输出类型为 `bfloat16_t`。

## BufferManager

使用 `Gemm::BufferManager<2, 2, 2>` 管理 L1/L0/L0C 缓冲：
- L1A：双缓冲（slot 0/1）
- L1B：单缓冲（slot 0，存放 P1/P2）
- L0：双缓冲（PingPong）
- L0C：双缓冲（PingPong，条件启用）

### L1 空间布局
```
L1 空间布局：
AL1Buf0 | AL1Buf1 | BL1Buf(P1+P2)
```

## 调用示例

### 组件组装
```cpp
// 与 examples/flat_quant/flat_quant/flat_quant.cpp 使用同一类型描述方式。
template <typename T_>
struct MatmulType {
    using T = T_;
};

using AType = MatmulType<bfloat16_t>;
using BType = MatmulType<bfloat16_t>;
using CType = MatmulType<bfloat16_t>;
using OutType = MatmulType<bfloat16_t>;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutOut = LayoutC;

using DispatchPolicy = Blaze::Attention::BlockFlatQuant<>;
using BlockMmad = Blaze::Attention::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC, CType, LayoutOut>;
```

### 参数准备
```cpp
BlockMmad::Params params = {
    .aGmAddr = aGM,
    .bGmAddr = p1GM,
    .cGmAddr = p2GM,
    .problemShape = {m, n, k, batch},
    .tileL1 = {mL1, nL1, kL1, iterBatch},
    .tileL0 = {0, 0, baseK, 0},
    .hasP2 = true
};
```

### 组件初始化与执行
```cpp
BlockMmad blockMmad;
blockMmad.Init(params);

// GM Tensor 由 Kernel 层创建并 Slice
// isFirstRound 控制首轮 P1/P2 加载
blockMmad(gmBlockA, gmP1, gmP2, tileShape, isFirstRound);
```

## 数据流

### 存储层次
```
Phase 1: GM(A, P2) → L1(A, P2) → L0A/L0B → L0C → Fixpipe → L1(temp)
Phase 2: GM(P1) → L1(P1) + L1(temp) → L0A/L0B → L0C → Fixpipe → UB(bf16)
```

### 流水线并行
1. **L0 PingPong**：L1→L0 搬运与 Mmad 计算并行
2. **L0C PingPong**：Mmad 计算与 Fixpipe 搬出并行（条件启用）
3. **P1/P2 复用**：首轮加载后常驻 L1，后续轮次免搬运

## 性能优化建议

### iterBatch 配置
- `iterBatch` 控制每轮处理的 batch 数量，影响 L1 A 缓冲大小
- `aL1OneBuffer = CeilAlign(iterBatch * m, BLOCK_CUBE) * CeilAlign(kL1, BLOCK_CUBE)`
- 较大 `iterBatch` 提高吞吐但增加 L1 压力

### baseK 配置
- `baseK` 控制 L0 K 轴切分粒度
- 建议 `kL1 / baseK` 为整数，减少尾块

### L0C PingPong
- 当 `mL1 * nL1` 较大时可能超出 `HALF_L0C_SIZE`，此时退化为单缓冲
- 适当减小 `mL1` 或 `nL1` 以启用 PingPong

### 适用场景
- FlatQuant Kernel 的 AIC 侧双矩阵乘
- 需要将矩阵乘结果通过 L0C→UB 传递给 AIV 后处理的场景
