# Block Mmad Basic
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_basic.h)

## 功能说明
基础矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。支持 L1/L0C 可配置缓冲、Bias 加法，适用于 Basic Kernel 场景。

**继承自**：BlockMmad 基础模板（特化实现）

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulMultiBlockBasic<>`（非全载模式）
- `MatmulMultiBlockBasic<B_FULL_LOAD_MODE>`（B 矩阵全载）
- `MatmulMultiBlockBasic<A_FULL_LOAD_MODE>`（A 矩阵全载）
- `MatmulMultiBlockBasic<FULL_LOAD_MODE_, FUSED_OP_TYPE_, KernelSchedule_, NON_CONTIGUOUS_TYPE_SLICE>`（A 矩阵 ND slice 非连续场景）

不支持 `MatmulMultiBlockWithStreamK` 等其他调度策略。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
结果直接输出到 GM，不支持 workspace。

### Layout Trait
使用 `IsTrans` 和 `IsWeightNz` traits 判断 Layout：
- `IsTrans<LayoutA>::value`：判断 A 矩阵是否转置
- `IsTrans<LayoutB>::value`：判断 B 矩阵是否转置
- `IsWeightNz<LayoutB>::value`：判断 B 矩阵是否为 NZ 格式

### L1 缓冲布局
```
L1 空间布局（2 buffer）：
AL1Ping|BL1Ping|BiasPing|AL1Pong|BL1Pong|BiasPong

L1 空间布局（4 buffer）：
AL1Buf0|BL1Buf0|BiasBuf0|AL1Buf1|BL1Buf1|BiasBuf1|AL1Buf2|BL1Buf2|BiasBuf2|AL1Buf3|BL1Buf3|BiasBuf3
```

### MM Layout Transform
构造函数和析构函数中设置 MM Layout Transform：
```
// 构造函数（ASCEND_IS_NOT_AIV）
SetMMLayoutTransform(true);  // 适配 Fixpipe 输出

// 析构函数（ASCEND_IS_NOT_AIV）
SetMMLayoutTransform(false); // 关闭
```

## 模板参数

| 参数 | 类型 | 说明 |
|------|------|------|
| FULL_LOAD_MODE_ | uint64_t | 全载模式：0=非全载, 1=A全载, 2=B全载 |
| FUSED_OP_TYPE_ | uint64_t | 融合操作类型 |
| KernelSchedule_ | class | Kernel 调度类型 |
| NON_CONTIGIOUS_TYPE_ | uint64_t | 非连续类型：0=连续路径，`NON_CONTIGUOUS_TYPE_SLICE`=A 矩阵 ND slice 非连续路径 |
| AType_ | class | A 矩阵数据类型 |
| LayoutA_ | class | A 矩阵布局类型 |
| BType_ | class | B 矩阵数据类型 |
| LayoutB_ | class | B 矩阵布局类型 |
| CType_ | class | C 矩阵输出类型 |
| LayoutC_ | class | C 矩阵布局类型 |
| BiasType_ | class | Bias 数据类型 |
| LayoutBias_ | class | Bias 布局类型 |


## 特殊静态常量

| 常量 | 说明 |
|------|------|
| transA | A 矩阵是否转置（通过 IsTrans trait 判断） |
| transB | B 矩阵是否转置（通过 IsTrans trait 判断） |
| weightNZFormat | B 矩阵是否为 NZ 格式（通过 IsWeightNz trait 判断） |
| NON_CONTIGIOUS_TYPE | 非连续类型，来自 `DispatchPolicy` |
| MTE1_MTE2_EVENT_ID_NUM | L1 双缓冲事件标志数量（固定 4 个） |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    GM_ADDR aGmAddr{nullptr};         // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};         // B 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};         // C 矩阵 GM 地址
    GM_ADDR biasGmAddr{nullptr};      // Bias GM 地址
    GM_ADDR groupListGmAddr{nullptr}; // GroupList 地址（预留扩展）
    GM_ADDR workspaceGmAddr{nullptr}; // Workspace 地址（预留扩展）
    uint64_t ml1{0};                  // L1 M 维度尺寸
    uint64_t nl1{0};                  // L1 N 维度尺寸
    uint64_t kl1{0};                  // L1 K 维度尺寸
    uint32_t ml0{0};                  // L0 M 维度尺寸
    uint32_t nl0{0};                  // L0 N 维度尺寸
    uint32_t kl0{0};                  // L0 K 维度尺寸
    uint32_t l1Stages{1};             // L1 缓冲数量
    uint16_t l0cStages{1};            // L0C 缓冲数量
};
```

### 参数详解

#### GM 地址参数
| 参数 | 类型 | 说明 |
|------|------|------|
| aGmAddr | GM_ADDR | A 矩阵 GM 地址 |
| bGmAddr | GM_ADDR | B 矩阵 GM 地址 |
| cGmAddr | GM_ADDR | C 矩阵 GM 地址 |
| biasGmAddr | GM_ADDR | Bias GM 地址（nullptr 表示无 bias） |

#### L1/L0 形状参数
| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| ml1 | uint64_t | L1 M 维度尺寸 | 128~256 |
| nl1 | uint64_t | L1 N 维度尺寸 | 128~256 |
| kl1 | uint64_t | L1 K 维度尺寸 | 64~128 |
| ml0 | uint32_t | L0 M 维度尺寸 | 64~128 |
| nl0 | uint32_t | L0 N 维度尺寸 | 64~128 |
| kl0 | uint32_t | L0 K 维度尺寸 | 32~64 |

#### 缓冲配置参数
| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| l1Stages | uint32_t | L1 缓冲数量 | 1、2 或 4（默认 1） |
| l0cStages | uint16_t | L0C 缓冲数量 | 1 或 2（默认 1） |

## 公共成员方法（Public API）

### 构造函数
```cpp
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmad 对象，初始化硬件事件标志和 MM Layout Transform。
执行流程：
1. ASCEND_IS_NOT_AIV 时设置 4 个 MTE1_MTE2 标志、2 个 FIX_M 标志、2 个 M_MTE1 标志
2. ASCEND_IS_NOT_AIV 时调用 `SetMMLayoutTransform(true)`（适配 Fixpipe）

### 析构函数
```cpp
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmad 对象，等待硬件事件完成并关闭 MM Layout Transform。
执行流程：
1. ASCEND_IS_NOT_AIV 时等待 4 个 MTE1_MTE2 标志、2 个 FIX_M 标志、2 个 M_MTE1 标志
2. ASCEND_IS_NOT_AIV 时调用 `SetMMLayoutTransform(false)`（关闭）

### Init函数
```cpp
__aicore__ inline void Init(
    const TupleShape& shape,     // 问题规模
    const Params& params)        // BlockMmad 参数
```
功能：初始化 BlockMmad 组件。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | TupleShape | 问题规模 `(m, n, k, batch)` |
| params | Params | BlockMmad 参数 |

执行流程：
1. 设置问题规模：m_, n_, k_
2. 设置 L1/L0 形状：mL1_, nL1_, kL1_, baseM_, baseN_, baseK_
3. 判断 Bias：`isBias_ = params.biasGmAddr != nullptr`
4. 设置缓冲策略：l1Stages_, enableL0cPingPong_
5. 计算缓冲偏移：aL1Buffer_[i], bL1Buffer_[i], biasL1Buffer_[i]

### operator函数
```cpp
template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(
    TensorA& gmA,                // A 矩阵 GM Tensor
    TensorB& gmB,                // B 矩阵 GM Tensor
    TensorBias& gmBias,          // Bias GM Tensor
    TensorC& gmC,                // C 矩阵 GM Tensor
    TupleL1L0Shape& tileShape)   // Tile 形状
```
功能：执行单个 block 的矩阵乘计算。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| gmA | TensorA | A 矩阵输入 Tensor（已 Slice 到当前 block） |
| gmB | TensorB | B 矩阵输入 Tensor（已 Slice 到当前 block） |
| gmBias | TensorBias | Bias 输入 Tensor（已 Slice） |
| gmC | TensorC | C 矩阵输出 Tensor（已 Slice 到当前 block） |
| tileShape | TupleL1L0Shape | Tile 形状 `(mL1, nL1, k, batch, mL0, nL0)` |

执行流程：
1. **K 轴外层循环**：按 kL1 切分
2. **搬运 A/B/Bias 到 L1**：根据 l1Stages 决定缓冲数量；slice 非连续 A 使用原有
   `CopySliceGM2L1`；仅当策略显式指定 `NON_CONTIGUOUS_TYPE_BATCHED_B` 时，batch Layout B 构造与二维 BL1
   共用地址的 batch 视图，并调用一次标准 batched `CopyGM2L1` 沿 N 拼接；默认策略和普通二维 B 保持原连续搬运
3. **K 轴内层循环**：按 baseK 切分
4. **搬运 A/B 到 L0**：双缓冲模式
5. **Mmad 计算**：首次迭代时加载 Bias
6. **结果搬出**：L0C → GM（通过 Fixpipe）

## 事件同步

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 (0-3) | L1 缓冲同步（4 个标志） |
| FIX_M (0-1) | L0C 双缓冲同步（2 个标志） |
| M_MTE1 (6-7) | L0 双缓冲同步（2 个标志） |
| MTE2_MTE1 | GM→L1 完成同步 |
| MTE1_M | L1→L0 完成同步 |

## 调用示例

### 组件组装
```cpp
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = asc::te::nz_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn;
using LayoutC = asc::te::nd_layout_ptn;
using LayoutBias = LayoutC;

using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0>;  // 非全载
// A 矩阵 ND slice 非连续场景可使用：
// using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
//     0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, Blaze::Gemm::NON_CONTIGUOUS_TYPE_SLICE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 参数准备
```cpp
BlockMmad::Params params = {
    .aGmAddr = aGM,
    .bGmAddr = bGM,
    .cGmAddr = cGM,
    .biasGmAddr = biasGM,  // nullptr 表示无 bias
    .ml1 = 256,
    .nl1 = 256,
    .kl1 = 128,
    .ml0 = 128,
    .nl0 = 128,
    .kl0 = 64,
    .l1Stages = 2,         // L1 双缓冲
    .l0cStages = 1         // L0C 单缓冲
};
```

### 组件实例化
```cpp
BlockMmad blockMmad;
```

### 组件初始化
```cpp
TupleShape problemShape{m, n, k, batch};
blockMmad.Init(problemShape, params);
```

### 组件执行
```cpp
// 准备 GM Tensor（已在 kernel 层创建）
auto gmA = asc::te::make_tensor(...);
auto gmB = asc::te::make_tensor(...);
auto gmC = asc::te::make_tensor(...);
auto gmBias = asc::te::make_tensor(...);

// Slice 到当前 block
auto gmBlockA = gmA.slice(AscendC::MakeCoord(coordM, 0), AscendC::MakeShape(shapeM, shapeK));
auto gmBlockB = gmB.slice(AscendC::MakeCoord(0, coordN), AscendC::MakeShape(shapeK, shapeN));
auto gmBlockC = gmC.slice(AscendC::MakeCoord(coordM, coordN), AscendC::MakeShape(shapeM, shapeN));
auto gmBlockBias = gmBias.slice(AscendC::MakeCoord(0, coordN), AscendC::MakeShape(1, shapeN));

// 执行矩阵乘
TupleL1L0Shape tileShape{shapeM, shapeN, shapeK, batch, mL0, nL0};
blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, tileShape);
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → L1 (多缓冲) → L0A/L0B (双缓冲) → L0C (单缓冲或双缓冲) → GM (C)
```

### 执行流程
```
K 轴外层循环：按 kL1 切分
    ↓
搬运 A、B、Bias 到 L1（多缓冲）
    ↓
K 轴内层循环：按 baseK 切分
    ↓
搬运 A、B 到 L0（双缓冲）
    ↓
Mmad 计算：C += A × B + Bias（首次迭代）
    ↓
结果搬出：L0C → GM（Fixpipe）
```

## 性能优化建议

### L1 缓冲配置
- **单缓冲（l1Stages=1）**：小矩阵场景，减少缓冲开销
- **双缓冲（l1Stages=2）**：中等矩阵场景，平衡并行度和开销
- **四缓冲（l1Stages=4）**：大矩阵场景，最大化流水线并行度

### L0C 双缓冲
- **单缓冲（l0cStages=1）**：小矩阵场景，默认配置
- **双缓冲（l0cStages=2）**：大矩阵场景，隐藏搬出延迟

### L1/L0 形状配置
- **mL1/baseM 成倍数关系**：减少尾块开销
- **nL1/baseN 成倍数关系**：减少尾块开销
- **kL1/baseK 成倍数关系**：减少尾块开销



### NZ 格式优化
- **权重矩阵（B）**：优先使用 NZ 格式，提升 L1/L0 搬运效率
- **激活矩阵（A）**：使用 ND 格式即可

### 适用场景
- Basic Kernel 的 BlockMmad 实现
- 不需要 workspace 中间结果
- 不需要 AIC-AIV 跨核同步
