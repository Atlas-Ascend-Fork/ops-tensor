# Block Mmad StreamK
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_streamk.h)

## 功能说明
StreamK 矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。支持输出到 GM 或 workspace、K 轴切分（K split）、固定双缓冲优化，适用于 StreamK Kernel 场景。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)



## 模板参数

### 模板参数列表
```cpp
template <
    typename DispatchPolicy,    // 调度策略
    typename AType,             // A 矩阵数据类型
    typename LayoutA,           // A 矩阵布局
    typename BType,             // B 矩阵数据类型
    typename LayoutB,           // B 矩阵布局
    typename CType,             // C 矩阵计算类型
    typename LayoutC,           // C 矩阵布局
    typename BiasType,          // Bias 数据类型
    typename LayoutBias         // Bias 布局
>
class BlockMmad;
```

### DispatchPolicy 详解

StreamK BlockMmad 仅支持以下两种 DispatchPolicy：

**ON_THE_FLY 模式**：
```cpp
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<
    Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>;
```
- 实时输出模式，无需 stride 对齐
- 适用于小到中等矩阵
- 较低的内存占用

**ND_FIXPIPE_1_2 模式**：
```cpp
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<
    Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>;
```
- ND 1v2 优化模式，要求 stride 对齐到 32B
- 适用于大矩阵场景
- 更高的输出效率


## 特殊静态常量

| 常量 | 说明 |
|------|------|
| BUFFER_NUM | L1/L0 缓冲数量（固定为 2） |
| HALF_L0_SIZE | L0 缓冲区半大小 |
| L1_EVENT_ID_OFFSET | L1 B 缓冲事件偏移（2） |
| MTE1_MTE2_EVENT_ID_NUM | L1 双缓冲事件数量（4） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR aGmAddr{nullptr};         // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};         // B 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};         // C 矩阵 GM 地址（DP 模式）
    GM_ADDR biasGmAddr{nullptr};      // Bias GM 地址
    GM_ADDR groupListGmAddr{nullptr}; // GroupList 地址（预留扩展）
    GM_ADDR workspaceGmAddr{nullptr}; // Workspace GM 地址（SK 模式）
    uint64_t mL1{0};                  // L1 M 维度尺寸
    uint64_t nL1{0};                  // L1 N 维度尺寸
    uint64_t kL1{0};                  // L1 K 维度尺寸
    uint32_t mL0{0};                  // L0 M 维度尺寸
    uint32_t nL0{0};                  // L0 N 维度尺寸
    uint32_t kL0{0};                  // L0 K 维度尺寸
    uint32_t l1Stages{2};             // L1 缓冲数量
    uint16_t l0cStages{1};            // L0C 缓冲数量
};
```

说明：`cGmAddr` 和 `workspaceGmAddr` 均需提供，根据 `checkIsSkScene` 选择输出目标。

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>`（实时输出模式）
- `MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>`（ND 1v2 优化模式）

不支持 `MatmulMultiBlockBasic` 或 `MatmulWithScaleMx`。

### 输出目标
支持两种输出目标：
- **GM**：DP 模式，结果直接输出到 GM（`checkIsSkScene = false`）
- **Workspace**：SK 模式，结果输出到 workspace（`checkIsSkScene = true`）

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。后处理在 Kernel 层的 AIV 核执行。

### K 轴切分支持
支持 K 轴切分（K split），通过 `kCntIndex` 参数标识当前 K 轴切分索引：
- `kCntIndex = 0`：首次 K 轴切分，Bias 加载并累加
- `kCntIndex > 0`：后续 K 轴切分，累加计算

### L1/L0 缓冲
- **L1 双缓冲**：固定使用 2 个缓冲（`BUFFER_NUM = 2`）
- **L0 双缓冲**：固定使用 2 个缓冲
- **L0C 单缓冲**：固定使用单缓冲（offset = 0）

### L1 缓冲布局
StreamK当前只支持`Double Buffer`
```
L1 空间布局：
a1|b1|bias1|a2|b2|bias2|
```

说明：
- `a1|a2`：A 矩阵双缓冲（ping-pong）
- `b1|b2`：B 矩阵双缓冲（ping-pong）
- `bias1|bias2`：Bias 双缓冲
- 优势：避免 A/B 缓冲区 bank conflict

### Bias 处理
Bias 仅在首次 K 轴切分（`kCntIndex = 0`）时加载并累加：
- 首次迭代且 `kCntIndex = 0`：加载 Bias
- 后续迭代：不加载 Bias，累加计算

### CmatrixInitVal
Cmatrix 初始化值根据迭代位置和 Bias 状态确定：
```
cmatrixInitVal = (iter0 == 0 && iter1 == 0 && (!isBias_ || (isBias_ && kCntIndex != 0)))
```

说明：
- 无 Bias 且首次迭代：初始化为 0
- 有 Bias 且首次 K 切分：不初始化（Bias 提供初始值）
- 有 Bias 且后续 K 切分：初始化为 0

### Layout Trait
使用 `IsTrans` 和 `IsWeightNz` traits 判断 Layout：
- `IsTrans<LayoutA>`：判断 A 矩阵是否转置
- `IsTrans<LayoutB>`：判断 B 矩阵是否转置
- `IsWeightNz<LayoutB>`：判断 B 矩阵是否为 NZ 格式

说明：通过 trait 自动判断 transpose 信息，无需模板参数传递。

### MM Layout Transform
构造函数和析构函数中设置 MM Layout Transform：
```
// 构造函数（ASCEND_IS_NOT_AIV）
SetMMLayoutTransform(true);  // 适配 Fixpipe 输出

// 析构函数（ASCEND_IS_NOT_AIV）
SetMMLayoutTransform(false); // 关闭
```

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmad 对象，初始化硬件事件标志和 MM Layout Transform。
执行流程：
1. 设置 4 个 MTE1_MTE2 标志、2 个 M_MTE1 标志
2. ASCEND_IS_NOT_AIV 时调用 `SetMMLayoutTransform(true)`（适配 Fixpipe）

### 析构函数
```
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmad 对象，等待硬件事件完成并关闭 MM Layout Transform。
执行流程：
1. 等待 4 个 MTE1_MTE2 标志、2 个 M_MTE1 标志
2. ASCEND_IS_NOT_AIV 时调用 `SetMMLayoutTransform(false)`（关闭）

### Init函数
```
__aicore__ inline void Init(
    const Params& params)        // mmad 参数
```
功能：初始化 BlockMmad 组件。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| params | Params | mmad 参数（包含 isBias 判断） |

说明：
- isBias 通过 `params.biasGmAddr != nullptr` 判断

### operator函数
```
template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias, typename TensorWorkspace>
__aicore__ inline void operator()(
    TensorA gmA,                // A 矩阵 GM Tensor
    TensorB gmB,                // B 矩阵 GM Tensor
    TensorBias gmBias,          // Bias GM Tensor
    TensorC gmC,                // C 矩阵 GM Tensor（DP 模式输出）
    TensorWorkspace gmWorkspace, // Workspace GM Tensor（SK 模式输出）
    TupleShape tileShape,       // Tile 形状
    int64_t kCntIndex,          // K 轴切分索引
    bool checkIsSkScene)        // 是否为 SK 模式
```
功能：执行单个 block 的矩阵乘计算。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| gmA | TensorA | A 矩阵输入 Tensor（已 Slice） |
| gmB | TensorB | B 矩阵输入 Tensor（已 Slice） |
| gmBias | TensorBias | Bias 输入 Tensor（已 Slice） |
| gmC | TensorC | C 矩阵输出 Tensor（DP 模式） |
| gmWorkspace | TensorWorkspace | Workspace 输出 Tensor（SK 模式） |
| tileShape | TupleShape | Tile 形状 `(m, n, k)` |
| kCntIndex | int64_t | K 轴切分索引（0 = 首次切分） |
| checkIsSkScene | bool | true = SK 模式（输出到 workspace） |

执行流程：
1. **K 轴外层循环**：按 kL1 切分
2. **搬运 Bias 到 L1**：首次切分且首次迭代时搬运
3. **搬运 A 到 L1**：双缓冲模式（事件 0、1）
4. **搬运 B 到 L1**：双缓冲模式（事件 2、3，偏移 L1_EVENT_ID_OFFSET）
5. **K 轴内层循环**：按 baseK 切分（Iterate）
6. **搬运 A/B/Bias 到 L0**：双缓冲模式
7. **Mmad 计算**：根据 `kCntIndex` 决定是否加载 Bias
8. **结果输出**：
   - SK 模式：输出到 workspace
   - DP 模式：输出到 GM

### L1 B 缓冲事件偏移
```
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);  // B 使用事件 2、3
```

说明：A 使用事件 0、1，B 使用事件 2、3，避免事件冲突。

## 事件同步（StreamK 特有）

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 (0-1) | A 矩阵 L1 双缓冲同步 |
| MTE1_MTE2 (2-3) | B 矩阵 L1 双缓冲同步 |
| M_MTE1 (0-1) | L0 双缓冲同步 |
| MTE2_MTE1 | GM→L1 完成同步 |
| MTE1_M | L1→L0 完成同步 |

说明：A 和 B 使用不同的事件 ID，避免冲突，最大化流水线并行度。

## 调用示例

### 组件组装
```
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nz_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = LayoutC;

using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 参数准备
```
BlockMmad::Params params = {
    aGM,           // A 矩阵 GM 地址
    bGM,           // B 矩阵 GM 地址
    cGM,           // C 矩阵 GM 地址（DP 模式）
    biasGM,        // Bias GM 地址（nullptr 表示无 bias）
    workspaceGM    // Workspace GM 地址（SK 模式）
};
```

### 组件初始化
```
BlockMmad blockMmad;
blockMmad.Init(params);
```

### 组件执行
```
// 准备 GM Tensor
auto gmA = asc::te::make_tensor(...);
auto gmB = asc::te::make_tensor(...);
auto gmC = asc::te::make_tensor(...);          // DP 模式输出
auto gmWorkspace = asc::te::make_tensor(...);  // SK 模式输出
auto gmBias = asc::te::make_tensor(...);

// Slice 到当前 tile
auto gmBlockA = gmA.slice(...);
auto gmBlockB = gmB.slice(...);
auto gmBlockC = gmC.slice(...);
auto gmBlockWorkspace = gmWorkspace.slice(...);
auto gmBlockBias = gmBias.slice(...);

// 执行矩阵乘
TupleShape tileShape{shapeM, shapeN, shapeK};
int64_t kCntIndex = 0;  // K 轴切分索引
bool checkIsSkScene = true;  // SK 模式（输出到 workspace）
blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, gmBlockWorkspace, tileShape, kCntIndex, checkIsSkScene);
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → L1 (双缓冲: a1|b1|bias1|a2|b2|bias2) → L0A/L0B (双缓冲) → L0C → GM/Workspace
```

### DP 模式流程
```
GM → L1 → L0 → Mmad → L0C → GM (C)
```

### SK 模式流程
```
GM → L1 → L0 → Mmad → L0C → Workspace (K 轴切分中间结果)
```

### 执行流程
```
K 轴外层循环：按 kL1 切分
    ↓
搬运 Bias（首次切分） + A + B 到 L1
    ↓
K 轴内层循环：按 baseK 切分
    ↓
搬运 A + B + Bias 到 L0
    ↓
Mmad 计算（根据 kCntIndex 决定 Bias 加载）
    ↓
结果输出：checkIsSkScene 决定输出到 GM 或 Workspace
```

## 性能优化建议

### L1 缓冲配置
- 固定双缓冲（BUFFER_NUM = 2）
- A 和 B 使用不同事件 ID（0-1 vs 2-3），最大化并行度
- L1 布局：`a1|b1|bias1|a2|b2|bias2`，避免 bank conflict

### K 轴切分策略
- `kCntIndex` 用于标识当前 K 轴切分索引
- Bias 仅在首次切分时加载
- 后续切分累加计算（NON_FINAL_ACCUMULATION）

### 输出目标选择
- **DP 模式**：完整 tile，输出到 GM
- **SK 模式**：K 轴切分，输出到 workspace
- workspace 大小：`skKTileNum × BLOCK_BASE_M × BLOCK_BASE_N × sizeof(float)`

### MM Layout Transform
- 构造函数中设置 `SetMMLayoutTransform(true)`
- 析构函数中关闭 `SetMMLayoutTransform(false)`
- 仅在 ASCEND_IS_NOT_AIV 时执行

### L0C 单缓冲
- StreamK BlockMmad 固定使用 L0C 单缓冲
- L0C 双缓冲在 Kernel 层处理（通过多个 tile 并行）

### 适用场景
- **StreamK Kernel**：AIC + AIV 双核协同
- **K 轴切分场景**：K 维度远大于 M/N
- **workspace 输出**：用于 AIV 后处理汇聚
