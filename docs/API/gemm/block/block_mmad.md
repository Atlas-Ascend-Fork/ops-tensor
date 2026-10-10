# Block Mmad 基础框架
> 公共接口说明

## 概述
Block 层矩阵乘计算组件，执行单个 block 的矩阵乘计算。基于 Tensor API 实现，支持 Bias 加法、L1/L0 多级双缓冲优化。不同实现（Basic、StreamK、MX）在此基础上扩展特定功能，其中 MX 实现支持量化计算（MxFP4/FP8）和 Scale 因子处理。

详见：[README.md](./README.md) 查看 API 清单和实现对比。


## 类模板概述

### 模板参数
| 参数 | 说明 |
|------|------|
| DispatchPolicy_ | 调度策略类型（MatmulMultiBlockBasic 或 MatmulMultiBlockWithStreamK） |
| AType_ | A 矩阵数据类型（如 half, float 等） |
| LayoutA_ | A 矩阵布局类型（nd_ext_layout_ptn 或 nz_layout_ptn） |
| BType_ | B 矩阵数据类型 |
| LayoutB_ | B 矩阵布局类型，支持 NZ 格式 |
| CType_ | C 矩阵（输出）数据类型 |
| LayoutC_ | C 矩阵布局类型 |
| BiasType_ | Bias 数据类型 |
| LayoutBias_ | Bias 布局类型 |

### 类型别名
| 类型 | 说明 |
|------|------|
| AType | A 矩阵数据类型（继承自模板参数） |
| BType | B 矩阵数据类型（继承自模板参数） |
| CType | C 矩阵数据类型（继承自模板参数） |
| BiasType | Bias 数据类型（继承自模板参数） |
| LayoutA | A 矩阵布局类型（继承自模板参数） |
| LayoutB | B 矩阵布局类型（继承自模板参数） |
| LayoutC | C 矩阵布局类型（继承自模板参数） |
| LayoutBias | Bias 布局类型（继承自模板参数） |
| DispatchPolicy | 调度策略类型（继承自模板参数） |
| TupleShape | Tile 形状类型 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` |

### 静态常量
| 常量 | 说明 |
|------|------|
| TRANS_A | A 矩阵是否转置（根据 LayoutA 判断） |
| TRANS_B | B 矩阵是否转置（根据 LayoutB 判断） |

### Layout 构建类型
| 类型 | 说明 |
|------|------|
| MakeLayoutAL1 | A 矩阵 L1 Layout 构建器 |
| MakeLayoutBL1 | B 矩阵 L1 Layout 构建器 |

说明：
- 根据 TRANS_A/TRANS_B 自动选择 nz_layout_ptn 或 zn_layout_ptn

### 核心数据结构

#### Params
```
struct Params {
    GM_ADDR aGmAddr{nullptr};         // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};         // B 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};         // C 矩阵 GM 地址
    GM_ADDR biasGmAddr{nullptr};      // Bias GM 地址
    GM_ADDR groupListGmAddr{nullptr}; // GroupList 地址（预留扩展）
    GM_ADDR workspaceGmAddr{nullptr}; // Workspace 地址（StreamK）
    uint64_t mL1{0};                  // L1 M 维度尺寸
    uint64_t nL1{0};                  // L1 N 维度尺寸
    uint64_t kL1{0};                  // L1 K 维度尺寸
    uint32_t mL0{0};                  // L0 M 维度尺寸
    uint32_t nL0{0};                  // L0 N 维度尺寸
    uint32_t kL0{0};                  // L0 K 维度尺寸
    uint32_t l1Stages{1};             // L1 缓冲数量
    uint16_t l0cStages{1};            // L0C 缓冲数量
};
```

### 核心成员变量
| 变量 | 类型 | 说明 |
|------|------|------|
| mL1_, nL1_, kL1_ | uint64_t | L1 tile 形状 |
| baseM_, baseN_, baseK_ | uint64_t | L0 tile 形状 |
| isBias_ | bool | 是否启用 bias 计算 |

## 核心成员方法

### 构造函数
```
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmad 对象，初始化硬件事件标志。

### 析构函数
```
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmad 对象，等待硬件事件完成。

### Init函数
```
__aicore__ inline void Init(
    const Params& params)        // BlockMmad 参数
```
功能：初始化 BlockMmad 组件，设置 tile 形状和缓冲策略。

### operator函数
```
__aicore__ inline void operator()(
    TensorA gmA,                // A 矩阵输入 Tensor
    TensorB gmB,                // B 矩阵输入 Tensor
    TensorBias gmBias,          // Bias 输入 Tensor
    TensorC gmC,                // C 矩阵输出 Tensor
    TupleShape blockShape,      // Block 形状 (m, n, k, batch)
    ... 其他参数)
```
功能：执行单个 block 的矩阵乘计算。

**公共执行流程**：
1. **K 轴外层循环**：按 kL1 切分，迭代搬运 A、B 块到 L1
2. **K 轴内层循环**：按 baseK 切分，迭代搬运 A、B 块到 L0
3. **Bias 处理**：首次迭代时加载并累加 Bias
4. **Mmad 计算**：执行 `C += A × B + Bias`
5. **结果搬出**：通过 Fixpipe 将 L0C 数据搬出到 GM/workspace

## 公共约束

1. **模板参数要求**：
   - DispatchPolicy 必须为 MatmulMultiBlockBasic 或 MatmulMultiBlockWithStreamK
   - LayoutA/LayoutB 必须为 asc::te 的合法布局类型

2. **数据格式**：
   - A 矩阵：根据 TRANS_A 自动选择布局
   - B 矩阵：支持 ND 和 NZ 格式
   - L0C：固定使用 nz_layout_ptn 布局

3. **Bias 支持**：可选 bias 输入，仅首次迭代时加载

4. **对齐要求**：
   - baseM、baseN 需对齐到 `AscendC::BLOCK_CUBE`（通常为 16）
   - L0C 布局按 16 字节对齐

## 公共调用示例

### 组件组装模板
```
// 定义数据类型和布局
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = LayoutC;

// 定义调度策略
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE>;

// 定义 BlockMmad（根据需求选择实现）
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 初始化模板
```
BlockMmad blockMmad;
BlockMmad::Params params = {
    .aGmAddr = aGM,
    .bGmAddr = bGM,
    .cGmAddr = cGM,
    .biasGmAddr = biasGM,
    .mL1 = mL1, .nL1 = nL1, .kL1 = kL1,
    .mL0 = baseM, .nL0 = baseN, .kL0 = baseK,
    .l1Stages = 2,
    .l0cStages = 1
};
blockMmad.Init(params);
```

### 执行模板
```
TupleShape blockShape{shapeM, shapeN, shapeK, batch};
blockMmad(gmA, gmB, gmBias, gmC, blockShape);
```

## 数据流与流水线

### 存储层次
```
GM (A/B/Bias) → L1 (双缓冲) → L0A/L0B (双缓冲) → L0C → GM/Workspace
```

### 流水线并行
1. **L1 双缓冲**：GM→L1 搬运与 L1→L0 搬运并行
2. **L0 双缓冲**：L1→L0 搬运与 Mmad 计算并行

## 性能优化建议（公共）

1. **Tile 大小选择**：
   - mL1 × kL1 应充分利用 L1 容量
   - baseM × baseK 应匹配 L0A 大小（通常 128KB）
   - baseN × baseK 应匹配 L0B 大小

2. **K 轴切分**：
   - kL1 和 baseK 应根据数据局部性和复用率优化
   - 避免 K 轴切分过于细碎导致搬运开销增加

3. **数据布局**：
   - 权重矩阵（B）优先使用 NZ 格式，提升 L1/L0 搬运效率
   - 激活矩阵（A）使用 ND 格式即可
