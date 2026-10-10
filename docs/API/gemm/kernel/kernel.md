# Kernel基础框架
> 公共接口说明

## 概述
矩阵乘 Kernel 基础框架，提供统一的模板参数、数据结构和核心流程。不同实现（Basic、QBMM MX、StreamK）在此基础上扩展特定功能，其中 QBMM MX 支持 MxFP4/MxFP8 量化、Scale 因子处理、多 Batch 维度。

详见：[README.md](./README.md) 查看 API 清单和实现对比。

## 类模板概述

### 模板参数
| 参数 | 说明 |
|------|------|
| ProblemShape_ | 问题形状类型，通常为 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` (m, n, k, batch) |
| BlockMmad_ | BlockMmad 类，矩阵乘计算组件 |
| BlockEpilogue_ | BlockEpilogue 类，后处理组件 |
| BlockScheduler_ | BlockScheduler 类，任务调度组件 |

### 特殊模板参数（量化 Kernel）
| 参数 | 说明 |
|------|------|
| AtomicAdd_ | 是否启用 Atomic Add 模式（QBMM MX） |

说明：QBMM MX Policy 通过 `AtomicAdd_` 模板参数控制多核并行累加，并公开编译期常量
`IS_ATOMIC_ADD` 供 Kernel 使用。

### 类型别名
| 类型 | 说明 |
|------|------|
| ProblemShape | 问题形状类型（继承自模板参数） |
| BlockMmad | BlockMmad 类型（继承自模板参数） |
| BlockEpilogue | BlockEpilogue 类型（继承自模板参数） |
| BlockScheduler | BlockScheduler 类型（继承自模板参数） |
| AType | A 矩阵数据类型（继承自 BlockMmad） |
| BType | B 矩阵数据类型（继承自 BlockMmad） |
| CType | C 矩阵数据类型（继承自 BlockMmad） |
| BiasType | Bias 数据类型（继承自 BlockMmad） |
| LayoutA | A 矩阵布局类型（继承自 BlockMmad） |
| LayoutB | B 矩阵布局类型（继承自 BlockMmad） |
| LayoutC | C 矩阵布局类型（继承自 BlockMmad） |
| LayoutBias | Bias 布局类型（继承自 BlockMmad） |
| TupleShape / TileShape | Tile 形状类型 `asc::te::shape<...>` |

### Layout 构建类型
| 类型 | 说明 |
|------|------|
| MakeLayoutA | A 矩阵 Layout 构建器 `frame_layout_format<LayoutA, ...>` |
| MakeLayoutB | B 矩阵 Layout 构建器 `frame_layout_format<LayoutB, ...>` |
| MakeLayoutC | C 矩阵 Layout 构建器 `frame_layout_format<LayoutC, ...>` |
| MakeLayoutBias | Bias Layout 构建器 `frame_layout_format<LayoutBias, ...>` |

### 核心数据结构

#### Params
```
struct Params {
    ProblemShape problemShape;            // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;           // mmad 参数（包含 GM 地址）
    BlockEpilogueParams epilogueParams;   // epilogue 参数
    BlockSchedulerParams schParams;       // scheduler 参数
};
```

### 核心成员变量
| 变量 | 类型 | 说明 |
|------|------|------|
| problemShape_ | TupleShape | 问题规模 (m, n, k, batch) |
| isBias_ | bool | 是否启用 bias 计算 |
| aGmAddr_ | `__gm__ AType*` | A 矩阵 GM 地址 |
| bGmAddr_ | `__gm__ BType*` | B 矩阵 GM 地址 |
| cGmAddr_ | `__gm__ CType*` | C 矩阵 GM 地址 |
| biasGmAddr_ | `__gm__ BiasType*` | Bias GM 地址（可选） |

## 核心成员方法

### 构造函数
```cpp
__aicore__ inline GemmUniversal()
```
功能：构造 GemmUniversal Kernel 对象。

### 析构函数
```cpp
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal Kernel 对象。

### Init函数
```
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Kernel，提取问题规模和 GM 地址。
执行流程：
1. 设置问题规模 `problemShape_`
2. 提取 BlockMmad 参数 `mmadParams`
3. 设置 A、B、C 的 GM 地址
4. 判断 bias 地址是否为 nullptr，设置 `isBias_` 和 `biasGmAddr_`

### operator函数
```
__aicore__ inline void operator()(Params const& params)
```
功能：执行矩阵乘 Kernel 计算。

**公共执行流程**：
1. **初始化**：调用 `Init(params)` 设置参数
2. **BlockScheduler 初始化**：创建调度器，获取 tile 信息
3. **Layout 构建**：构建 A、B、C、Bias 的 ND layout
4. **GM Tensor 创建**：创建 A、B、C、Bias 的 GM Tensor
5. **Tile 循环处理**：遍历 tile 执行矩阵乘计算
6. **清理**：关闭 HF32/MM Layout Transform

## operator执行流程

### 执行流程
```
operator(params)
    ↓
Init：设置问题规模、GM地址
    ↓
BlockScheduler初始化
    ↓
Block索引检查：超出实际数量则返回
    ↓
HF32模式设置（可选）
    ↓
BlockMmad初始化
    ↓
创建GM Tensor（使用Tensor API）
    ├── Layout构建：frame_layout_format<LayoutPattern, c0_element>
    ├── MemPtr创建：make_mem_ptr<asc::te::location::gm>
    └── Tensor创建：make_tensor(memPtr, layout)
    ↓
L2 Cache配置（可选）
    ↓
Tile循环处理
    ├── GetBlockShape：获取当前tile形状
    ├── GetBlockCoord：获取当前tile坐标
    ├── Slice Tensor：gmA.slice(coord, shape)
    └── BlockMmad执行
    ↓
清理：关闭HF32模式
```

### Tensor API使用示例
```cpp
// Layout构建（使用frame_layout_format和LayoutPattern）
using LayoutA = asc::te::nz_layout_ptn;      // NZ格式布局
using LayoutB = asc::te::nd_layout_ptn;      // ND格式布局
using LayoutC = asc::te::nd_ext_layout_ptn;   // ND扩展布局

using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

// Layout实例化
auto layoutA = MakeLayoutA{}(m_, k_);  // 创建A矩阵layout (m, k)
auto layoutB = MakeLayoutB{}(k_, n_);  // 创建B矩阵layout (k, n)
auto layoutC = MakeLayoutC{}(m_, n_);  // 创建C矩阵layout (m, n)

// GM Tensor创建（使用Tensor API）
auto gmA = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_),
    layoutA);
auto gmB = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_),
    layoutB);
auto gmC = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_),
    layoutC);

// Tensor Slice操作（获取当前tile的数据）
auto gmBlockA = gmA.slice(
    AscendC::MakeCoord(coordM, 0L),          // 起始坐标
    AscendC::MakeShape(shapeM, shapeK));     // 形状
auto gmBlockB = gmB.slice(
    AscendC::MakeCoord(0L, coordN),
    AscendC::MakeShape(shapeK, shapeN));
auto gmBlockC = gmC.slice(
    AscendC::MakeCoord(coordM, coordN),
    AscendC::MakeShape(shapeM, shapeN));
```

### operator调用示例
```cpp
// 1. 定义Kernel类型
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// 2. 准备Params参数
using Params = typename MatmulKernel::Params;
Params params;
params.problemShape = {m, n, k, batch};
params.mmadParams.aGmAddr = aGM;
params.mmadParams.bGmAddr = bGM;
params.mmadParams.cGmAddr = cGM;
params.mmadParams.biasGmAddr = biasGM;  // 可选
params.schParams.l2CacheMode = L2_CACHE_DEFAULT;

// 3. 实例化并执行
MatmulKernel kernel;
kernel(params);  // 执行矩阵乘计算
```

### Tile循环策略
不同Kernel类型的Tile循环策略：
- **Basic Kernel**：单核stride循环，`tileIdx += blockNum`
- **StreamK Kernel**：DP+SK混合策略，AIC计算+AIV汇聚
- **QBMM MX Kernel**：多Batch循环+Tile循环

```cpp
// 定义数据类型和布局
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::nd_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using LayoutBias = LayoutC;

// 定义问题 shape
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

// 定义 BlockScheduler
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

// 定义 Kernel（根据需求选择 Basic 或 StreamK）
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备模板
```
using Params = typename MatmulKernel::Params;
Params params = {
    {m, n, k, batch},        // problem shape
    {aGM, bGM, cGM, biasGM}, // mmad args
    {...},                   // epilogue args
    {mL1, nL1, kL1, baseM, baseN, baseK, ...} // scheduler params
};
```

### Kernel 执行模板
```
MatmulKernel mm;
mm(params);
```

## 公共约束

1. **模板参数要求**：
   - ProblemShape 必须为 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` 类型, 分别表示 **m n k b** 维度大小。
   - BlockMmad 必须继承自相应的 BlockMmad 基类
   - BlockEpilogue 必须与 Kernel 类型匹配
   - BlockScheduler 必须提供 tile 切分和调度功能

2. **数据格式**：
   - A、C、Bias：支持 ND 格式
   - B 矩阵：支持 ND 和 NZ 格式

3. **Bias 支持**：可选 bias 输入，通过 `biasGmAddr` 是否为 nullptr 判断

4. **Layout 构建**：使用 `frame_layout_format` 根据数据类型自动适配 layout

## 性能优化建议（公共）

1. **Tile 大小选择**：
   - L1 tile：匹配 L1 容量（通常 512KB）
   - L0 tile：匹配 L0A/L0B 容量（通常 64KB）

2. **Block 数量配置**：根据问题规模合理设置 block 数量

3. **HF32 模式**：FP16 输入 + FP32 输出场景建议启用

4. **数据布局**：权重矩阵（B）优先使用 NZ 格式
