# Block Epilogue 基础框架
> 公共接口说明

## 概述
Block 层后处理组件，用于矩阵乘计算后的额外处理。不同实现（Empty、StreamK）提供不同功能：Empty 为空实现，StreamK 支持 workspace 汇聚、类型转换、ReLU 激活。

详见：[README.md](../README.md) 查看 API 清单和实现对比。

## 类概述

### 类型别名
| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状类型 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` |
| BlockCoord | Block 坐标类型 `asc::te::coord<int64_t, int64_t, int64_t, int64_t>` |

### 核心数据结构

#### Params
```
struct Params {
    // 具体成员根据实现不同
    // 通常为运行时参数
};
```
说明：Kernel 运行时参数结构体。

## 核心成员方法

### 构造函数
```
__aicore__ inline BlockEpilogue()
```
功能：构造 BlockEpilogue 对象。

### 析构函数
```
__aicore__ inline ~BlockEpilogue()
```
功能：析构 BlockEpilogue 对象。

### Init函数
```
__aicore__ inline void Init(Params const& params, ...)
```
功能：初始化 BlockEpilogue 组件，设置 GM 地址、Block 形状等参数。
说明：仅部分实现提供（如 StreamK）；Empty 无 Init 方法。参数数量和类型根据实现不同。

### Run函数
```
__aicore__ inline void Run()
```
功能：执行后处理操作。
说明：Empty 实现为空（直接返回），StreamK 实现包含实际计算。

### operator函数（参数版本）
```
__aicore__ inline void operator()(Params const& params)
```
功能：执行后处理操作（参数版本）。
说明：调用 `Init` 和 `Run`，提供统一调用接口。

### operator函数（Block版本）
```
__aicore__ inline void operator()(
    BlockShape const& blockShape, BlockCoord const& blockCoord,
    int64_t dstStartOffset = 0, int64_t srcStartOffset = 0)
```
功能：执行后处理操作（Block 坐标版本）。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockShape | BlockShape const& | Block 形状 |
| blockCoord | BlockCoord const& | Block 坐标 |
| dstStartOffset | int64_t | 目标起始偏移（默认 0） |
| srcStartOffset | int64_t | 源起始偏移（默认 0） |

## 公共约束

1. **模板参数要求**：
   - Empty 无模板参数要求
   - StreamK 需要 WorkspaceType、OutType、DispatchPolicy 参数

2. **计算位置**：
   - Empty：无计算（空实现）
   - StreamK：在 AIV 核执行

3. **接口一致性**：
   - 所有实现必须提供 `Run`、`operator` 方法并保持统一的类型别名
   - `Init` 仅部分实现提供：Empty 无 Init，StreamK 含 Init

## 公共调用示例

### 组件组装模板
```
// Empty 实现
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

// StreamK 实现
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<float, half, DispatchPolicy>;
```

### 在 Kernel 中使用
```
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```
说明：Kernel 统一为 `GemmUniversal` 模板特化，具体调度（Basic / StreamK 等）由
BlockMmad 的 DispatchPolicy 调度策略（如 `KernelMmadMultiBlockBasic`、`KernelMultiBlockStreamK`）决定。

### 执行模板
```
// Empty（无实际效果）
BlockEpilogue epilogue;
epilogue.Run();
epilogue({});
epilogue(blockShape, blockCoord, 0, 0);

// StreamK（实际后处理）
BlockEpilogue epilogue;
epilogue.Init(params, blockShape, tileL1, coord, usedCoreNum, isSkScene);
epilogue.Run();
```

## 设计说明

### Epilogue 层的作用
Epilogue 层位于矩阵乘计算之后，用于：
1. **结果汇聚**：将 workspace 中 K 轴切分的中间结果累加
2. **类型转换**：将 float 结果转换为 half 或 bfloat16
3. **激活函数**：执行 ReLU 等激活操作
4. **其他后处理**：如量化、缩放等（未来扩展）

### 为什么采用空实现设计
1. **模板参数要求**：Kernel 模板需要 BlockEpilogue 参数，Empty 满足此要求
2. **零开销**：空实现不引入任何计算或内存开销
3. **类型安全**：保持类型系统完整性，编译期检查
4. **扩展性**：可无缝替换为实际实现，无需修改 Kernel 代码

## 性能影响

| Epilogue 类型 | 编译时间 | 运行时间 | 内存占用 | 流水线 |
|--------------|---------|---------|---------|--------|
| BlockEpilogueEmpty | 极小 | 无影响 | 无 | 不参与 |
| BlockEpilogueMatmulStreamK | 正常 | 有影响 | 有 | 参与计算 |
