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

#### AccumulatorTensorParams
```
using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;
```
公共定义位于 `block_epilogue_types.h`，包含 `rows`、`cols`、`slot`、`batch`。仅接收 L0C 输出的 Epilogue 需要提供该别名及对应 Tensor 接口。

## 统一对外接口

新增后处理文件必须遵守 [Epilogue 层开发指导](../epilogue_developer_guide.md#2-统一对外接口新增文件必须遵守)。通用 tile 后处理提供以下接口：

```
__aicore__ inline BlockEpilogue();
__aicore__ inline void Init(const Params& params);
__aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;
// operator() 的 Tensor、形状、坐标和偏移入参由具体 Epilogue 场景决定。
```

| 接口 | 说明 |
|------|------|
| 构造函数 | 构造 BlockEpilogue 对象；需要同步或资源清理时提供对应析构函数 |
| `Init` | 一次性初始化，绑定 GM 地址、保存问题规模并配置 UB 布局 |
| `GetAccumulatorTensor` | 返回接收 L0C 输出的 UB Tensor，由 Epilogue 决定地址和 Layout；不接收 L0C 输出的实现无需提供 |
| `operator()` | 每个 tile 的执行入口，Tensor、形状、坐标和偏移入参由具体场景决定 |

多 batch/group 场景可按需提供 `UpdateGlobalAddr()`、`UpdateNextProblem()`。专用多阶段后处理可以提供 `ReduceV`、`FinalizeSoftmax` 等业务方法，但仍须遵循统一的参数聚合和类组织规则。

## 接口废弃说明

以下接口在CANN 9.2.0版本标记废弃，将在2027年11月10日之后的版本删除：

- `BlockEpilogueFixpipe::Init(const Params&, const ProblemShape&)`，替换为 `Init(const Params&)`。
- `BlockEpilogueFmmWithScaleAdd::Init(const Params&, const ProblemShape&)`，替换为 `Init(const Params&)`。
- `BlockEpilogueMulsAdd::Init(const Params&, int64_t, int64_t, const ProblemShape&)`，替换为 `Init(const Params&)`。
- `BlockEpilogueSwigluMxQuant::GetL0c2UbTensor` 和 `GetConcatL0c2UbTensor`，替换为 `GetAccumulatorTensor(const AccumulatorTensorParams&)`。

9.2.0 分支没有的文件可直接调整接口，不增加兼容重载或废弃标签；两个 StreamK 文件的七参数 `Init` 暂保持不变。

## 公共约束

1. **参数组织**：初始化配置使用 `Params`，执行入参沿用具体场景的既有形式，公共偏移可使用 `OutputOffsets` 等聚合类型。
2. **计算位置**：通用 Epilogue 在 AIV 核执行；Empty 不执行计算。
3. **类组织**：成员函数先声明、后在类外实现；`public` 在前，`private` 在后；新增状态放入 `private`，`9.2.0` 已公开的成员保持原有可见性。
4. **接口顺序**：新接口集中放在 `public` 前部；既有兼容接口和数据成员放在其后，并添加 `Compatibility interfaces` 前缀的注释。
5. **源码兼容**：`9.2.0` 已有文件保留旧接口并转发到统一执行入口，默认实参只写在函数声明中；其余文件不增加旧式散参接口。
6. **独立包含**：头文件自行包含所需依赖，不能依赖其他文件的包含顺序。

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
BlockEpilogue epilogue;
epilogue.Init(params.epilogueParams);

// 仅从 L0C 接收输入的 Epilogue 需要获取累加结果 Tensor。
auto ubC = epilogue.GetAccumulatorTensor({baseM, baseN});
blockMmad(gmA, gmB, ubC, blockShape);

// 以接收 BlockShape 和 BlockCoord 的 Epilogue 为例。
epilogue(blockShape, blockCoord);
```
说明：`operator()` 的入参以具体 Epilogue 的定义为准；不从 L0C 接收输入时跳过 Tensor 获取步骤。

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
