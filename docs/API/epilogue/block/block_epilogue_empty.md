# Block Epilogue Empty
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_empty.h)

## 功能说明
空后处理组件，用于不支持后处理操作的矩阵乘 Kernel。作为 `BlockEpilogue` 的默认实现，满足 Kernel 模板参数要求但不执行任何实际计算。

**继承自**：[Block Epilogue 基础框架](./block_epilogue.md)

## 特殊约束

### 用途限制
仅用于不支持后处理的 Kernel，作为模板参数占位。

### Kernel 兼容性
仅支持 Basic Kernel：
```
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

不支持 StreamK Kernel（StreamK Kernel 需要 `BlockEpilogueMatmulStreamK`）。

### 无实际计算
所有成员函数均为空实现，不执行任何操作：
- `Run()`：直接返回
- `operator()`：直接返回或调用 `Run()`（空）

### 参数兼容
提供多种调用接口以匹配 Kernel 模板要求，但参数均无实际用途。

## 特殊数据结构

### Params
```
struct Params {
    Params() = default;
};
```
说明：空参数结构体，无任何成员。

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockEpilogueEmpty()
```
功能：构造 BlockEpilogueEmpty 对象（空实现）。

### Run函数
```
__aicore__ inline void Run()
{
    return;  // 直接返回，不执行任何操作
}
```
功能：执行后处理操作（空实现）。

### operator函数（参数版本）
```
__aicore__ inline void operator()(Params const& params)
{
    Run();  // 调用空的 Run()
}
```
功能：执行后处理操作（参数版本）。
说明：参数 `params` 无实际用途，仅满足接口要求。

### operator函数（Block版本）
```
__aicore__ inline void operator()(
    BlockShape const& blockShape, BlockCoord const& blockCoord,
    int64_t dstStartOffset = 0, int64_t srcStartOffset = 0)
{
    return;  // 直接返回，不执行任何操作
}
```
功能：执行后处理操作（Block 坐标版本）。
说明：所有参数均无实际用途，仅满足接口要求。

## 调用示例

### 在 Kernel 模板中使用
```
// 定义 BlockEpilogue
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

// 组装 Kernel
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 实例化与调用（可选）
```
BlockEpilogue epilogue;

// 以下调用均无实际效果
epilogue.Run();
epilogue({});
epilogue(blockShape, blockCoord, 0, 0);
```

## 设计说明

### 为什么需要 BlockEpilogueEmpty
1. **模板参数要求**：`GemmUniversal` 模板需要 `BlockEpilogue` 参数
2. **接口一致性**：保持与其他 Epilogue 组件（如 StreamK）相同的接口
3. **扩展性**：未来可替换为实际的后处理组件（如 ReLU、Add 等）
4. **零开销**：空实现不会引入额外计算开销

### 性能影响
| 影响项 | 说明 |
|------|------|
| 编译时间 | 无影响（空类编译开销极小） |
| 运行时间 | 无影响（空函数直接返回） |
| 内存占用 | 无影响（无成员变量） |
| 流水线 | 无影响（不参与计算流程） |

### 与 BlockEpilogueMatmulStreamK 的关系
```
BlockEpilogueEmpty         ← 当前实现（空，用于 Basic Kernel）
    ↓
BlockEpilogueMatmulStreamK ← 实际实现（用于 StreamK Kernel）
    ├── workspace 汇聚（K 轴切分累加）
    ├── 类型转换（float → half/bf16）
    └── ReLU 激活（可选）
```

## 扩展建议

如需添加后处理功能，可参考以下设计模式：
```
// 自定义 Epilogue 示例（伪代码）
class BlockEpilogueRelu {
public:
    struct Params {
        float threshold;  // ReLU 参数
    };

    __aicore__ inline void Init(Params const& args) {
        threshold_ = args.threshold;
    }

    __aicore__ inline void Run() {
        AscendC::Relu(outputTensor, inputTensor, threshold_);
    }

    __aicore__ inline void operator()(Params const& args) {
        Init(args);
        Run();
    }

private:
    float threshold_;
    AscendC::GlobalTensor<float> outputTensor_;
    AscendC::GlobalTensor<float> inputTensor_;
};
```

## 适用场景

- **Basic Kernel**：不需要后处理的简单矩阵乘场景
- **无 workspace**：不需要存储中间结果
- **单核计算**：仅需 AIC 计算，无需 AIV 参与
- **原型开发**：快速验证 Kernel 框架，后续可替换为实际实现
