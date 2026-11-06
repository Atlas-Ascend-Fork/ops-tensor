# ProcessSharedInput

## 功能

`Blaze::Epilogue::Tile::ProcessSharedInput<OutType,DoubleBuffered>` 将 BF16 shared input 从 GM 搬入 UB，完成加权处理后写入输出 GM。该 arch35 AIV 组件不处理分组、rowIndex、全局核间划分或输出区清零，调用方负责这些工作。

OutType 为 FP32 或 BF16，DoubleBuffered 默认为 true。

## 接口

通过 `#include "blaze/epilogue/tile/compute.h"` 引入组件。

```cpp
struct Buffers {
    uint64_t inputPing, inputPong, outputPing, outputPong, chunkElements;
};
template <class InputTensor, class OutputTensor>
static void Run(const InputTensor& input, const OutputTensor& output,
                float weight, const Buffers& buffers);
```

`Buffers` 的前四项是 UB 字节偏移，`chunkElements` 是每次处理的最大元素数。调用方传入已定位到当前 AIV 分片的一维连续 GM Tensor，并在构造输出 Tensor 时计入 shared input 窗口偏移。处理元素数从输入 Tensor 的 shape 获取。

## 约束与行为

- 输入 Tensor 元素类型为 BF16，输出 Tensor 元素类型为 `OutType`。输出 Tensor 的有效元素数至少等于输入 Tensor 的元素数。空输入不触发搬运。
- chunkElements 必须大于 0；每个 input 槽容量至少为 chunkElements×sizeof(BF16)，每个 output 槽容量至少为 chunkElements×sizeof(OutType)。
- 使用中的 input/output 槽不能重叠。DoubleBuffered=true 使用 BufferSlot 0/1 的本地锁；false 使用事件 0 的串行拷贝、计算和回写。
- 在 AIC 上调用直接返回，在 AIV 上按 chunk 顺序处理。该接口不发出跨核结束通知。
- FP32 输出将 BF16 输入转换为 FP32，再乘 FP32 weight。BF16 输出将 weight 按 RINT 转成 BF16，并使用 BF16 乘法。
- 本接口写入输出初值，不执行原子加；后续路由写回前，调用方须保证所有核的初始化均已完成。

[实现](../../../../../include/blaze/epilogue/tile/arch35/process_shared_input.h)；[GMMFR 使用指南](../../../gemm/kernel/grouped_matmul_finalize_routing.md)。
