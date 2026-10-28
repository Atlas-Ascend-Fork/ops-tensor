# Sigmoid（Tile 级 Sigmoid 激活）
> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/sigmoid.h)

## 功能说明

`Blaze::Epilogue::Tile::Sigmoid` 对 UB Tensor 逐元素计算 `1 / (1 + exp(-x))`。
通过 [compute.h](../compute.md) 引入，与 [Gelu](./gelu.md) 同属可复用的 Tile 计算组件。
Tile 仅负责向量计算；UB 分配、跨核同步和 GM 输出由 Block 层管理。

## 接口

```cpp
class Sigmoid {
public:
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor);
};
```

- 输入、输出 dtype 分别从源、目的 Tensor 推导，无需显式指定类模板参数。输入仅支持 `float`；输出支持 `float`、`half`、`bfloat16_t`。
- 源、目的 Tensor 必须位于 UB，采用 NDExt 排布；入口以 `static_assert` 校验 dtype、位置和布局。
- 输入、输出 Tensor 的逻辑 shape 必须一致。有效行数、列数直接从输入 layout 的 `asc::te::get<0, 1>(shape)`、`asc::te::get<1, 1>(shape)` 获取，对应 `MakeNDExtLayout` 的 `((1, rows), (1, cols))`；无需单独传入尺寸。
- 源、目的行步长分别从各自 layout 的 `asc::te::get<0, 1>(stride)` 读取，允许不同 pitch。
- AIC 路径直接返回；AIV 提取 Tensor 地址后调用 VF，不分配临时 UB。

## 计算与精度

VF 按行和向量长度遍历有效元素，以尾部 mask 限制最后一组。
取负、指数、加一及除法均以 FP32 计算；除法保留 `preciseDiv=true`。
FP32 输出直接存储；FP16/BF16 输出以 `NO_SAT`、`CAST_RINT` 转换后存储。

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/utils/layout_utils.h"

auto srcLayout = Blaze::Gemm::MakeNDExtLayout(3L, 17L, 24L);
auto dstLayout = Blaze::Gemm::MakeNDExtLayout(3L, 17L, 32L);
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, float>(0), srcLayout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(1024), dstLayout);
Blaze::Epilogue::Tile::Sigmoid::Run(srcTensor, dstTensor);
```

调用方先保证输入就绪，计算结束后安排后续同步和输出。
现有使用方为 [BlockEpilogueMatmulWithSigmoid](../../block/block_epilogue_matmul_with_sigmoid.md)。
