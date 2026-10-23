# Tile Swish

> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/swish.h)

`Tile::Swish` 接收 FP32 UB Tensor，计算 `x / (1 + exp(-β*x))` 后以 BF16 RINT 写回。
β 默认为 1，此时等价于 SiLU。计算顺序为 FP32 `Muls(x, -β)` → `Exp` → `Adds` → `Div` → BF16 `Cast`。
每行尾部补零到 32 元素边界，供 MX 量化使用。

## 接口

```cpp
class Swish {
public:
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor, float beta = 1.0F);
};
```

| 参数 | 约定 |
| --- | --- |
| `beta` | 所有元素共用的有限 FP32 标量，默认 1 |
| `srcTensor` / `dstTensor` | UB、NDExt，逻辑形状相同，元素类型分别为 `float` / `bfloat16_t`，基址 32B 对齐且缓冲不重叠 |

元素类型从 Tensor 推导，并在编译期校验。
行数 `mSize` 和每行有效元素数 `nSize` 从 `srcTensor.layout()` 获取；两个 Tensor 的行距均须为 `Align32(nSize)` 个元素。

当前仅支持标量 β。

目标缓冲须容纳 `mSize * Align32(nSize)` 个 BF16 元素。
`nSize` 为 32 的倍数时合并所有行计算，否则逐行处理。由于 `LoadAlign` 每次读取 64 个 FP32 元素，
当 `mSize > 0` 且 `nSize > 0` 时，源基址之后须至少有下列数量的 FP32 元素可读：

| 条件 | 可读元素数 |
| --- | --- |
| `nSize % 32 == 0` | `Align64(mSize * nSize)` |
| 其他 | `(mSize - 1) * Align32(nSize) + Align64(nSize)` |

## 使用

通过 `blaze/epilogue/tile/compute.h` 引入：

```cpp
Blaze::Epilogue::Tile::Swish::Run(srcTensor, dstTensor, beta);
```

Tile 只负责计算；调用方负责 UB 缓冲、同步和 GM 写回。
[BlockEpilogueSwishMxQuant](../../block/block_epilogue_swish_mx_quant.md) 在 Tile 返回后通知 AIC 复用 FP32 输入区，再执行 MX 量化。
