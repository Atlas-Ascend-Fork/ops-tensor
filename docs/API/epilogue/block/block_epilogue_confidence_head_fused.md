# Block Epilogue ConfidenceHeadFused

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_confidence_head_fused.h)

## 功能说明

`BlockEpilogueConfidenceHeadFused` 管理 FP32 UB 缓冲、温度读取、零 K 的输入准备和 GM 写回。非空 K 路径接收已包含 bias 的 Cube 结果，调用 [ConfidenceHeadFused::Run](../tile/arch35/confidence_head_fused.md) 执行后处理。

## 特殊约束

- InType_、OutType_ 均为 float，输入使用 UB NDExt 布局。
- problemShape 为 `(1,B*N,K1,K2)`，baseN 与 Cube Block、调度器一致。
- 带 bias 时，tile 不跨 bias[N] 的广播边界。
- 每个 UB 槽保存两行物理累加器和一行输出，双缓冲占用 `2*3*baseN*sizeof(float)` 字节。

## 模板参数

```cpp
template <typename InType_, typename OutType_>
class BlockEpilogueConfidenceHeadFused;
```

## Params 参数结构

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| outputGmAddr | GM_ADDR | FP32 输出地址。 |
| temperatureGmAddr | GM_ADDR | 可选 FP32 [1] 温度；nullptr 或值为正负零时按 1 计算。 |
| biasGmAddr | GM_ADDR | 零 K 路径使用的可选 FP32 [N] bias。 |
| lowLimit / highLimit | float | 指数输入的截断阈值。 |
| biasN | uint32_t | 每个 batch 的 bias 长度。 |
| hasBias / hasLowLimit / hasHighLimit | bool | bias 与两侧截断的启用标识。 |
| problemShape | ProblemShape | 完整问题 shape。 |
| baseN | uint32_t | UB 槽的 N 容量，默认 128。 |

## 成员方法

```cpp
epilogue.Init(params);
epilogue(ubTensor, blockShape, blockCoord);
epilogue.template operator()<true>(ubTensor, blockShape, blockCoord); // K1+K2=0
```

`blockShape` 为 `(1,curN,0,1)`，`blockCoord` 的 N 分量为展平输出坐标。`GetUbAddr(dbIdx)` 返回 Cube 写入 UB 双缓冲的地址，dbIdx 取 0 或 1。

零 K 且带 bias 时，将 `bias[coordN % biasN : coordN % biasN + curN]` 搬入输入槽作为 logits；无 bias 时由 Tile 使用零输入。MTE2/V 事件保护输入搬运，V/MTE3 事件保护输出写回。Cube 与 Vector 的跨核握手由 Kernel 管理。
