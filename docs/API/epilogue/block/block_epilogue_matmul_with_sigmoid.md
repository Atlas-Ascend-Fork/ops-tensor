# Block Epilogue Matmul With Sigmoid

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_matmul_with_sigmoid.h)

## 功能说明

FusedMatMul sigmoid 场景的向量后处理 Block，在 AIV 上读取 AIC 经 Fixpipe 搬入 UB 的 FP32 累加结果，执行 sigmoid、类型转换和输出写回：

$$ y = \frac{1}{1 + \exp(-z)}, \quad z = x1 @ x2 + bias $$

bias 可选，在上游 BlockMmad 的首轮 K 累加时加入；本组件接收的累加结果已经包含 bias。先完成矩阵乘和 bias 相加，再执行 sigmoid。

**关联框架**：[Block Epilogue 基础框架](./block_epilogue.md)

**上游计算**：[BlockMmadMatmulFixpipeOpti](../../gemm/block/block_mmad_matmul_fixpipe_opti.md)

## 特殊约束

### 数据类型

- 累加器和 sigmoid 向量计算使用 FP32。
- FusedMatMul 的输出 `ElementType_` 为 FP16 或 BF16，转换采用 round-to-nearest-even。

### splitM 子块

一个 AIC 对应两个 AIV，按 M 轴分配有效行。M 为奇数时，第一个 AIV 多处理一行；M=1 时，另一个 AIV 不计算有效数据，但仍完成 ready/free 同步。

### UB 复用

FP32 累加器放在 UB 起始位置，输出缓冲区放在实际累加器数据之后。累加器的 N 轴按 8 个 FP32 元素对齐，FP16/BF16 输出的 N 轴按 16 个元素对齐，均满足 32 字节对齐要求。剩余 UB 空间不足以一次处理所有行时，按 M 轴分段计算和写回。

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| DispatchPolicy | BlockMmad 调度策略 |
| L0CDataType | 累加类型，固定为 float |
| OutputType | 输出类型 |
| Tile::Sigmoid | [Tile 级 sigmoid](../tile/arch35/sigmoid.md)，对 FP32 输入执行激活和输出转换 |
| BlockCoord | 当前 tile 的 (M, N, K, Batch) 全局坐标 |
| BlockShape / ProblemShape | 当前 Block / 完整问题的 (M, N, K, Batch) 形状 |

## 特殊数据结构

### Params

```cpp
struct Params {
    GM_ADDR outputGmAddr{nullptr};
    ProblemShape problemShape{};
    bool splitM{false};
};
```

## 特殊成员方法

### Init函数

```cpp
__aicore__ inline void Init(const Params& params)
```

从 `params` 读取输出 GM 地址、完整 `problemShape` 和 `splitM` 配置，用于由 tile 坐标计算输出地址。入口组装时填写输出地址、`problemShape` 和 `splitM`，公共 Kernel 直接调用 `Init(params.epilogueParams)`。

### operator函数

```cpp
template <typename TensorC>
__aicore__ inline void operator()(TensorC& ubTensor, const BlockShape& blockShape,
                                const BlockCoord& blockCoord)
```

每次调用处理一个实际 Fixpipe tile；M/N 长度取自 `blockShape`，全局位置取自 `blockCoord`，无需传入 baseM/baseN 或输出偏移。

1. 根据初始化保存的 `splitM` 计算当前 AIV 的有效行数与行偏移。
2. 以 `((batch * M + coordM + localRowOffset) * N + coordN)` 计算输出偏移，并等待 AIC 当前 tile 就绪。
3. 将当前分段的输入、输出 UB Tensor 交给 `Tile::Sigmoid::run`；Tile 从输入 Tensor 的 shape 获取有效行列数，分别从输入、输出 layout 读取行跨度，以 FP32 执行 sigmoid 和输出转换。
4. 转换为输出类型，写回 GM，并通知 AIC 当前缓冲区可复用。

## 数据流

```text
AIC: x1 @ x2 + optional bias → FP32 L0C → Fixpipe → FP32 UB
AIV: FP32 UB → sigmoid → Cast(FP16/BF16) → GM
```

## 适用场景

arch35 上 FusedMatMul 的二维 FP16/BF16 矩阵乘与 sigmoid 融合后处理。组装方式见 [Kernel Matmul With Epilogue](../../gemm/kernel/kernel_matmul_with_epilogue.md)。
