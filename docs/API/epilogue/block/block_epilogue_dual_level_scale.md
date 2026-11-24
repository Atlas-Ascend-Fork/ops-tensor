# BlockEpilogueDualLevelScale

[头文件](../../../../include/blaze/epilogue/block/block_epilogue_dual_level_scale.h)

## 功能与约束

`Blaze::Epilogue::Block::BlockEpilogueDualLevelScale<Output_>` 在 Ascend 950 的双 AIV 上消费 MXFP4 分段矩阵乘的 FP32 partial。每个 K 段最多包含 512 个元素；两侧 level0 scale 的乘积以 FP32 计算，再通过 `MulAddDst` 累加 partial。最后一个 K 段完成后加入可选 FP32 bias，并按 RINT 转换为 FP16 或 BF16。

组件按初始化时传入的布局接收 split-M 部分和。当前 GMM 的部分和基本块上限为 128×128，沿 M 分给两路 AIV。Kernel 负责选择参加计算的矩阵及其 GM 片段，算子 Host 校验 shape、dtype 和容量。

## 初始化参数

GMM 调用方通过 `Kernel::Params::epilogueParams` 提供两路 level0 scale 的 GM 地址，通过 `mmadParams` 提供 bias 和输出地址。Kernel 使用这些地址和块配置一次构造公开的 `BlockEpilogueDualLevelScale::Params`，再调用 `Init`。

- `scaleA`：`__gm__ float*`。每个矩阵的 level0 scale 为连续 ND `(M,segmentCount)`。
- `scaleB`：`__gm__ float*`。每个矩阵的 level0 scale 为连续 ND `(segmentCount,N)`。
- `bias`：`__gm__ float*`。当前矩阵使用连续 ND `(N,)`。无 bias 时传入空指针，并设置 `hasBias=false`。
- `output`：`__gm__ Output_*`。每个输出矩阵为连续 ND `(M,N)`，`Output_` 为 FP16 或 BF16。
- `n`：由 Kernel 填写，等于 B 的 level0 scale 和输出矩阵的行步长，单位为元素。
- `baseM`、`baseN`：由 Kernel 填写，分别为调度器基本块的最大行数和列数。

- `segmentCount`：由 Kernel 根据逻辑 K 和 level0 段长计算，计入尾段。

`Init<PartialRows, PartialCols, RowAlignment, PartialBufferCount>(const Params&)` 在 AIC 和 AIV 上调用。行列上限和 split-M 行对齐来自 BlockMmad，部分和缓冲区数量由 Kernel 指定。AIC 初始化 UB 视图使用的布局，AIV 还负责绑定 GM Tensor、计算 UB 区域并初始化流水事件。调用方须保证 UB 容量不少于 `PartialBufferCount * partialBytes + mTiles * nTiles * partialBytes + mTiles * rowsPerAiv * 32 * sizeof(float) + 2 * baseN * sizeof(float)`。

其中，`mTiles = ceil(baseM / PartialRows)`，`nTiles = ceil(baseN / PartialCols)`，`rowsPerAiv = PartialRows / 2`，`partialBytes = rowsPerAiv * PartialCols * sizeof(float)`。容量按单个 AIV 计算。

```cpp
Kernel::Params params{};
params.epilogueParams.level0ScaleAGmAddr = a0;
params.epilogueParams.level0ScaleBGmAddr = b0;
params.mmadParams.cGmAddr = y;
params.mmadParams.biasGmAddr = bias; // 无 bias 时传入空指针。
```

## 输出块偏移

`operator()(shape, scaleAOffset, scaleBOffset, biasOffset, outputOffset, hasBias, channel)` 接收相对于上述 GM Tensor 起点的四个元素偏移。

- `scaleAOffset` 指向块首行的第一个 K 段，后续行按 K 段数跨步。
- `scaleBOffset` 指向第一个 K 段的块首列，后续段按 `n` 跨步。
- `biasOffset` 指向块首列，`hasBias=false` 时不读取。
- `outputOffset` 指向输出块左上角，后续行按 `n` 跨步。

GMM Kernel 按专家和全局行列计算这四个偏移。QBMM 接入时可按 A、B、bias 和输出各自的 batch 基址计算偏移，广播 Tensor 可复用同一基址。

## Partial 通道与同步

Epilogue 通过 `GetAccumulatorTensor(const AccumulatorTensorParams&) const` 提供 FP32 partial 接收 Tensor。参数使用公共的 `AccumulatorTensorParams`，`rows/cols` 须在初始化时传入的基本块上限内，`slot` 为 0 或 1，`batch` 不参与寻址。列跨度和行对齐取自 `Init`，AIC 须在初始化后获取视图。

`DualLevelPartialChannel` 是 `kernel_qgmm_mx.h` 中 `GemmUniversal` 的私有嵌套类，只管理 ready/credit 同步和连续序号，不提供 UB Tensor。缓冲区编号由连续序号对 2 取模确定，不在输出块或专家边界重置。BlockMmad 和 Epilogue 通过模板参数接收通道，不依赖具体 Kernel 类型。

Kernel 将 BlockMmad 的布局和 `DOUBLE_BUFFER_COUNT` 传给 Epilogue 初始化。Kernel 的视图数组和 Channel 同样使用 `DOUBLE_BUFFER_COUNT`，并在编译期检查数组长度和可用的跨核同步标志。Epilogue 根据传入的数量计算部分和及后续 UB 区域的位置。Host Tiling 独立计算 UB 容量。当前 GMM 使用 Kernel 私有的 `DualLevelPartialTiles` 遍历微块。

Kernel 从 Epilogue 获取两个完整微块的接收 Tensor，向 BlockMmad 的 `operator()<PartialTiles_>` 传入逻辑 K 长度和完整 K 段长度，在消费端向 Epilogue 的 `operator()<PartialTiles_>` 传入 GM 偏移、bias 标志和通道。双方共享 K 分段描述与 M/N 微块遍历器类型，按 K 段、N 微块、M 微块的顺序交接。

BlockMmad 保留完整 K 遍历和 L1/scale 缓存，只在 AIC 执行；每个微块选择当前缓冲区的 Tensor，按有效行数切片，取得 credit 后通过 split-M Fixpipe 写入并通知两个 AIV。Epilogue 的 `operator()` 仅在 AIV 执行，等待 ready、读取 partial 后归还 credit；零行 AIV 也完成握手。Kernel 负责通道的初始 credit 和最终 drain。

`PartialTiles_` 支持 `(rows,cols)` 构造，提供 `Tile{m,n,rows,cols}` 和 `Next(Tile&)`。消费端通过 `Channel_` 调用 `BufferIndex/WaitReady/Release`。`Release` 归还 credit 后推进本地通道序号。

```cpp
constexpr auto rows = BlockMmad::DUAL_LEVEL_L0_BASE_M;
constexpr auto cols = BlockMmad::DUAL_LEVEL_L0_BASE_N;
using PartialView = decltype(epilogue.GetAccumulatorTensor({rows, cols, 0}));
const PartialView partialBuffers[]{
    epilogue.GetAccumulatorTensor({rows, cols, 0}),
    epilogue.GetAccumulatorTensor({rows, cols, 1}),
};
// Kernel 将缓冲区视图数组和通道传给 BlockMmad。
```

## 执行顺序

1. Kernel 在 AIV 调用 `channel.InitConsumer()`，随后在 AIC 和 AIV 调用 `epilogue.Init<PartialRows, PartialCols, RowAlignment, PartialBufferCount>(epilogueParams)`。
2. Kernel 调度输出块。GMM 的 AIC 调用 `mmad.operator()<PartialTiles_>`，AIV 计算四个 GM 偏移并调用 `epilogue.operator()<PartialTiles_>`。
3. `operator()` 内部等待前一块输出 DMA，逐段加载 level0 scale、消费 partial 并累加；最后一段加 bias、转换并写回。块内的 scale 搬运和 DMA/Vector 事件均由 epilogue 自己管理。
4. 所有块处理完成后，已初始化的 AIV 在 Epilogue 对象析构时回收本地事件，AIC 调用 `channel.DrainProducer()` 收回所有缓冲区 credit。

`operator()` 将遍历器返回的微块直接交给 `Consume`，每个 K 段结束后归还 scale 缓冲，每个输出块结束后记录写回事件。

同一物理核的 AIC 和两个 AIV 必须使用一致的块、K 段、微块遍历顺序和缓冲区编号。scale、bias 和累加器区域不可被其他并行组件复用。GM 偏移使用 `int64_t`；VF 的两组列寄存器分别覆盖 `[0,64)` 和 `[64,128)`，输出 DMA 只写有效元素。
