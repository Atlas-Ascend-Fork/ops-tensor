# BlockEpilogueFinalizeRouting

## 功能说明

`BlockEpilogueFinalizeRouting` 是 GroupedMatmulFinalizeRouting 的 AIV 后处理组件，支持 MX 全量化、pertoken 全量化和 MX A8W4。每个 AIC tile 按 M 维拆给两个 AIV 子核处理。组件从 UB 读取 MMAD 结果，按模式执行反量化和 logit 加权，再根据 rowIndex 写回输出。确定性 pertoken 先写 workspace，再按输入行顺序归并。

## 模板参数

```cpp
template <typename DataTypeOut, typename DataTypeIn, typename DataTypeLogit,
          typename RowIndexType, typename DataTypeX2Scale = float,
          bool IsDeter = false, FinalizeRoutingMode Mode = FinalizeRoutingMode::MX>
class BlockEpilogueFinalizeRouting;
```

| 参数 | 约束 |
| --- | --- |
| `Mode` | `MX`、`PERTOKEN` 或 `A8W4`。仅传前四个模板参数时使用 MX 接口 |
| `DataTypeOut` | FP32 或 BF16。A8W4 要求 FP32 |
| `DataTypeIn` | MX 和 A8W4 要求 FP32。Pertoken 支持 INT32 或 FP32，须与 MMAD 的累加类型一致 |
| `DataTypeLogit` | FP32 或 BF16。A8W4 要求 FP32，pertoken 算子组装使用 FP32 |
| `RowIndexType` | MX 的 FP32 输出对应 INT64，BF16 输出对应 INT32。Pertoken 支持 INT32 或 INT64，A8W4 要求 INT64 |
| `DataTypeX2Scale` | Pertoken 每列 scale 的类型，支持 FP32 或 BF16 |
| `IsDeter` | 仅 pertoken 模式支持设为 `true` |

算子接口可施加更严格的类型组合约束。Pertoken 的逐 token scale 类型为 FP32，bias 类型为 BF16。

## Params

| 字段 | 适用模式 | 说明 |
| --- | --- | --- |
| `yGmAddr` | 全部 | 输出基址 |
| `x2ScaleGmAddr` | 全部 | Pertoken 必需的每列 ScaleB，形状为 `[E, N]`。MX 和 A8W4 的 Epilogue 不读取此字段 |
| `x1ScaleGmAddr` | 全部 | Pertoken 可选的逐 token ScaleA，形状为 `[M]`，空指针表示省略该项。MX 和 A8W4 的 Epilogue 不读取此字段 |
| `biasGmAddr` | 全部 | Pertoken 可选 bias，形状为 `[E, N]`。MX 和 A8W4 的 bias 由 MMAD 侧处理 |
| `logitGmAddr` | 全部 | 每个输入行对应一个 logit |
| `rowIndexGmAddr` | 全部 | 每个输入行对应一个输出行号 |
| `baseM`、`baseN` | MX | 原 MX 聚合参数的末尾两个字段，默认值均为 256 |
| `workspaceGmAddr` | Pertoken、A8W4 | 确定性 pertoken 的 workspace 基址 |
| `nSize` | Pertoken、A8W4 | 初始化时使用的完整输出列数 |

表格按聚合初始化顺序列出字段。输入地址、输出地址和 `Params` 对象须在组件使用期间有效。调用方负责初始化输出 y、校验 GM 容量和有效行号，并通过跨核通知保证 MMAD 结果就绪。

## 调用接口

所有模式通过 `Init(params)` 初始化组件，通过 `GetL0c2UbTensor(rows, cols)` 获取 Fixpipe 的 UB 目标。AIC 与 AIV 须使用相同的 UB 基址和行跨度。MX 也提供无参 `GetL0c2UbTensor()`，返回累加区的 LocalTensor。

- **MX**。每组调用 `UpdateNextProblem(problemShape)` 设置输出列数，并调用 `UpdateGlobalAddr(baseOffset)` 更新元数据地址。分组偏移使用 `baseOffset` 的 logit 坐标，同时推进 logit 和 rowIndex 的 GM 基址。随后调用 `operator()(blockShape, blockCoord)` 消费 tile，返回前等待该 tile 写回完成。
- **Pertoken 和 A8W4**。调用 `operator()(problemShape, blockShape, blockCoord, baseOffset, groupAccumM = 0)` 消费 tile。`problemShape` 的 N 为完整输出列数。`BlockCoord` 的坐标依次为输出列、bias、ScaleB、ScaleA、logit 和 rowIndex 的元素偏移。`baseOffset` 表示分组或分段的基址偏移，`blockCoord` 表示 tile 内寻址所用的偏移。A8W4 延后消费时传入待消费 tile 所属分组的偏移。

### A8W4 路由参数预取

按 tile 顺序调用 `PrefetchRoutingMetadata(rows, mOffset, groupMOffset)`，每个实际 tile 预取一次。预取使用原始 GM 基址，可跨组提前执行。生产和消费各自推进双缓冲索引。没有行可处理的 AIV 也须执行对应的消费调用，释放槽位。Kernel 须在组件析构前消费最后一个已预取的 tile。

### 确定性 pertoken

`groupAccumM` 表示当前分段在 workspace 窗口内的起始行。组件先将逐输入行的贡献线性写入 workspace。Kernel 通过 `PrefetchRowIndicesFromBase` 和 `GetPrefetchedRowIndex` 读取窗口内的行号，再按既定顺序调用 `DeterministicFlushRow(mOffset, outRow, nSize)`，保证同一输出行按输入行顺序累加。

## 计算与写回

- **MX**。对 MMAD 已应用 MX scale 和 bias 的 FP32 结果执行 logit 加权。
- **Pertoken**。将 INT32 累加值转换为 FP32，FP32 累加值直接使用。依次乘 ScaleB、可选 ScaleA，加可选 bias，再乘 logit。
- **A8W4**。将 MMAD 结果乘 64，再乘 logit。对当前 AIV 负责的整块结果原地加权，写回后等待搬出完成，随后权重转换可复用该 UB 区域。

FP32 输出使用 FP32 乘法和原子加。BF16 输出先将计算结果和 logit 转为 BF16，再执行 BF16 乘法和原子加。非确定性模式将每行贡献加到 `y[rowIndex, nOffset:nOffset + cols]`。MX 和 pertoken 要求有效行号，A8W4 跳过负行号。

MX 和 pertoken 的 UB 累加区按 M≤128、N≤256 规划，输出分段最多 32 行并使用双缓冲。Pertoken 的累加行跨度按 256 B 对齐，MX 和 A8W4 的累加行跨度按 32 个元素对齐。A8W4 的累加和输出区域从 80 KiB 开始，rowIndex 和 logit 的双缓冲分别从 242 KiB 和 244 KiB 开始。调用方须遵循配套 Kernel 的 UB 复用和同步约定。

## 相关文件

- [头文件](../../../../include/blaze/epilogue/block/block_epilogue_finalize_routing.h)
- [GroupedMatmulFinalizeRouting Kernel 使用指南](../../gemm/kernel/grouped_matmul_finalize_routing.md)
- [MX 全量化 Kernel](../../gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.md)
- [Pertoken 全量化 Kernel](../../gemm/kernel/kernel_qgmm_mix_pertoken_finalize_routing.md)
