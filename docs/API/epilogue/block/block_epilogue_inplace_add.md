# BlockEpilogue Inplace Add

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_inplace_add.h)

## 功能说明

`Blaze::Epilogue::Block::BlockEpilogueInplaceAdd` 在 AIV 上将 AIC 经 Fixpipe 写入 UB 的
FP32 结果与 GM 中的旧 C 相加，并写回同一 GM 地址：`C = C + result`。
旧 C 同时作为输入和输出，调用前必须初始化。不执行类型转换、Bias 或激活，不需要 GM workspace。

该组件由 [QGMM MX Kernel](../../gemm/kernel/kernel_qgmm_mx_basic.md) 使用。
Kernel 对每组按 K 长度选择路径：`2048 <= groupK <= 16384` 使用 UB 后处理，
其余非空组使用 Fixpipe AtomicAdd；空组不修改输出。此选择属于 Kernel 的调度逻辑。

## 使用约束

- UB 路径要求 9201/9202 架构、`KERNEL_TYPE_MIX_AIC_1_1`，一个 AIC 配对一个 AIV。
  AIC/AIV 必须执行相同的 group、tile 和 panel 遍历，不能在 1:2 MIX 中调用。
- `ubResult` 为 UB 中的 FP32 Tensor，`gmC` 为 GM 中的 FP32 ND Tensor，二者形状对应当前面板。
- 当前 QGMM 组装要求 A 为 FP8 E4M3、B 为 FP4 E2M1 或 HiFloat4；A/B/C 布局为 DN/ND/ND，
  `isBias=0`、`singleW=1`，沿 K 分组。N 和 `baseN` 为 64 的倍数；完整约束见 Kernel 文档。
- UB 从字节偏移 0 开始由该组件规划，不得与同时存活的其他 UB 数据重叠。
- 所有面板必须装入规划容量，`0 < m <= stageM`、`0 < n <= baseN`。
  Kernel 从 `GetAccumulatorTensor` 返回的 Tensor shape 获取实际面板行数。

## 参数

```cpp
struct Params {
    int64_t baseM{128};
    int64_t baseN{128};
    int64_t ubPanelM{0};
};
```

| 字段 | 说明 |
|------|------|
| `baseM` | 最大 tile 行数，必须为正并与 Kernel tiling 一致 |
| `baseN` | 最大 tile 列数，必须为正并与 Kernel tiling 一致 |
| `ubPanelM` | 结果面板行数，0 为自动规划；显式值不得超过 `baseM`，并须留出旧 C 缓冲 |

Kernel 直接使用 `params.epilogueParams` 初始化该组件，面板策略不属于公共 `GMMTiling`。

## 接口

```cpp
__aicore__ inline void Init(const Params& params);
__aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;
template <typename TensorUbC, typename TensorC>
__aicore__ inline void operator()(const TensorUbC& ubResult, const TensorC& gmC,
                                  int64_t m, int64_t n, uint64_t panelIndex = 0);
__aicore__ inline ~BlockEpilogueInplaceAdd();
```

- `Init`：根据 `AscendC::TOTAL_UB_SIZE` 规划 UB，在 AIV 上为两个结果槽位各发送一次初始 ACK。
  旧 C 缓冲使用 mutex，无需预置空闲 token。AIC/AIV 应使用相同参数初始化。
- `GetAccumulatorTensor`：使用统一的 `AccumulatorTensorParams`，返回槽位 `params.slot & 1` 的 UB Tensor；
  行数为 `min(params.rows, stageM)`，行跨度为 `params.cols` 向上对齐至 64 个 FP32 元素。
  调用方传入槽位 0/1；面板索引在调用处取低位转换为槽位。此接口返回二维 Tensor，`batch` 保持默认值 1。AIC 用它接收 Fixpipe 结果，AIV 用它读取结果。
- `operator()`：仅在 AIV 调用。预取最多两个旧 C 分块，等待结果 READY，分块执行 FP32 Add 并写回 GM，
  完成向量读取后发送 ACK。面板序号在 UB 路径的 tile/group 间连续推进，不能逐 tile 重置。
- 析构函数：AIC 等待两个槽位最终 ACK，AIV 不添加额外的空锁。
  已初始化但无 UB 计算的配对核也会执行收尾，以配平初始化信号；未初始化的对象直接返回。
  每个对象仅初始化并执行一次，不应复制或在析构前重新初始化。

非法尺寸或不足以容纳四个最小缓冲时，规划得到 `stageM=0`，`Init` 和析构函数直接返回；
Kernel 会跳过零行面板。调用方必须保证参数合法，这不是自动切换到其他路径的机制。

## UB 布局与容量规划

UB 布局为 `[result 0][result 1][old C 0][old C 1]`，两类缓冲互不重叠。
设 `rowBytes = align_up(baseN, 64) * sizeof(float)`，`totalRows = TOTAL_UB_SIZE / rowBytes`，
容量必须满足 `2 * stageM + 2 * chunkRows <= totalRows`，每个缓冲至少一行。

自动规划先为每个旧 C 缓冲预留最多 16 行。能容纳完整 tile 时使用 `stageM=baseM`；
否则优先选择向上对齐至 16 行的半 tile，容量不足时按可用容量向下对齐，低于 16 行时使用实际容量。
显式 `ubPanelM` 覆盖自动面板行数，但必须小于 `totalRows / 2`。
剩余容量分配给旧 C，`chunkRows` 不超过结果面板行数或 64 行。

## 同步协议

| 方向 | 信号与流水 | 含义 |
|------|------------|------|
| AIC → AIV | READY：`FIXPIPE_AIC_READY_FLAG_BASE + slot`（6/7），`PIPE_FIX` → `PIPE_V` | Fixpipe 结果可读 |
| AIV → AIC | ACK：`FIXPIPE_AIV_ACK_FLAG_BASE + slot`（4/5），`PIPE_V` → `PIPE_FIX` | 结果槽位可复用 |
| AIV 内部 | 同一 mutex 的 `MTE3 → MTE2` | 上次旧 C 写回完成，可覆盖分块缓冲 |
| AIV 内部 | 同一 mutex 的 `MTE2 → V` | 旧 C 搬入完成，可执行 Add |
| AIV 内部 | 同一 mutex 的 `V → MTE3` | Add 完成，可写回 GM |

跨核同步使用 `SYNC_MODE_INTRA`。旧 C 缓冲通过 `BufferSlot` 的 `asc_lock/asc_unlock` 管理，
mutex ID 为 0/1；调用方不得与同时存活的其他同步协议冲突使用这些 mutex。
两个结果槽位与两个旧 C 缓冲分别轮转；
旧 C 缓冲的相位跨面板保留，以处理奇数分块。Add 结果存入旧 C 缓冲，因此 ACK 不必等待 MTE3；
析构函数仅在 AIC 侧等待两个结果槽位的 ACK，保留跨核同步的收尾；Kernel 不再显式调用 `Finish`。
AIV 在循环内通过实际读写位置的锁管理缓冲复用，退出前不再添加空的 MTE2 锁；最后的 GM 写回由正常 kernel 结束机制保证。
该析构函数不提供返回时 GM 写回已经完成的独立同步保证。

组装方负责 include 具体 Epilogue，再 include Kernel。Kernel 通过 `BlockEpilogueEmpty` 区分空尾处理与 UB 累加路径；
非空 Epilogue 需要提供本 Kernel 使用的 `Params`、`Init`、`GetAccumulatorTensor`、调用运算符接口，并自行在析构时完成同步收尾。

## 组装示例

```cpp
#include "blaze/epilogue/block/block_epilogue_inplace_add.h"

// InplaceKernel 为使用此 Epilogue 的 QGMM MX GemmUniversal 实例。
InplaceKernel::Params params{};
// 填写 problemShape、mmadParams、groupListGmAddr 和 gmmParams 后：
params.epilogueParams = {params.gmmParams.baseM, params.gmmParams.baseN, 0};
// 在 __global__ __aicore__ 入口中指定：
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
InplaceKernel kernel;
kernel(params);
```
