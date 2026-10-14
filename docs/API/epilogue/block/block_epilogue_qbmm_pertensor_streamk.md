# Block Epilogue QBMM Per-tensor StreamK
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h)

## 功能说明

QBMM per-tensor StreamK 的专用 AIV 后处理模板。它拥有完整的 vector 后处理流程：

1. 从 workspace 读取同一输出 tile 的多个 split-K raw partial；
2. 在 UB 中完成 K 分片归约；
3. 应用 X2 per-tensor scale；
4. 当可选输入地址存在时应用一个 X1 per-tensor 标量 scale；
5. 应用需要在反量化后相加的 bias；
6. 转换为输出类型并写回 C GM。

该模板不复用通用 Matmul StreamK epilogue，也不复用非 StreamK dequant epilogue；
但 scale/bias 数值计算链与它们复用同一个
[`Tile::Dequant<B32ToFpTag>`](../tile/arch35/dequant.md)，本模板只额外拥有 workspace 归约与
标量预合并（InitDequant）逻辑。

## 模板参数

```cpp
template <
    class WorkspaceType,
    class OutType,
    class DispatchPolicy,
    class X2ScaleType = float,
    class X1ScaleType = float>
class BlockEpilogueQbmmPertensorStreamK;
```

| 参数 | 说明 |
|------|------|
| `WorkspaceType` | raw partial 类型；int8 通路为 int32，浮点量化通路为 float |
| `OutType` | 最终输出类型，如 half、bfloat16_t 或 float |
| `DispatchPolicy` | `MatmulWithScaleFixpipeQuant<FullLoadMode, false, KernelQbmmPertensorMultiBlockStreamK>` |
| `X2ScaleType` | X2 per-tensor scale 的存储类型 |
| `X1ScaleType` | 可选 X1 per-tensor 标量 scale 类型，当前通路使用 float |

## Params

```cpp
struct Params {
    GM_ADDR cGmAddr{nullptr};
    GM_ADDR workspaceGmAddr{nullptr};
    GM_ADDR scaleGmAddr{nullptr};
    GM_ADDR perTokenScaleGmAddr{nullptr};
    GM_ADDR biasGmAddr{nullptr};
    bool isBias{false};
    uint32_t biasDtype{0};
};
```

| 字段 | 说明 |
|------|------|
| `cGmAddr` | 最终输出地址 |
| `workspaceGmAddr` | AIC 写出的 raw partial |
| `scaleGmAddr` | X2 per-tensor scale，读取第一个标量 |
| `perTokenScaleGmAddr` | 可选 X1 per-tensor scale；非空时只读取第一个标量 |
| `biasGmAddr` | AIV 侧 bias；MMAD 已处理的 bias 不应再次传入 |
| `isBias` | 是否在 epilogue 应用 bias |
| `biasDtype` | bias 运行时 dtype |

## X1 scale 判定

本模板不接收量化模式。`perTokenScaleGmAddr == nullptr` 表示没有第二路 scale；地址非空时读取一个
X1 per-tensor 标量。X2 scale 始终按 per-tensor 标量处理。

## X2 scale 类型

| `X2ScaleType` | 读取方式 |
|---------------|----------|
| float | 读取完整 fp32 标量 |
| bfloat16_t | 将 bf16 bits 扩展到 fp32 位域 |
| uint64_t / int64_t | 取 dequant scale 编码的低 32 bits 并解释为 fp32 |

读取后按非 StreamK 通路区分处理：

- 无 AIV 后加 bias：单 scale 直接应用 `0xFFFFE000` 掩码；双 scale 先以 fp32 合并，再对合并结果应用一次掩码；
- 有 AIV 后加 bias：保留完整 scale，不应用 Fixpipe 掩码，按 X2、X1 的顺序分别相乘。

## Bias 支持

AIV bias 支持以下运行时类型：

- `DT_FLOAT`：float；
- `DT_FLOAT16`：half；
- 其他已由上层校验为合法的分支按 bfloat16_t 处理。

int32 bias 属于反量化前累加，由 BlockMmad 处理，不应传入本模板。

以下 bias 属于反量化域，wrapper 将它传给本模板而不是 BlockMmad：

- INT8 输入、单路 FP32 per-tensor scale、FP32 bias；
- INT8 输入、单路 BF16 per-tensor scale、BF16 bias；
- FP8/HiFloat8 输入、双 FP32 per-tensor scale、FP32 bias。

计算顺序分别为：

```text
INT8: raw accumulator × X2 scale + same-dtype bias
FP8/HiFloat8: raw accumulator × X2 FP32 scale × X1 FP32 scale + FP32 bias
```

即 bias 位于全部 scale 之后。为避免 DP block 绕过 AIV 后处理，
host 只在所有输出 tile 都采用 SK/workspace 路径时选择该 StreamK 模板；其他调度
继续回退非 StreamK MIX。

## 核心接口

### Init

```cpp
__aicore__ inline void Init(
    Params const& params,
    BlockShape blockShapeInAiv,
    BlockShape tileL1ShapeInAiv,
    BlockCoord coordInAiv,
    uint64_t usedCoreNum,
    bool checkIsSkScene);
```

初始化问题规模、tile 数量、workspace、scale/bias 地址和 AIC-AIV 同步事件。`coordInAiv` 传入 `(mCnt, nCnt, kCnt, 1)`，用于描述 workspace 中的分片组织。

### Run / operator()

```cpp
__aicore__ inline void Run();
__aicore__ inline void operator()();
```

`operator()` 调用 `Run()`，完成 workspace 归约和 `DequantAndStore()`。

### DequantAndStore

```cpp
__aicore__ inline void DequantAndStore();
```

根据当前 AIV tile：

1. 建立 UB workspace、bias 和输出 ping-pong 布局；
2. 读取 X2 标量和可选 X1 标量；
3. `DequantCompute` 按运行时 `biasDtype_` 分发到 `DequantComputeTyped<BiasDtype>`，构造
   acc / dst / bias tensor 及 x2 / x1 的 fp32 占位 tensor（x2 恒为 per-tensor 标量语义，
   raw scale dtype 已由 `ReadRawX2ScaleScalar` 解析进 `x2ScaleScalar_`）；
4. 无 AIV bias 时 InitDequant 已把 x2*x1 预合并（并 `0xFFFFE000` 掩码）进
   `x2ScaleScalar_`，调用 `Tile::Dequant<B32ToFpTag>::Run<PER_TENSOR, NONE, false, BiasDtype>`
   以单条 Muls 复现 fixpipe 行为；有 AIV bias 时 x2/x1 保持分离，分别按
   `Run<PER_TENSOR, PER_TENSOR/NONE, true, BiasDtype>` 相乘后再加 bias；
5. cast 为 `OutType` 并写回 GM。

## Workspace 和 UB 对齐

- workspace 行 stride 按 `GetVecLen() / sizeof(WorkspaceType)` 对齐；
- scale 和 bias UB 区按 32B 对齐；
- 输出 N 轴按 `32 / sizeof(OutType)` 对齐；
- 输出使用 ping-pong 区，避免 vector 计算与 MTE3 写回互相阻塞。

## 数据流

```text
Workspace partial[0..kCnt)
        ↓ GM→UB
UB reduction
        ↓
WorkspaceType → fp32
        ↓
× merged-and-masked scale (no AIV bias)
或 × raw X2 scale [× raw X1 scale] (AIV bias)
        ↓
[+ fp32/fp16/bf16 bias]
        ↓
cast OutType
        ↓ UB→GM
C
```

## 使用示例

```cpp
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    0, false, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
using BlockEpilogue =
    Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        int32_t, half, DispatchPolicy, float, float>;
```

## 约束

- 必须与 QBMM per-tensor StreamK BlockMmad/Kernel 的 workspace 布局一致；
- X2 只支持 per-tensor 标量语义；
- X1 只支持 shape 为 `{1}` 的可选 per-tensor 标量；shape 为 `{M}` 的 per-token 输入不会路由到本模板；
- bias 必须由 wrapper 在 MMAD 和 epilogue 之间唯一分配；
- 上述 post-dequant bias 组合仅在 all-SK 调度下接入，DP+SK 调度由 host 回退；
- 仅在 AIV 上执行。
