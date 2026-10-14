# Block Epilogue Dequant
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_dequant.h)

## 功能说明
MIX 模板的 dequant 向量后处理 Block，在 **AIV** 上执行。读取 AIC 经 fixpipe（NoQuant）搬到 UB 的 int32或fp32累加结果，在向量上完成反量化：

$$ out = (x1@x2) \times x2Scale\ [\times\ x1Scale] + bias $$

输出 bf16 / fp16 / fp32。x1@x2 累加器从 UB 读，x2Scale / x1Scale / bias 从 GM 读；scale 乘法与 bias 加法委托可复用的
[`Tile::Dequant<B32ToFpTag>`](../tile/arch35/dequant.md) 计算链，本 Block 保留 GM→UB staging、
runtime 模式分发、双 AIV 子块切分、4 路 M 划分与 ping-pong 输出。

**关联框架**：[Block Epilogue 基础框架](./block_epilogue.md)
**上游计算**：[block_mmad_a8w8_mix](../../gemm/block/block_mmad_a8w8_mix.md)（AIC 写 L0C→UB）

## 特殊约束

### 计算位置
仅在 AIV 上执行；由 MIX Kernel（[kernel_qbmm_mix](../../gemm/kernel/kernel_qbmm_mix.md) / [without_batch](../../gemm/kernel/kernel_qbmm_mix_without_batch.md)）在 `WaitForCube()` 后调用。

### 量化模式（QuantMode）
| 模式 | 值 | 说明 |
|------|----|------|
| DEFAULT | 0x0 | 默认 |
| PERTENSOR_MODE | 0x1 | per-tensor（x1 标量 scale） |
| PERCHANNEL_MODE | 0x2 | x2 per-channel scale（从 GM 读向量） |
| PERTOKEN_MODE | 0x4 | x1 per-token scale（从 GM 读向量） |

- `isPerChannel_`：x2Scale 为 per-channel（向量）；否则按标量读取（`ReadX2ScaleScalar`）。
- `isPerToken_`：x1Scale 为 per-token（向量）；`isX1PerTensor_`：x1Scale 为 per-tensor 标量。

### 对齐要求
- `DATA_BLOCK = 32`，需与 Kernel 侧 L0C→UB 行距 `CeilAlign(curN, L0C_ALIGN)` 一致（`L0C_ALIGN = 32 / sizeof(L0CType)`），否则 N-tail 场景 writer/reader 行距错位。
- `FP32_OUTPUT_TIMES`：OutType 为 float 时 4，否则 2；`CV_RATIO = 2`（双 AIV 子块）。

## 特殊静态常量
| 常量 | 说明 |
|------|------|
| DATA_BLOCK | 向量数据块大小（32 字节） |
| FLOAT_ALIGN / L0C_ALIGN / OUT_ALIGN | float / L0C / Out 类型的对齐元素数 |
| CV_RATIO | cube:vector 比（2，双子块） |
| FP32_OUTPUT_TIMES | 输出展开倍数（float:4 / 其它:2） |

## 特殊类型别名
| 类型 | 说明 |
|------|------|
| OutType | 输出类型（bf16 / fp16 / fp32） |
| BiasType | 编译期 bias 类型（int8 场景为 int32；实际按 biasDtype 解释） |
| X2ScaleType | x2Scale 类型 |
| X1ScaleType | x1Scale 类型（默认 float） |
| L0CType | L0C 累加类型（int32_t或fp32） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR x2ScaleGmAddr{nullptr};  // x2Scale GM 地址（per-channel 向量 / 标量）
    GM_ADDR x1ScaleGmAddr{nullptr};  // x1Scale GM 地址（per-token 向量 / per-tensor 标量）
    GM_ADDR biasGmAddr{nullptr};     // bias GM 地址（可选）
    GM_ADDR outGmAddr{nullptr};      // 输出 GM 地址
    int64_t m{0}, n{0};              // 全局 M / N
    int64_t baseM{0}, baseN{0};      // tile 基准
    uint32_t x1QuantMode{0};         // x1 量化模式（per-token / per-tensor）
    uint32_t x2QuantMode{0};         // x2 量化模式（per-channel / per-tensor）
    bool isBias{false};              // 是否启用 bias
    uint32_t biasDtype{0};           // 真实 bias dtype（DT_FLOAT/DT_FLOAT16/DT_BF16）
};
```

## 特殊成员方法

### 构造/析构函数
```
__aicore__ inline BlockEpilogueDequant() {}
__aicore__ inline ~BlockEpilogueDequant()   // 等待 V_MTE2(0..2) 与 MTE3_V(0..1)
```

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能：记录 m/n/baseM/baseN 与各量化标志；按量化模式设置 x2Scale / x1Scale（向量 GM 或标量）；`isBias_` 时记录 `biasDtype_` 与 bias 地址；`SetupUbLayout()` 规划 UB 布局；设置 V_MTE2 / MTE3_V 同步标志。

### operator函数
```
__aicore__ inline void operator()(
    int64_t singleCoreM,    // 当前 tile 的 M
    int64_t singleCoreN,    // 当前 tile 的 N
    int64_t offsetScale,    // x2Scale 偏移（= nPos）
    int64_t offsetPtScale,  // x1Scale 偏移（= mPos）
    int64_t offsetBias,     // bias 偏移（= nPos，三维 bias 叠加 batch 偏移）
    int64_t offsetC)        // 输出 C 偏移
```
执行流程：
1. 双 AIV 子块按 `CV_RATIO` 切分 M（`subBlockIdx_`），`singleMInVec <= 0` 直接返回。
2. `DequantCompute` 按运行时 `biasDtype_` 分发到 `DequantComputeTyped<BiasDtype>`（fp32 / half / bf16）。
3. 构造五路 UB tensor（`MakeNDExtLayout`）：acc（行距 nSrcAligned）、dst（行距 nDstAligned）、
   x2Scale（per-channel 单行向量，否则占位）、x1Scale（per-token fp32 向量，否则占位）、
   bias（单行向量，否则占位）；禁用模式槽位传同 dtype 占位张量（Tile 不解引用）。
4. 把 runtime 量化标志（`isPerChannel_` / `isPerToken_` / `isX1PerTensor_` / `isBias_`）翻译成
   编译期模板参数，12 分支调用矩阵分发到
   `Tile::Dequant<B32ToFpTag>::Run<x2Gran, x1Gran, hasBias, BiasDtype>`
   （x2: PER_TENSOR/PER_CHANNEL × x1: PER_TOKEN/PER_TENSOR/NONE × bias），数值链
   （累加器 cast → ×x2Scale → ×x1Scale → +bias → cast → store）全部在 Tile 内完成。
5. ping-pong 写回 GM。

## 数据流
```
UB(int32或fp32) ─Cast→ fp32 ─× x2Scale ─[× x1Scale] ─[+ bias(按 biasDtype)] ─Cast→ OutType ─→ GM
```

## 设计说明
- 与上游 [block_mmad_a8w8_mix](../../gemm/block/block_mmad_a8w8_mix.md) 分工：AIC 只做 int32或fp32 累加 + L0C→UB，AIV 独占 dequant，避免 cube 上做向量后处理。
- 与 [`Tile::Dequant<B32ToFpTag>`](../tile/arch35/dequant.md) 分工：Block 负责 GM→UB staging、
  runtime 模式/bias dtype 分发与张量构造（含占位张量约定），Tile 负责纯计算；
  同一 Tile 计算链被 StreamK 通路（[BlockEpilogueQbmmPertensorStreamK](./block_epilogue_qbmm_pertensor_streamk.md)）复用。

## 适用场景
- int8（per-token / per-channel / per-tensor）量化矩阵乘的反量化后处理。
- 支持带 bias（bf16/fp16/fp32）输出 bf16/fp16/fp32。
