# Epilogue 层开发指导

> Epilogue 层在 AIV（向量核）上执行 MMAD 计算后的后处理：反量化、激活函数、量化、写回 GM。

## 1. 核心定位

```
AIC（Cube 核）                    AIV（向量核）
    |                                  |
    |-- MMAD 计算 → L0C               |
    |-- L0C → UB（Fixpipe）----------→|
    |                                  |-- Epilogue：读 UB → 后处理 → 写 GM
```

Epilogue 运行在 AIV 上，与 AIC 的 MMAD 计算**并行**。两者通过块内 AIC-AIV 同步标志（`Gemm::Sync`）协同。

## 2. 标准接口

每个 Epilogue 必须暴露：

```cpp
struct Params { GM_ADDR outGmAddr; /* 其他运行时参数 */ };

void Init(const Params& params, ...);           // 一次性初始化（绑定 GM、配置 UB 布局）
void operator()(BlockShape, ...);                // 每个 tile 执行一次
// 可选：UpdateGlobalAddr(), UpdateNextProblem()  // 多 batch/group 场景
```

## 3. 主要 Epilogue 类型

| 类型 | 代表 | 数据流 |
|------|------|--------|
| **空操作** | `BlockEpilogueEmpty` | 无（占位符） |
| **Fixpipe 直写** | `BlockEpilogueFixpipe` | UB → 可选 ReLU → GM |
| **反量化** | `BlockEpilogueDequant` | int32→fp32→×scale→+bias→cast→GM |
| **MX 量化+激活** | `BlockEpilogueGeluMxQuant` | fp32→Gelu→bf16→MaxExp→Scale→FP8/FP4→GM |
| **SwiGLU+MX 量化** | `BlockEpilogueSwigluMxQuant` | fp32→SwiGLU→bf16→MX 量化→GM |
| **StreamK 归约** | `BlockEpilogueMatmulStreamK` | workspace GM→UB→K 轴 Add→cast→GM |
| **Per-TOKEN Scale** | `BlockEpiloguePerTokenScale` | UB→×perTokenScale→cast→GM |
| **Scale+Add** | `BlockEpilogueFmmWithScaleAdd` | UB→×alpha+beta×x3→GM |

## 4. 关键实现模式

### 4.1 AIC/AIV 同步（Fixpipe 路径）

AIC 写 L0C→UB 后设 flag，AIV 等 flag 后读 UB 处理：

```cpp
// AIV 侧
Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);
// ... 处理 UB 数据 ...
Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
```

### 4.2 UB 布局

大多数 Epilogue 手动按字节偏移划分 UB：

```cpp
// Init 中计算各区域偏移
uint64_t inputOffset = 0;
uint64_t outputOffset = inputOffset + inputSize;
uint64_t scaleOffset = outputOffset + outputSize;
// 通过 MakeMemPtr<UB>(offset) 创建 tensor
```

### 4.3 VF 向量计算

Epilogue 的核心计算通过 `__simd_vf__` 函数在向量单元执行：

```cpp
static __simd_vf__ inline void DequantVf(__ubuf__ float* input, __ubuf__ float* output, ...) {
    // RegTensor 寄存器操作：Cast, Mul, Add, Exp, Reduce...
}

// 调用
asc_vf_call<DequantVf>(inputAddr, outputAddr, ...);
```

### 4.4 Ping-Pong 输出

M 维度分块处理，奇偶块交替使用 UB 的两个输出区域，使向量计算和 MTE3 写回重叠：

```cpp
for (int mChunk = 0; mChunk < mChunks; ++mChunk) {
    int outOffset = (mChunk & 1) ? pongOffset_ : pingOffset_;
    // 计算写入 outOffset，上一块的 MTE3 同时写另一块
}
```

### 4.5 变体分支（`if constexpr`）

```cpp
if constexpr (DispatchPolicy::FUSED_OP_TYPE == OP_TYPE_RELU) { /* 启用 ReLU */ }
if constexpr (is_same_v<OutType, bfloat16_t>) { /* 跳过 ReLU（硬件限制） */ }
if constexpr (is_same_v<OutType, fp8_e4m3fn_t>) { /* FP8 量化路径 */ }
if constexpr (is_same_v<OutType, fp4x2_e2m1_t>) { /* FP4 量化路径 */ }
```

## 5. 与 Kernel 层的交互

Kernel 通过模板参数注入 Epilogue：

```cpp
// Kernel 模板
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal { ... };

// Kernel 内部
BlockEpilogue epilogueOp;
epilogueOp.Init(params.epilogueParams);
// 在 tile 循环中调用
epilogueOp(blockShape, offsetC, splitM, baseM, baseN);
```

## 6. 检查清单

- [ ] 暴露 `Params` + `Init` + `operator()`
- [ ] UB 布局在 `Init` 中一次性计算
- [ ] 块内 AIC-AIV 同步 flag ID 与 AIC 侧一致
- [ ] VF 函数用 `__simd_vf__`，可组合子函数用 `__simd_callee__`
- [ ] 向量写操作后 `LocalMemBar` 保证可见性
- [ ] `static_assert` 校验元素类型、内存位置、Layout
