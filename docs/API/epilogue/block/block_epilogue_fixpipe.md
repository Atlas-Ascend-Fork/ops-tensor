# Block Epilogue Fixpipe
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_fixpipe.h)

## 功能说明
Fixpipe 模式矩阵乘后处理 Block，运行在 AIV（Vector）核。与 AIC（Cube）核协同：AIC 将 L0C 计算结果通过 fixpipe 搬运到 UB，AIV 等待 UB 数据就绪后，可选执行 ReLU 激活，再通过 `DataCopyPad` 将结果搬出到 GM。

**继承自**：[Block Epilogue 基础框架](./block_epilogue.md)

## 特殊约束

### AIV 核专用
仅运行在 AIV（Vector）核，与 AIC（Cube）核通过块内 AIC-AIV 同步标志协同工作：
- **AIC 核**：执行 Mmad 计算 → `CopyL0C2UB`（fixpipe 搬运 L0C→UB）→ 设置"数据就绪"标志
- **AIV 核**：等待"数据就绪"标志 → （可选）ReLU → `DataCopyPad`（UB→GM）→ 设置"UB 空闲"标志

### 任务类型
配套 Kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2`（AIC:AIV = 1:2）混合核任务类型。

### 数据类型约束
- `sizeof(DataTypeIn) >= sizeof(DataTypeOut)`（`Init` 中 `static_assert` 强制校验）
- 典型组合：`DataTypeIn` 与 `DataTypeOut` 相同（如 `half`/`half`、`float`/`float`），或 `DataTypeIn` 为高精度类型

### ReLU 融合支持
由 `DispatchPolicy::FUSED_OP_TYPE` 控制：
- `OP_TYPE_RELU`（5）：在搬出前对 UB 数据执行 `AscendC::Relu`
- `OP_TYPE_EMPTY`（0）等其他值：不执行融合，直接搬出
- **bfloat16_t 输出不支持 ReLU**（硬件限制，编译期跳过）

### Ping-Pong 双缓冲
当 `ubDB > 1` 时启用：
- UB 分为两个 slot，AIC 与 AIV 交替使用不同 slot
- 通过 `cvPingPong_` 计数器轮转 slot（`slot = cvPingPong_ & 1`）
- `cvPingPong_` 跨 tile 不重置，实现连续 tile 间的 loc2ub 与 ub2gm 并行
- 同步标志 ID 也随 slot 偏移，避免覆盖

## 特殊静态常量

| 常量 | 值 | 说明 |
|------|----|------|
| DATA_BLOCK | 32 | 数据块对齐字节数（32B） |
| OUT_ALIGN | `DATA_BLOCK / sizeof(DataTypeOut)` | 输出元素对齐个数 |
| AIC_SYNC_AIV_MODE_4 | 4 | 块内 AIC-AIV 同步模式（SYNC_MODE_INTRA） |
| AIV_SYNC_AIC_FLAG | 4 | AIV→AIC"UB 空闲"标志基址 |
| AIC_SYNC_AIV_FLAG | 6 | AIC→AIV"数据就绪"标志基址 |
| FLAG_ID_MAX | 16 | 标志 ID 上限 |
| SPLIT_M_ALIGN | 2 | M 切分对齐因子 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| DataTypeOut | 输出数据类型（模板参数 `DataTypeOut_`，GM 输出） |
| DataTypeIn | 输入数据类型（模板参数 `DataTypeIn_`，UB 输入，来自 fixpipe） |
| FusionOp | 融合算子（默认 `Gemm::Block::DefaultFusion`，仅拷贝） |
| DispatchPolicy | 调度策略（模板参数，需含 `FUSED_OP_TYPE` 静态成员） |
| BlockShape | Block 形状 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` |
| ProblemShape | 问题规模 `asc::te::shape<int64_t, int64_t, int64_t, int64_t>` |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR outGmAddr{nullptr};   // 输出矩阵 C 的 GM 地址
};
```
说明：仅包含输出 GM 地址，由 Kernel 框架在组装 `Params` 时填入。

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockEpilogueFixpipe()
```
功能：构造 BlockEpilogueFixpipe 对象（空实现）。

### Init函数
```
__aicore__ inline void Init(Params const& params, ProblemShape& problemShape)
```
功能：初始化组件，绑定输出 GM buffer 并记录问题规模。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| params | Params const& | 包含 `outGmAddr`（输出 GM 地址） |
| problemShape | ProblemShape& | 问题规模 `(m, n, k, batch)` |

执行流程：
1. `outputGlobal_.SetGlobalBuffer(outGmAddr)` 绑定输出 GM
2. 记录 `problemShape_`
3. `static_assert(sizeof(DataTypeIn) >= sizeof(DataTypeOut))` 编译期类型校验

### Run函数
```
__aicore__ inline void Run(
    BlockShape const& blockShape,   // Block 形状 (mL1, nL1, kL1, ...)
    int64_t dstOffset,              // GM 输出起始偏移
    bool splitM,                    // 是否按 subBlock 切分 M
    int64_t baseM,                  // M 轴基准大小（0 表示用 mL1）
    int64_t baseN,                  // N 轴 tile 大小（0 表示用 nL1）
    uint64_t ubDB = 1)              // UB 双缓冲级数（>1 启用 ping-pong）
```
功能：执行 Fixpipe 后处理，将 UB 中的 L0C 结果搬出到 GM。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockShape | BlockShape const& | 当前 Block 形状，取 M、N 维度 |
| dstOffset | int64_t | GM 输出起始偏移（按元素计） |
| splitM | bool | true 时按 `GetSubBlockIdx()` 将 M 切分给不同 subBlock |
| baseM | int64_t | M 轴基准大小；非 0 时 `curM = min(mL1, baseM)` |
| baseN | int64_t | N 轴 tile 大小；非 0 时按 `baseN` 切分 N 轴多次搬出 |
| ubDB | uint64_t | UB 双缓冲级数；`>1` 时启用 ping-pong |

执行流程：
1. **M 维计算**：`curM = min(mL1, baseM)`；`halfBlockShapeM = CeilDiv(curM, GetTaskRation())`；`splitM` 时按 subBlock 取奇偶半块
2. **N 维切分**：`curBaseN = min(nL1, baseN)`；`nL1Iter = CeilDiv(nL1, curBaseN)`
3. **双缓冲判定**：`enablePp = (ubDB > 1)`
4. **N 轴循环**（`nIdx = 0 .. nL1Iter-1`）：
   - 计算 `tileN`（尾块取余量）与 `blockShapeNAlign = CeilAlign(tileN, c0Size)`
   - `slot = enablePp ? (cvPingPong_ & 1) : 0`
   - **等待 AIC 数据就绪**：`Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V/MTE3>(AIC_SYNC_AIV_FLAG + slot)`
     - `FUSED_OP_TYPE == OP_TYPE_RELU` 时等 `PIPE_V`，否则等 `PIPE_MTE3`
   - 取 UB 对应 slot：`ubLocalTmp_ = ubLocal[slot * ubHalfElems]`
   - **ReLU 融合**（可选）：`FUSED_OP_TYPE == OP_TYPE_RELU` 且非 `bfloat16_t` 时执行 `Relu` + `V_MTE3` 标志同步
   - **搬出**：`DataCopyPad<DataTypeOut>(outputGlobal_[offset], ubLocalTmp_, copyParams)`
     - `offset = dstOffset + nIdx * curBaseN + halfBlockShapeM * N * (subBlockIdx & 1)`
   - **通知 AIC UB 空闲**：`Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot)`
   - `cvPingPong_++`

### operator函数
```
__aicore__ inline void operator()(
    BlockShape const& blockShape, int64_t dstOffset = 0, bool splitM = false,
    int64_t baseM = 0, int64_t baseN = 0, uint64_t ubDB = 1)
```
功能：执行后处理，直接调用 `Run(blockShape, dstOffset, splitM, baseM, baseN, ubDB)`。
说明：由 Kernel 框架（如 `GemmUniversal`）在 AIV 侧以 `epilogue(validBlockShape, offsetC, splitM, baseM, baseN, ubDB)` 形式调用。

## 事件同步

| 事件 | 方向 | 标志 ID | 用途 |
|------|------|---------|------|
| AIC→AIV 数据就绪 | AIC Set / AIV Wait | `AIC_SYNC_AIV_FLAG + slot`（6+slot） | 通知 AIV 该 slot 的 UB 数据可消费 |
| AIV→AIC UB 空闲 | AIV Set / AIC Wait | `AIV_SYNC_AIC_FLAG + slot`（4+slot） | 通知 AIC 该 slot 的 UB 可覆写 |
| V_MTE3（ReLU 时） | V → MTE3 | 0x0 | ReLU 完成后再搬出 |

说明：`AIC_SYNC_AIV_MODE_4`（= `SYNC_MODE_INTRA`）为块内 AIC-AIV 同步模式（底层 `asc_sync_intra_wait/arrive`）；等待管线下标在 ReLU 场景为 `PIPE_V`（先算后搬），其余为 `PIPE_MTE3`（直接搬）。

## 调用示例

### 组件组装
```
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
    Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 0>;   // L0C2OUT=1v2 fixpipe, 无融合
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFixpipe<
    half, half, DispatchPolicy>;             // DataTypeOut, DataTypeIn, DispatchPolicy
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```
using Params = typename MatmulKernel::Params;
Params params = {
    {m, n, k, 1},       // ProblemShape
    {...},              // BlockMmad::Params（aGmAddr/bGmAddr/cGmAddr/tiling 等）
    {cGM},              // BlockEpilogue::Params（outGmAddr）
    {...}               // BlockScheduler::Params
};
```

### 组件执行
```
MatmulKernel kernel;
kernel(params);   // 框架内部在 AIV 侧调用 epilogue(blockShape, offsetC, splitM, baseM, baseN, ubDB)
```

> 完整示例参见 `examples/mat_mul/mat_mul_fixpipe_opti/mat_mul_fixpipe_opti.cpp`。

## 数据流

### 存储层次
```
AIC: Mmad → L0C ─(CopyL0C2UB/fixpipe)→ UB[slot]
                                        ↓ 块内同步 数据就绪
AIV: WaitFlag ─→ (ReLU, 可选) ─→ DataCopyPad ─→ GM(C)
                                        ↓ 块内同步 UB 空闲
AIC: 复用 UB[slot] 写下一块
```

### 执行流程
```
AIC 侧（BlockMmad）                        AIV 侧（BlockEpilogueFixpipe）
  Mmad → L0C                                  cvPingPong_ 跨 tile 不重置
    ↓                                          ↓
  WaitFlag(AIV→UB_free, slot)                N 轴循环 (nIdx)
    ↓                                          ├─ WaitFlag(AIC→数据就绪, slot)
  CopyL0C2UB → UB[slot]                        │   (PIPE_V 若 ReLU，否则 PIPE_MTE3)
    ↓                                          ├─ 取 ubLocalTmp_ = ubLocal[slot]
  SetFlag(数据就绪, slot)                      ├─ ReLU（可选）+ V_MTE3 同步
    ↓                                          ├─ DataCopyPad: UB → GM
  (写下一块...)                                └─ SetFlag(UB 空闲, slot)；cvPingPong_++
```

### GM 输出布局
```
offset = dstOffset + nIdx * curBaseN + halfBlockShapeM * N * (GetSubBlockIdx() & 1)
```
说明：N 轴按 `curBaseN` 步进；`splitM` 时不同 subBlock 在 M 方向偏移 `halfBlockShapeM * N`。

## 性能优化建议

### 双缓冲配置
- `ubDB > 1` 时启用 ping-pong，AIC 写 UB[slot] 与 AIV 读 UB[slot^1] 并行
- `cvPingPong_` 跨 tile 不重置，使连续 tile 间交替使用 slot 0/1，实现 loc2ub 与 ub2gm 流水并行
- `ubDB = 1` 时退化为单缓冲，固定 slot 0

### M 切分
- `splitM = true` 时按 `GetTaskRation()` 将 M 分配给多个 subBlock，提升并行度
- 尾块 `curM` 为奇数时，subBlock 0 取 `halfBlockShapeM`，subBlock 1 取 `halfBlockShapeM - 1`

### N 轴切分
- `baseN` 控制 N 轴 tile 大小，影响 `nL1Iter` 次数与单次搬出 burst 长度
- `DataCopyExtParams.burstLen = tileN * sizeof(DataTypeOut)`，`dstGap = (N - tileN) * sizeof(DataTypeOut)`

### ReLU 融合位置
- ReLU 在 UB→GM 搬出前执行（`PIPE_V`），搬出在 `PIPE_MTE3`
- bfloat16_t 输出跳过 ReLU（硬件不支持），直接搬出

## 适用场景

- **Fixpipe Kernel**：`GemmUniversal` 按 `KernelMmadMultiBlockFixpipeOpti` / `KernelMmadMultiBlockBFullLoad` 调度策略的特化（Fixpipe 模式）
- **L0C→UB→GM 输出路径**：非 `ON_THE_FLY` 直销场景
- **AIC/AIV 协同**：`MIX_AIC_1_2` 混合核任务
- **可选 ReLU**：`FUSED_OP_TYPE = OP_TYPE_RELU` 的融合矩阵乘

## 相关文档

- [Block Epilogue 基础框架](./block_epilogue.md) — Epilogue 公共接口说明
- [BlockMmadMatmulFixpipeOpti](../../gemm/block/block_mmad_matmul_fixpipe_opti.md) — Fixpipe 优化的 BlockMmad
- [BlockMmadMatmulBL1FullLoad](../../gemm/block/block_mmad_matmul_bl1_full_load.md) — B 全载 BlockMmad（支持 Fixpipe）
- [KernelMatmulFixpipeOpti](../../gemm/kernel/kernel_matmul_fixpipe_opti.md) — Fixpipe 优化 Kernel（GemmUniversal 特化）
