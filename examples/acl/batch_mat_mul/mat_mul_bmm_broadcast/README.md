# mat_mul_bmm_broadcast Example

## 概述

本示例演示基于 Blaze 框架的 BatchMatmul Broadcast 算子在昇腾 NPU 上的实现。该算子执行批量矩阵乘法，支持 A 和 B 具有不同的 batch 维度，通过广播机制将输入扩展到 C 的 batch 维度，适用于带广播语义的批量矩阵乘场景。

- **算子**: mat_mul
- **场景**: mat_mul_bmm_broadcast
- **算法特点**: Per-tile batch broadcast，A/B 的 batch 维度可不同，按 modulo 算法广播到 C batch；支持 FP16/BF16/FP32/HF32
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_batch_matmul_broadcast.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[batchA, M, K]`（transA=false）或 `[batchA, K, M]`（transA=true）
- 输入 B shape: `[batchB, K, N]`（transB=false）或 `[batchB, N, K]`（transB=true）
- 输出 C shape: `[batch, M, N]`
- 数据类型: float16, bfloat16, float32
- batch 广播约束: `batchA` 和 `batchB` 可为 1（触发广播）或等于 `batch`；当 `batchA=1` 且 `batch>1` 时 A 沿 batch 维广播，`batchB=1` 且 `batch>1` 时 B 沿 batch 维广播
- bias: 1D 向量，大小必须等于 n 或 0（无 bias）

## Batch Broadcast 说明

BatchMatmul Broadcast 的核心特点是 A 和 B 的 batch 维度可以不同，通过取模广播对齐到 C 的 batch 维度：

```
标准 BatchMatMul:    batchA == batchB == batch（三者必须相等）
BMM Broadcast:       batchA、batchB 可独立取 1 或 batch，按 modulo 广播
```

**广播规则**：

| batchA | batchB | batch | 行为                                              |
| ------ | ------ | ----- | ------------------------------------------------- |
| 1      | batch  | batch | A 广播：每个 batch 复用同一份 A                   |
| batch  | 1      | batch | B 广播：每个 batch 复用同一份 B                   |
| 1      | 1      | batch | A、B 均广播（退化为单次 matmul 复制到所有 batch） |
| batch  | batch  | batch | 无广播（标准 BMM）                                |

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=batch_mat_mul --target=mat_mul_bmm_broadcast
```

### 测试用例定义

测试用例定义在 `mat_mul_bmm_broadcast.csv` 中，格式如下：

```csv
casename,m,k,n,batch,batchA,batchB,bias,dtype,transA,transB,hf32
bmm_broadcast_fp16_ab,64,128,64,4,1,4,0,float16,false,false,false
bmm_broadcast_fp16_bb,64,128,64,4,4,1,0,float16,false,false,false
bmm_broadcast_bf16,32,64,32,8,1,8,0,bfloat16,false,false,false
bmm_broadcast_fp32,32,64,32,4,4,4,0,float32,false,false,false
bmm_broadcast_hf32,32,64,32,4,4,4,0,float32,false,false,true
```

**列说明**：

| 列       | 说明                                      |
| -------- | ----------------------------------------- |
| casename | 用例名称                                  |
| m, k, n  | 矩阵维度                                  |
| batch    | C 的 batch 维度大小                       |
| batchA   | A 的 batch 维度大小（1 表示广播）         |
| batchB   | B 的 batch 维度大小（1 表示广播）         |
| bias     | bias 向量大小，必须等于 n 或 0（无 bias） |
| dtype    | 数据类型：float16 / bfloat16 / float32    |
| transA   | A 矩阵是否转置                            |
| transB   | B 矩阵是否转置                            |
| hf32     | 是否启用 HF32 模式（仅 float32 有效）     |

### 结果输出

执行完成后结果写入 `mat_mul_bmm_broadcast_result.csv`。

## 数据与校验

### 输入数据

由 `../scripts/gen_data.py` 生成：

- `input/input_a.bin`: A 矩阵 `[batchA, M, K]`（或转置布局）
- `input/input_b.bin`: B 矩阵 `[batchB, K, N]`（或转置布局）
- `input/bias.bin`: bias 向量（bias>0 时生成，shape 为 `[batch, n]`）
- `output/cpu_output.bin`: CPU 参考结果 `[batch, M, N]`（已包含广播计算和 bias）

### 输出数据

- `output/npu_out.bin`: NPU 计算结果 `[batch, M, N]`

### 验证标准

由 `../scripts/verify_result.py` 执行，采用混合容差策略（per-element 阈值 + 超差比例阈值，二者均为 `ratio_tol`）：

| dtype    | ratio_tol |
| -------- | --------- |
| float16  | 5e-3      |
| bfloat16 | 5e-3      |
| float32  | 1e-4      |
| hf32     | 1e-3      |

- float16/bfloat16：per-element 绝对误差阈值
- float32：per-element 相对误差阈值（`|golden - npu| / max(|golden|, |npu|)`）
- 通过条件：超差元素比例 `error_ratio <= ratio_tol`

## Bias 支持

mat_mul_bmm_broadcast 支持 bias 功能。bias 是一个 2D 张量，shape 为 `[batch, n]`，大小必须等于 n 或 0。

**数据流**：

1. `gen_data.py` 生成 `input/bias.bin`（`batch × n` 个元素）
2. `gen_data.py` 生成 `output/cpu_output.bin`（已包含 bias：`C = A @ B + bias`）
3. C++ kernel 读取 `input/bias.bin`，在计算时应用 bias
4. `verify_result.py` 比较 NPU 输出和 CPU golden（两者都已包含 bias）

**约束**：

- bias 必须等于 n 或 0（无 bias）
- bias 数据类型与 dtype 一致

## 代码结构

```
mat_mul_bmm_broadcast/
├── mat_mul_bmm_broadcast.cpp              # kernel 实现
├── mat_mul_bmm_broadcast.conf             # 参数路由配置
├── mat_mul_bmm_broadcast.csv              # CSV 测试用例
└── README.md                              # 本文档
```

构建配置在 op 层 `examples/acl/batch_mat_mul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/batch_mat_mul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件：

| 组件            | 头文件                                                | 职责                                                                         |
| --------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_batch_matmul_broadcast.h` | 完整 kernel 入口（BMM Broadcast 专用）                                       |
| Block MMAD      | `blaze/gemm/block/block_mmad_matmul_basic.h`        | Block 级矩阵乘                                                               |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_basic.h`   | 基础调度器                                                                   |
| Epilogue        | `blaze/epilogue/block/block_epilogue_empty.h`       | 后处理（空）                                                                 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`               | 派发策略（`MatmulMultiBlockBasic` + `KernelMmadMultiBlockBmmBroadcast`） |
