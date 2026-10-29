# mat_mul_iterbatch_broadcast Example

## 概述

本示例演示基于 Blaze 框架的 IterBatch-Broadcast MatMul 算子在昇腾 NPU 上的实现。该算子在 BatchMatmul 广播语义基础上引入 iterbatch 流水线机制：多个 batch 被同时加载到 L1 缓存并在 L0 流水线中并行处理，通过 L1/L0 两级 iterbatch 深度提升 batch 维度的计算吞吐。适用于带广播语义且需要跨 batch 流水重叠的批量矩阵乘场景。

- **算子**: mat_mul
- **场景**: mat_mul_iterbatch_broadcast
- **算法特点**: 结合 batch broadcast 与 iterbatch L1/L0 流水线，A/B 的 batch 维度可不同并按 modulo 广播；支持 FP16/BF16/FP32/HF32
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_batch_matmul_iterbatch_broadcast.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[batchA, M, K]`（transA=false）或 `[batchA, K, M]`（transA=true）
- 输入 B shape: `[batchB, K, N]`（transB=false）或 `[batchB, N, K]`（transB=true）
- 输出 C shape: `[batch, M, N]`
- 数据类型: float16, bfloat16, float32
- batch 广播约束: `batchA` 和 `batchB` 可为 1（触发广播）或等于 `batch`；当 `batchA=1` 且 `batch>1` 时 A 沿 batch 维广播，`batchB=1` 且 `batch>1` 时 B 沿 batch 维广播，A/B 可同时广播
- iterBatchL1: L1 缓存同时驻留的 batch 数（≥1）
- iterBatchL0: L0 流水线并行处理的 batch 数（≥1）
- bias: 1D 向量，大小必须等于 n 或 0（无 bias）

## IterBatch 流水线与广播说明

IterBatch-Broadcast 在 BMM 广播语义上叠加 L1/L0 两级 batch 流水线：

```
标准 BMM Broadcast:  逐 batch 串行处理，每个 batch 独立完成 L1 加载 → L0 计算
IterBatch-Broadcast: 多个 batch 同时驻留 L1，在 L0 流水线中交错计算
```

**iterbatch 参数**：

| 参数        | 层级      | 含义                                          |
| ----------- | --------- | --------------------------------------------- |
| iterBatchL1 | L1 缓存   | 同时驻留 L1 的 batch 数，控制 L1 数据复用深度 |
| iterBatchL0 | L0 流水线 | L0 流水线并行 batch 数，控制计算重叠深度      |

**广播规则**（与 mat_mul_bmm_broadcast 一致）：

| batchA | batchB | batch | A_BC  | B_BC  | 行为                            |
| ------ | ------ | ----- | ----- | ----- | ------------------------------- |
| 1      | batch  | batch | true  | false | A 广播：每个 batch 复用同一份 A |
| batch  | 1      | batch | false | true  | B 广播：每个 batch 复用同一份 B |
| 1      | 1      | batch | true  | true  | A、B 均广播                     |
| batch  | batch  | batch | false | false | 无广播（标准 BMM）              |

> `A_BC` / `B_BC` 为编译期模板参数，host 侧根据 `batchA`/`batchB` 与 `batch` 的关系在 4 种组合中分派。

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=batch_mat_mul --target=mat_mul_iterbatch_broadcast
```

### 测试用例定义

测试用例定义在 `mat_mul_iterbatch_broadcast.csv` 中，格式如下：

```csv
casename,m,k,n,batch,batchA,batchB,iterBatchL1,iterBatchL0,bias,dtype,transA,transB,hf32
iterbatch_fp16_ab,32,16,32,8,1,8,4,1,0,float16,false,false,false
iterbatch_fp16_bb,32,16,32,8,8,1,4,1,0,float16,false,false,false
iterbatch_bf16,16,16,16,4,1,4,2,1,0,bfloat16,false,false,false
iterbatch_fp32,16,32,16,4,4,4,2,1,0,float32,false,false,false
iterbatch_hf32,16,32,16,4,4,4,2,1,0,float32,false,false,true
iterbatch_fp16_bb_l0,32,16,32,8,8,1,4,2,0,float16,false,false,false
iterbatch_fp16_ab_both,32,16,32,4,1,1,2,1,0,float16,false,false,false
iterbatch_fp16_both_l0,32,16,32,8,1,1,4,2,0,float16,false,false,false
iterbatch_bf16_bb_l0,16,16,16,4,4,1,2,2,0,bfloat16,false,false,false
iterbatch_fp32_bb_l0,16,32,16,4,4,1,2,2,0,float32,false,false,false
```

**列说明**：

| 列          | 说明                                      |
| ----------- | ----------------------------------------- |
| casename    | 用例名称                                  |
| m, k, n     | 矩阵维度                                  |
| batch       | C 的 batch 维度大小                       |
| batchA      | A 的 batch 维度大小（1 表示广播）         |
| batchB      | B 的 batch 维度大小（1 表示广播）         |
| iterBatchL1 | L1 缓存同时驻留的 batch 数                |
| iterBatchL0 | L0 流水线并行 batch 数                    |
| bias        | bias 向量大小，必须等于 n 或 0（无 bias） |
| dtype       | 数据类型：float16 / bfloat16 / float32    |
| transA      | A 矩阵是否转置                            |
| transB      | B 矩阵是否转置                            |
| hf32        | 是否启用 HF32 模式（仅 float32 有效）     |

**用例覆盖**：

| 用例类型       | 代表用例                   | 覆盖点                           |
| -------------- | -------------------------- | -------------------------------- |
| A 广播         | `iterbatch_fp16_ab`      | batchA=1, batchB=batch           |
| B 广播         | `iterbatch_fp16_bb`      | batchA=batch, batchB=1           |
| A/B 双广播     | `iterbatch_fp16_ab_both` | batchA=1, batchB=1               |
| 无广播         | `iterbatch_fp32`         | batchA=batchB=batch              |
| L0 流水线加深  | `iterbatch_fp16_bb_l0`   | iterBatchL0=2                    |
| L1/L0 同时加深 | `iterbatch_fp16_both_l0` | iterBatchL1=4, iterBatchL0=2     |
| 全 dtype       | fp16/bf16/fp32/hf32        | float16, bfloat16, float32, HF32 |

### 结果输出

执行完成后结果写入 `mat_mul_iterbatch_broadcast_result.csv`。

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

mat_mul_iterbatch_broadcast 支持 bias 功能。bias 是一个 2D 张量，shape 为 `[batch, n]`，大小必须等于 n 或 0。

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
mat_mul_iterbatch_broadcast/
├── mat_mul_iterbatch_broadcast.cpp         # kernel 实现
├── mat_mul_iterbatch_broadcast.conf        # 参数路由配置
├── mat_mul_iterbatch_broadcast.csv         # CSV 测试用例
└── README.md                               # 本文档
```

构建配置在 op 层 `examples/acl/batch_mat_mul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/batch_mat_mul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件：

| 组件            | 头文件                                                          | 职责                                                 |
| --------------- | --------------------------------------------------------------- | ---------------------------------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_batch_matmul_iterbatch_broadcast.h` | 完整 kernel 入口（IterBatch Broadcast 专用）         |
| Block MMAD      | `blaze/gemm/block/block_mmad_iterbatch_broadcast.h`           | Block 级矩阵乘（含 iterbatch 流水）                  |
| Block Scheduler | `blaze/gemm/block/block_scheduler_iterbatch_broadcast.h`      | IterBatch 调度器（L1/L0 batch 流水）                 |
| Epilogue        | `blaze/epilogue/block/block_epilogue_empty.h`                 | 后处理（空）                                         |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`                         | 派发策略（`MatmulIterBatchBroadcast<A_BC, B_BC>`） |
