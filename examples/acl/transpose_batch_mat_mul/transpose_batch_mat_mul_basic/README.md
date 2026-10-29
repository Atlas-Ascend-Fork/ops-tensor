# transpose_batch_mat_mul_basic Example

## 概述

本示例演示基于 Blaze 框架的 TransposeBatchMatMul 算子在昇腾 NPU 上的实现。该算子执行批量矩阵乘法，输出 C 以转置 batch 方式存储（`[m, batch, n]`），输入 A 可选转置 batch 存储（`[m, batch, k]`）。

- **算子**: transpose_batch_mat_mul
- **场景**: transpose_batch_mat_mul_basic
- **算法特点**: 支持 batch 维度，C 固定转置 batch 存储，A 可选转置 batch 存储
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_tbmm_basic.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape:
  - 标准 batch: `[batch, m, k]`（transBatchA=false）
  - 转置 batch: `[m, batch, k]`（transBatchA=true）
- 输入 B shape: `[batch, k, n]`
- 输出 C shape: `[m, batch, n]`（始终转置 batch）
- 数据类型: float16, bfloat16, float32
- bias: 不支持（本算子使用空 Epilogue，不接收 bias 输入）

## 数据布局说明

TransposeBatchMatMul 的核心特点是 C 矩阵以转置 batch 方式存储：

```
标准 BatchMatMul:  C[batch, m, n]  → batch 在最外层
TransposeBatchMatMul: C[m, batch, n]  → batch 在中间层（转置）
```

**A 矩阵布局**:
- `transBatchA=false`: A 物理存储为 `[batch, m, k]`（标准 batch 布局）
- `transBatchA=true`: A 物理存储为 `[m, batch, k]`（转置 batch 布局）

**C 矩阵布局**:
- 无论 transBatchA 取值，C 始终物理存储为 `[m, batch, n]`（转置 batch 布局）

**计算公式**:
```
C[m, b, n] = Σ_k A_logical[b, m, k] × B[b, k, n]
```

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=transpose_batch_mat_mul --target=transpose_batch_mat_mul_basic
```

### 测试用例定义

测试用例定义在 `transpose_batch_mat_mul_basic.csv` 中，格式如下：

```csv
casename,m,k,n,batch,dtype,trans_batch_a,hf32
tbmm_basic_fp16_batch2,128,128,128,2,float16,false,false
tbmm_basic_bf16_batch4,128,128,128,4,bfloat16,false,false
tbmm_basic_fp32_batch2,128,128,128,2,float32,false,false
tbmm_basic_fp16_trans_batch_a,128,128,128,2,float16,true,false
tbmm_basic_bf16_trans_batch_a,128,128,128,4,bfloat16,true,false
tbmm_basic_fp32_trans_batch_a,128,128,128,2,float32,true,false
tbmm_basic_fp32_hf32,128,128,128,2,float32,false,true
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| batch | batch 维度大小 |
| dtype | 数据类型：float16 / bfloat16 / float32 |
| trans_batch_a | A 矩阵是否使用转置 batch 布局 |
| hf32 | 是否启用 HF32 模式（仅 float32 有效） |

### 结果输出

执行完成后结果写入 `transpose_batch_mat_mul_basic_result.csv`。

## 数据与校验

### 输入数据

由 `../scripts/gen_data.py` 生成:

- `input/input_a.bin`: A 矩阵（布局取决于 transBatchA）
- `input/input_b.bin`: B 矩阵 `[batch, k, n]`
- `output/cpu_output.bin`: CPU 参考结果 `[m, batch, n]`

### 输出数据

- `output/npu_out.bin`: NPU 计算结果 `[m, batch, n]`

### 验证标准

由 `../scripts/verify_result.py` 执行:

| dtype | ratio_tol |
|-------|-----------|
| float16 | 5e-3 |
| bfloat16 | 5e-3 |
| float32 | 1e-4 |
| hf32 | 1e-3 |

## 代码结构

```
transpose_batch_mat_mul_basic/
├── transpose_batch_mat_mul_basic.cpp           # kernel 实现
├── transpose_batch_mat_mul_basic.conf          # 参数路由配置
├── transpose_batch_mat_mul_basic.csv           # CSV 测试用例
└── README.md                                   # 本文档
```

构建配置在 op 层 `examples/acl/transpose_batch_mat_mul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/transpose_batch_mat_mul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_tbmm_basic.h` | 完整 kernel 入口（TBMM 专用） |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_basic.h` | Block 级矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_basic.h` | 基础调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_empty.h` | 后处理（空） |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | 派发策略（KernelMmadMultiBlockTBMM） |
