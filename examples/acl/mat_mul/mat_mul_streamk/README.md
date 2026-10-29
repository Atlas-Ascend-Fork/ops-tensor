# mat_mul_streamk Example

## 概述

本示例演示基于 Blaze 框架的 MatMul 矩阵乘法算子在昇腾 NPU 上的实现。StreamK 算法通过动态负载均衡策略，将矩阵乘计算任务智能分配到多个 NPU 核心，适用于各种形状的矩阵乘法场景。

- **算子**: mat_mul
- **场景**: mat_mul_streamk
- **算法特点**: StreamK 动态负载均衡，支持 FP16/BF16/FP32/HF32 及 Weight NZ 格式
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_streamk.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true）
- 输出 C shape: `[M, N]`
- 数据类型: float16, bfloat16, float32
- Weight NZ 场景: B 矩阵需使用 NZ 格式存储，仅支持 float16/bfloat16

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=mat_mul --target=mat_mul_streamk
```

### 测试用例定义

测试用例定义在 `mat_mul_streamk.csv` 中，格式如下：

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,layoutA,layoutB
mat_mul_streamk_fp16,100,8192,100,100,float16,false,false,false,ND,ND
mat_mul_streamk_bf16,100,8192,100,100,bfloat16,false,false,false,ND,ND
mat_mul_streamk_fp32,100,8192,100,100,float32,false,false,false,ND,ND
mat_mul_streamk_hf32,100,8192,100,100,float32,false,false,true,ND,ND
mat_mul_streamk_weightNz,100,8192,100,0,float16,false,false,false,ND,NZ
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 向量大小，必须等于 n 或 0（无 bias） |
| dtype | 数据类型：float16 / bfloat16 / float32 |
| transA | A 矩阵是否转置 |
| transB | B 矩阵是否转置 |
| hf32 | 是否启用 HF32 模式（仅 float32 有效） |
| layoutA, layoutB | 输入格式：`ND,ND` 或 `ND,NZ` |

### layout 说明

- `layoutA=ND, layoutB=ND`：标准 matmul，A 和 B 均为 ND 格式
- `layoutA=ND, layoutB=NZ`：weightNz 场景，A 为 ND 格式，B 为 NZ 格式（由 C++ 端自动转换）

### 结果输出

执行完成后结果写入 `mat_mul_streamk_result.csv`。

## 数据与校验

### 输入数据

由 `../scripts/gen_data.py` 生成:

- `input/input_a.bin`: A 矩阵
- `input/input_b.bin`: B 矩阵
- `input/bias.bin`: bias 向量（bias>0 时生成）
- `output/cpu_output.bin`: CPU 参考结果

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `../scripts/verify_result.py` 执行:

| dtype | ratio_tol | error_ratio_tol |
|-------|-----------|-----------------|
| float16 | 5e-3 | 5e-3 |
| bfloat16 | 5e-3 | 5e-3 |
| float32 | 1e-4 | 1e-4 |
| hf32 | 1e-3 | 1e-3 |

- 超差比例 < error_ratio_tol

## Bias 支持

mat_mul_streamk 支持 bias 功能。bias 是一个 1D 向量，大小必须等于 n 或 0。

**数据流**：
1. `gen_data.py` 生成 `input/bias.bin`（n 个元素）
2. `gen_data.py` 生成 `output/cpu_output.bin`（已包含 bias：`C = A @ B + bias`）
3. C++ kernel 读取 `input/bias.bin`，在计算时应用 bias
4. `verify_result.py` 比较 NPU 输出和 CPU golden（两者都已包含 bias）

**约束**：
- bias 必须等于 n 或 0（无 bias）
- bias 数据类型与 dtype 一致

## weightNz 场景

weightNz 场景表示 weight 矩阵（B 矩阵）使用 NZ 格式存储，优化 NPU 内存访问模式。

**格式说明**：
- A 矩阵：ND 格式 [M, K]
- B 矩阵：NZ 格式（16x16 分块存储）
- C 矩阵：ND 格式 [M, N]

**Layout 对应关系**：
- `(ND,ND)` 格式：
  - transB=true → B 使用 dn_ext_layout_ptn
  - transB=false → B 使用 nd_ext_layout_ptn
- `(ND,NZ)` 格式（weightNz 场景）：
  - transB=true → B 使用 zn_layout_ptn
  - transB=false → B 使用 nz_layout_ptn

**数据流**：
1. `gen_data.py` 生成标准 ND 格式的 `input_b.bin`
2. C++ 端读取后，自动转换为 NZ 格式
3. Kernel 根据 transB 选择对应的 NZ Layout（zn_layout_ptn 或 nz_layout_ptn）

**约束**：
- layoutB 必须为 `NZ`
- 仅支持 float16/bfloat16

## 代码结构

```
mat_mul_streamk/
├── mat_mul_streamk.cpp              # kernel 实现
├── mat_mul_streamk.conf             # 参数路由配置
├── mat_mul_streamk.csv              # CSV 测试用例
└── README.md                        # 本文档
```

构建配置在 op 层 `examples/acl/mat_mul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/mat_mul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_streamk.h` | 完整 kernel 入口 |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_streamk.h` | Block 级矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_streamk.h` | StreamK 调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_matmul_streamk.h` | 后处理 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | 派发策略 |
