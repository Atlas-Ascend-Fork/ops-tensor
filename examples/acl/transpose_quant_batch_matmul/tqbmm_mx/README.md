# tqbmm_mx Example

## 概述

本示例演示基于 Blaze 框架的 Transpose MX 量化 Batch MatMul 算子在昇腾 NPU 上的实现。本样例对 A、B 矩阵进行 MX 量化（mxFP8/mxFP4 激活 × mxFP8/mxFP4 权重），两侧各携带独立的 E8M0 MX Scale，并支持 Transpose 场景。

- **算子**: transpose_quant_batch_matmul
- **场景**: tqbmm_mx
- **算法特点**: A/B 双侧 MX 量化，仅支持 mxFP8（fp8_e4m3）和 mxFP4（fp4_e2m1），支持 Transpose Batch
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_tqbmm.h` + `blaze/gemm/block/block_mmad_qbmm_mx.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[M, Batch, K]`（transA=false）或 `[K, Batch, M]`（transA=true）
- 输入 B shape: `[K, Batch, N]`（transB=false）或 `[N, Batch, K]`（transB=true）
- 输出 C shape: `[M, Batch, N]`
- A/B dtype: fp8_e4m3, fp4_e2m1（A 和 B 必须为相同类型）
- C dtype: float16, bfloat16
- ScaleA shape: `[M, scaleK]`，ScaleB shape: `[scaleK, N]`，`scaleK = ceil(K/64) * 2`

## CSV 驱动测试

### 执行方式

本样例目录下不包含独立的运行脚本，统一使用 `examples/acl/common/run.sh` 入口执行，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=transpose_quant_batch_matmul --target=tqbmm_mx
```

如需仅运行部分用例，可通过 `--ti` 指定用例索引：

```bash
bash examples/acl/common/run.sh --ops=transpose_quant_batch_matmul --target=tqbmm_mx --ti=0-3
```

### 测试用例定义

测试用例定义在 `tqbmm_mx.csv` 中，格式如下：

```csv
casename,m,k,n,batch,a_dtype,b_dtype,c_dtype,transA,transB,format,base_m,base_n,base_k,tile_k_l1,scale_k_l1,l1_buffers,db_l0c,a_full_load
tqbmm_mx_32_512_128_4_fp8_bf16,32,512,128,4,fp8_e4m3,fp8_e4m3,bfloat16,false,false,"(ND,ND)",32,128,64,128,16,2,1,false
tqbmm_mx_64_1024_256_2_fp4_f16,64,1024,256,2,fp4_e2m1,fp4_e2m1,float16,false,false,"(ND,ND)",64,256,64,256,32,2,1,false
```

**列说明**：

| 列                     | 说明                                 |
| ---------------------- | ------------------------------------ |
| casename               | 用例名称                             |
| m, k, n                | 矩阵维度                             |
| batch                  | batch 维度                           |
| a_dtype, b_dtype       | A/B 量化 dtype（必须相同）           |
| c_dtype                | 输出 dtype                           |
| transA, transB         | A/B 矩阵是否转置                     |
| format                 | 输入格式: (ND,ND)                    |
| base_m, base_n, base_k | L0 tile shape                        |
| tile_k_l1              | L1 K 方向 tile 大小                  |
| scale_k_l1             | L1 scaleK 方向 tile 大小             |
| l1_buffers             | L1 buffer 数                         |
| db_l0c                 | L0C double buffer 开关（1=关，2=开） |
| a_full_load            | A 全载到 L1（true/false）            |

### 结果输出

执行完成后结果写入 `tqbmm_mx_result.csv`。

## 数据与校验

### 输入数据

由 `examples/acl/transpose_quant_batch_matmul/scripts/gen_data.py` 生成：

- `input/input_a.bin`: A 矩阵（FP8 或打包 FP4）
- `input/input_b.bin`: B 矩阵（FP8 或打包 FP4）
- `input/scale_a.bin`: ScaleA（E8M0）
- `input/scale_b.bin`: ScaleB（E8M0）
- `input/initial_c.bin`: 初始化 C（全零）

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `examples/acl/transpose_quant_batch_matmul/scripts/verify_result.py` 执行，支持 float16/bfloat16 两种输出 dtype 的尺寸与结构校验。

## 覆盖的场景

| 场景维度  | 用例                             |
| --------- | -------------------------------- |
| A/B dtype | mxFP8(fp8_e4m3), mxFP4(fp4_e2m1) |
| C dtype   | float16, bfloat16                |
| Batch     | 2, 4, 8, 16                      |

## 代码结构

```
tqbmm_mx/
├── tqbmm_mx.cpp       # kernel 实现
├── tqbmm_mx.conf      # 参数路由配置
├── tqbmm_mx.csv       # CSV 测试用例
└── README.md          # 本文档
```

构建配置在 op 层 `examples/acl/transpose_quant_batch_matmul/CMakeLists.txt` 中统一管理；运行统一通过 `examples/acl/common/run.sh` 调度（本目录下不含独立运行脚本）。数据生成和精度校验由 `examples/acl/transpose_quant_batch_matmul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件            | 头文件                                          | 职责                            |
| --------------- | ----------------------------------------------- | ------------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_tqbmm.h`         | Transpose MX quant batch matmul |
| Block MMAD      | `blaze/gemm/block/block_mmad_qbmm_mx.h`       | Block 级 MX 矩阵乘              |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h`     | QBMM V3 调度器                  |
| Epilogue        | `blaze/epilogue/block/block_epilogue_empty.h` | 空 epilogue                     |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`         | MatmulWithScaleMx 派发策略      |
