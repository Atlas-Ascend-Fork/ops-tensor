# quant_batch_matmul_mx Example

## 概述

本示例演示基于 Blaze 框架的 MX 量化 Batch MatMul 算子在昇腾 NPU 上的实现。本样例对 A、B
矩阵进行 MX 量化（FP8/FP4 激活 × FP8/FP4 权重），两侧各携带独立的 E8M0 MX Scale。
`kernel_variant` 可选择 `batch`、`without_batch`、`l0c_pingpong` 或 `streamk`。这些变体共享
输入数据格式、CSV 配置和校验脚本。

- **算子**: quant_batch_matmul
- **场景**: quant_batch_matmul_mx
- **算法特点**: A/B 双侧 MX 量化，支持 FP8×FP8、FP4×FP4，多种 C dtype
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_qbmm_mx.h` + `blaze/gemm/block/block_mmad_qbmm_mx.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true）
- 输出 C shape: `[M, N]`
- A/B dtype: fp8_e4m3, fp8_e5m2, fp4_e2m1（A 和 B 可为不同 FP8 子类型，但 FP4 与 FP8 不可混用）
- C dtype: float16, bfloat16, float32
- Bias dtype: float32
- ScaleA shape: `[M, scaleK]`，ScaleB shape: `[scaleK, N]`，`scaleK = ceil(K/64) * 2`
- `l0c_pingpong` 固定为 FP8 E4M3×E4M3→FP16、ND/ND、无转置/无 Bias，要求 `db_l0c=2`；
- `streamk` 固定为 FP8 E4M3×E5M2→FP16、ND/ND、无转置/无 Bias，要求 K 对齐 64，并显式分配
  与清零 StreamK workspace。

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=quant_batch_matmul --target=quant_batch_matmul_mx
```

### 测试用例定义

测试用例定义在 `quant_batch_matmul_mx.csv` 中，格式如下：

```csv
casename,m,k,n,bias,a_dtype,b_dtype,c_dtype,transA,transB,format,base_m,base_n,base_k,tile_k_l1,scale_k_l1,l1_buffers,db_l0c,a_full_load,kernel_variant
qbmm_mx_without_batch,16,128,16,0,fp8_e4m3,fp8_e4m3,float16,false,false,"(ND,ND)",16,32,64,64,64,2,1,false,without_batch
qbmm_mx_l0c_pingpong,64,128,128,0,fp8_e4m3,fp8_e4m3,float16,false,false,"(ND,ND)",64,128,64,64,64,2,2,false,l0c_pingpong
qbmm_mx_streamk,16,128,16,0,fp8_e4m3,fp8_e5m2,float16,false,false,"(ND,ND)",16,32,64,64,64,2,1,false,streamk
qbmm_mx_10240_1024_2624_FF_fp8e4m3_fp8e4m3_bfloat16_NZ,10240,1024,2624,0,fp8_e4m3,fp8_e4m3,bfloat16,false,false,"(ND,NZ)",256,256,128,256,1024,3,1,false,batch
```

**列说明**：

| 列                     | 说明                                      |
| ---------------------- | ----------------------------------------- |
| casename               | 用例名称                                  |
| m, k, n                | 矩阵维度                                  |
| bias                   | bias 向量大小，必须等于 n 或 0（无 bias） |
| a_dtype, b_dtype       | A/B 量化 dtype                            |
| c_dtype                | 输出 dtype                                |
| transA                 | A 矩阵是否转置                            |
| transB                 | B 矩阵是否转置                            |
| format                 | 输入格式: (ND,ND) 或 (ND,NZ)             |
| base_m, base_n, base_k | L0 tile shape                             |
| tile_k_l1              | L1 K 方向 tile 大小                       |
| scale_k_l1             | L1 scaleK 方向 tile 大小                  |
| l1_buffers             | L1 buffer 数（2/3/4）                     |
| db_l0c                 | L0C double buffer 开关（1=关，2=开）      |
| a_full_load            | A 全载到 L1（true/false）                 |
| kernel_variant         | `batch`、`without_batch`、`l0c_pingpong` 或 `streamk` |

当前样例为 `without_batch` 和 `l0c_pingpong` 预编译 FP8 E4M3 × FP8 E4M3、FP16 输出、ND/ND、
无转置、无 Bias 的固定组合；`streamk` 预编译 FP8 E4M3 × FP8 E5M2、FP16 输出、ND/ND、
无转置、无 Bias 的固定组合。`batch` 路径覆盖下表中的完整组合。

### 结果输出

执行完成后结果写入 `quant_batch_matmul_mx_result.csv`。

## 数据与校验

### 输入数据

由 `examples/acl/quant_batch_matmul/scripts/gen_data_mx.py` 生成：

- `input/input_a.bin`: A 矩阵（FP8 或打包 FP4）
- `input/input_b.bin`: B 矩阵（FP8 或打包 FP4，ND 或 NZ 格式）
- `input/scale_a.bin`: ScaleA（E8M0，布局随 transA 变化）
- `input/scale_b.bin`: ScaleB（E8M0，布局随 transB 变化）
- `input/bias.bin`: bias 向量（FP32，bias>0 时生成）
- `input/initial_c.bin`: 初始化 C（全零）

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `examples/acl/quant_batch_matmul/scripts/verify_result_mx.py` 执行，支持 float16/bfloat16/float32 三种输出 dtype 的精度比对。

## 覆盖的场景

| 场景维度             | 用例                                           |
| -------------------- | ---------------------------------------------- |
| A/B dtype            | FP8(e4m3,e5m2)×FP8(e4m3,e5m2) , FP4×FP4 |
| C dtype              | float16/float32/bfloat16                       |
| transA/transB        | FF/FT/TF/TT 四种组合（F=不转置，T=转置）       |
| format               | (ND,ND) / (ND,NZ)                             |
| L1 Buffer 数         | 2 (Double), 3 (Triple), 4 (Quad)               |
| A Full Load          | 开/关                                          |
| dbL0C (L0C PingPong) | 1/2                                            |
| K 维 L1 多次循环     | kL1 < K                                        |
| ScaleK 多次循环      | scaleKL1 > kL1                                 |
| 尾块处理             | M/N/K 尾块非对齐                               |
| 多核                 | 大 shape 利用全部 AIC core                     |
| Bias                 | 有/无                                          |
| Kernel 变体          | batch（支持 Batch 广播）/ without-batch / L0C ping-pong / StreamK |

## 代码结构

```
quant_batch_matmul_mx/
├── quant_batch_matmul_mx.cpp       # kernel 实现
├── quant_batch_matmul_mx.conf      # 参数路由配置
├── quant_batch_matmul_mx.csv       # CSV 测试用例
└── README.md                       # 本文档
```

构建配置在 op 层 `examples/acl/quant_batch_matmul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度。数据生成和精度校验由 `examples/acl/quant_batch_matmul/scripts/` 下的 `gen_data_mx.py` 和 `verify_result_mx.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件            | 头文件                                          | 职责                         |
| --------------- | ----------------------------------------------- | ---------------------------- |
| Kernel          | `kernel_qbmm_mx.h`、`kernel_qbmm_mx_without_batch.h`、`kernel_qbmm_streamk.h` | MX kernel 实现 |
| Block MMAD      | `block_mmad_qbmm_mx.h`、`block_mmad_qbmm_mx_l0c_pingpong.h` | Block 级 MX 矩阵乘 |
| Block Scheduler | `block_scheduler_qbmm.h`、`block_scheduler_matmul_streamk.h` | QBMM V3 / StreamK 调度器 |
| Epilogue        | `block_epilogue_empty.h`、`block_epilogue_matmul_streamk.h` | 常规路径无额外处理；StreamK 路径负责归约与输出 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`         | MatmulWithScaleMx 派发策略   |
