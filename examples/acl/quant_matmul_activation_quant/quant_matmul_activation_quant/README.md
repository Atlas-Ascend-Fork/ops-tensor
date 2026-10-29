# quant_matmul_activation_quant Example

## 概述

本示例基于 Blaze 融合 MX 量化矩阵乘、GELU/SwiGLU/Swish 激活和动态 MX 量化。
AIC 完成矩阵乘并通过 DualDst Fixpipe 写入 UB，AIV 完成激活和量化。
CSV 的 `batch`/`without_batch` 均使用同一 Kernel，本示例的 batch 维度固定为 1。

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true），支持 NZ 和 ND 两种布局
- GELU/Swish 输出 Y shape 为 `[M, N]`；Swish 对完整矩阵乘结果加 bias 后逐元素计算 `x / (1 + exp(-β*x))`。SwiGLU 按 `C=[gate | linear]` 等分并计算 `SiLU(gate) * linear`，输出 Y shape 为 `[M, N/2]`
- 令 `outputN=N`（GELU/Swish）或 `outputN=N/2`（SwiGLU）。输出 Y_scale 在样例中的物理 shape 为 `[M, ceil(outputN/64)*2]`，等价于产品接口的 `[M, ceil(outputN/64), 2]`
- GELU/SwiGLU 的 FP8 A/B 组合为 `fp8_e4m3 × fp8_e4m3`、`fp8_e5m2 × fp8_e4m3`；GELU 还支持 `fp4_e2m1 × fp4_e2m1`
- Swish（含 SiLU）的 ND 路径中，A、B 可分别选用 `fp8_e4m3` 或 `fp8_e5m2`，并分别设置是否转置
- Swish 的 NZ 路径中，A 可选 `fp8_e4m3` 或 `fp8_e5m2` 且不转置；B 固定为 `fp8_e4m3`，可设置是否转置
- AIC 矩阵乘累加结果和 epilogue 输入 C 的 dtype 为 `float32`
- 输出 Y dtype 跟随 A dtype（FP4 时 Y 为 `fp4x2_e2m1` 打包格式，2 元素/字节）
- ScaleA/ScaleB/Y_scale dtype: `fp8_e8m0`
- Bias dtype: `float32`
- SwiGLU 要求 `kernel_variant=without_batch`、`transA=false`、FP8 输入且原始 N 为 64 的倍数
- Swish 要求 `base_m` 在 [16, 256] 内且为 16 的倍数，`base_n` 在 [32, 256] 内且为 32 的倍数
- A 全载模式要求 `ceil(M/base_m)` 不超过可用 AIC 核数。示例将启动核数向下对齐到该 M 分块数的整数倍，保证每个核跨 N 轮次复用同一块 A；例如 36 核、5 个 M 分块时启动 35 核。

`activation=swish` 支持标量 `beta`，默认 1；`activation=silu` 固定 β=1，两者使用同一 Swish 组件。

### Scale 算法选择

- `scale_alg=0`（OCP）：根据组内最大指数生成 scale，支持 FP8/FP4 输出。
- `scale_alg=1`（BLAS）：根据组内最大绝对值和目标最大有限值生成 scale，本示例仅支持 FP8 输出。

每 32 个 BF16 激活结果共用一个 scale。两种算法的边界行为见 [MxQuant 算法说明](../../../../docs/API/epilogue/tile/arch35/mx_quant.md#ocp-与-cublas-算法选择)。

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=quant_matmul_activation_quant --target=quant_matmul_activation_quant
```

### 测试用例定义

测试用例定义在 `quant_matmul_activation_quant.csv` 中，格式如下：

```csv
casename,m,k,n,bias,a_dtype,b_dtype,transA,transB,format,base_m,base_n,base_k,tile_k_l1,scale_k_l1,l1_buffers,db_l0c,a_full_load,activation,scale_alg,kernel_variant,beta
without_batch_gelu_ocp,64,128,128,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,NZ)",64,128,64,64,64,2,1,false,gelu,0,without_batch,1
without_batch_swiglu_ocp_mtail,1,64,128,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,NZ)",64,128,64,64,64,2,1,false,swiglu,0,without_batch,1
without_batch_swiglu_blas_transb_bias_afl,64,128,128,128,fp8_e5m2,fp8_e4m3,false,true,"(ND,NZ)",64,128,64,64,64,2,2,true,swiglu,1,without_batch,1
without_batch_swiglu_ocp_nd_multitile,65,128,256,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,ND)",64,128,64,64,64,2,1,false,swiglu,0,without_batch,1
silu_nd_e5e5_tn_n192,65,128,192,192,fp8_e5m2,fp8_e5m2,true,false,"(ND,ND)",32,96,64,128,128,2,1,false,silu,0,without_batch,1
silu_nz_e5e4_nn_n193,3,65,193,193,fp8_e5m2,fp8_e4m3,false,false,"(ND,NZ)",32,96,64,128,128,2,2,true,silu,1,batch,1
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 元素数量，必须为 n 或 0 |
| a_dtype, b_dtype | A/B 量化 dtype；组合与激活、ND/NZ 约束见上文 |
| transA, transB | A/B 矩阵是否转置 |
| format | B 矩阵布局: `(ND,NZ)` 或 `(ND,ND)` |
| base_m, base_n, base_k | Cube 基础分块大小 |
| tile_k_l1 | L1 中 K 方向的 Tile 大小 |
| scale_k_l1 | L1 中 Scale K 方向的 Tile 大小 |
| l1_buffers | L1 Buffer 数量 |
| db_l0c | L0C double buffer 开关（1=关，2=开） |
| a_full_load | A 全载到 L1（true/false） |
| activation | 激活类型：`gelu`、`swiglu`、`silu` 或 `swish` |
| beta | 有限 FP32 标量，默认 1；仅 `swish` 允许取其他值 |
| scale_alg | MX scale 算法：0=OCP，1=BLAS |
| kernel_variant | `batch` 或 `without_batch`；均调用同一 Kernel，batch 维度固定为 1；SwiGLU 仅接受 `without_batch` |

### Scale 行重排用例

`scale_pad_{gelu,silu}_cols{1,3,5}_{rows,ntail}` 共 12 条用例，检查有效 scale 列数为
1、3、5 时的多行重排和尾 tile 补零。所有用例均使用 `base_m=base_n=256`。

| 用例后缀 | M | N | 覆盖边界 |
|----------|---|---|----------|
| `rows` | 255 | 31、95、159 | 两个 AIV 分别处理 128、127 行 |
| `ntail` | 257 | 287、351、415 | N 尾 tile 为 31、95、159 列，M 尾 tile 为 1 行 |

输出缓冲预填 `0xFF`，校验时逐字节比较有效 scale 和每行末尾的配对补零字节。

### Swish 标量 β 用例

`swish_beta_*` 共 8 条用例，覆盖 β=0、0.5、1、2、-1，FP8 E4M3/E5M2、OCP/BLAS、
M=1、奇数 M、N=1/31/32/33/65，以及 M/N 多 tile 和尾 tile。校验 Y、有效 scale 和配对补零字节。

### 结果输出

执行完成后结果写入 `quant_matmul_activation_quant_result.csv`。

## 数据与校验

### 输入数据

由 `examples/acl/quant_matmul_activation_quant/scripts/gen_data.py` 在样例目录的 `input/` 下生成：

- `input_a.bin`: A 矩阵（FP8 或打包 FP4）
- `input_b.bin`: B 矩阵（FP8 或打包 FP4，布局由 CSV 的 format 决定）
- `scale_a.bin`: ScaleA（E8M0，布局随 transA 变化）
- `scale_b.bin`: ScaleB（E8M0，布局随 transB 变化）
- `bias.bin`: bias 向量（FP32，bias>0 时生成）
- `golden_y.bin`: NumPy 计算得到的 Golden 输出（dtype 跟随 A）
- `golden_y_scale.bin`: NumPy 计算得到的 E8M0 Golden Scale

### 输出数据

- `output/npu_y.bin`: NPU 计算得到的量化输出
- `output/npu_y_scale.bin`: NPU 计算得到的 E8M0 输出 Scale

### 验证标准

`verify_result.py` 要求实际输出与 golden 长度相同且非空，并检查：

- Y：所有元素均为有限值，且满足 `abs(actual-golden) <= atol + rtol*abs(golden)`。
- Y_scale：E8M0 原始字节完全相同，包括每行末尾的配对补零字节；不允许 `0xFF`。

任一元素不满足要求即失败。

| dtype | rtol | atol |
|-------|------|------|
| fp8_e4m3 | 1e-3 | 1.0 |
| fp8_e5m2 | 1e-3 | 1.0 |
| fp4_e2m1 | 1e-3 | 1.0 |
| fp8_e8m0 (Y_scale 原始字节) | 0 | 0 |

### C++ UT

```bash
bash build.sh --opkernel -u --ops=quant_matmul_activation_quant
```

CPU-debug UT 检查模板装配，以及 `Init`、`UpdateNextProblem`、
`UpdateGlobalAddr` 的初始化执行，不调用 Epilogue 的 `operator()`。
设备计算、输出精度与尾部写回使用上述 CSV example 验证。

## 代码结构

```text
quant_matmul_activation_quant/
├── quant_matmul_activation_quant.cpp           # kernel 实现
├── quant_matmul_activation_quant.conf          # 参数路由配置
├── quant_matmul_activation_quant.csv           # CSV 测试用例
└── README.md                                   # 本文档
```

构建配置见上一级 `CMakeLists.txt`；数据生成和校验脚本位于上一级 `scripts/` 目录。

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h` | 统一融合 kernel 入口（Batch/单 Batch） |
| Block MMAD | `blaze/gemm/block/block_mmad_qbmm_mx.h` | Block 级 MX 量化矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h` | QBMM 调度器 |
| Epilogue | `block_epilogue_gelu_mx_quant.h`、`block_epilogue_swiglu_mx_quant.h`、`block_epilogue_swish_mx_quant.h` | GELU/SwiGLU/Swish 激活 + 动态 MX 量化 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulWithScaleMx (DualDst) |
