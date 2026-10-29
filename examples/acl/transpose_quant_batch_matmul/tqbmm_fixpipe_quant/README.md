# tqbmm_fixpipe_quant Example

## 概述

本示例演示基于 Blaze 框架的 Transpose HIFP8 量化 Batch MatMul 算子在昇腾 NPU 上的实现。本样例对 A、B 矩阵使用 HIFP8（hifloat8）量化，B 侧携带 uint64 per-channel Scale，通过 fixpipe 在 Cube 侧完成反量化，支持 Transpose 场景。

- **算子**: transpose_quant_batch_mat_mul
- **场景**: tqbmm_fixpipe_quant（HIFP8）
- **算法特点**: A/B 双侧 HIFP8 量化（hifloat8），x2Scale 为 uint64 per-channel 编码，AIC-only fixpipe dequant
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_tqbmm.h` + `blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[M, Batch, K]`（perm_x1=[1,0,2]，M 轴非连续）
- 输入 B shape: `[Batch, K, N]`（transB=false）或 `[Batch, N, K]`（transB=true）
- 输出 C shape: `[M, Batch, N]`
- A/B dtype: hifloat8（A 和 B 必须为相同类型）
- C dtype: float16, bfloat16
- x2Scale shape: `[N]`，dtype 为 uint64（per-channel，编码格式为 fixpipe 64-bit scale）
- x1Scale: 可选，per-tensor float32 scale（本示例中为 None）

## CSV 驱动测试

### 执行方式

本样例目录下不包含独立的运行脚本，统一使用 `examples/acl/common/run.sh` 入口执行，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/acl/common/run.sh --ops=transpose_quant_batch_matmul --target=tqbmm_fixpipe_quant
```

如需仅运行部分用例，可通过 `--ti` 指定用例索引：

```bash
bash examples/acl/common/run.sh --ops=transpose_quant_batch_matmul --target=tqbmm_fixpipe_quant --ti=0-1
```

### 测试用例定义

测试用例定义在 `tqbmm_fixpipe_quant.csv` 中，格式如下：

```csv
casename,m,k,n,batch,dtype,c_dtype,transB,base_m,base_n,base_k,tile_k_al1,tile_k_bl1,l1_buffers,db_l0c
tqbmm_hifp8_64_128_64_1_fp16,64,128,64,1,hifp8,float16,false,64,64,64,128,128,2,1
tqbmm_hifp8_64_128_128_2_bf16,64,128,128,2,hifp8,bfloat16,false,64,128,64,128,128,2,1
```

**列说明**：

| 列                     | 说明                                      |
| ---------------------- | ----------------------------------------- |
| casename               | 用例名称                                  |
| m, k, n                | 矩阵维度                                  |
| batch                  | batch 维度                                |
| dtype                  | 量化 dtype（固定为 hifp8）                |
| c_dtype                | 输出 dtype（float16 / bfloat16）          |
| transB                 | B 矩阵是否转置                            |
| base_m, base_n, base_k | L0 tile shape                             |
| tile_k_al1             | L1 A 矩阵 K 方向 tile 大小                |
| tile_k_bl1             | L1 B 矩阵 K 方向 tile 大小                |
| l1_buffers             | L1 buffer 数                              |
| db_l0c                 | L0C double buffer 开关（1=关，2=开）      |

### 结果输出

执行完成后结果写入 `tqbmm_fixpipe_quant_result.csv`。

## 数据与校验

### 输入数据

由 `examples/acl/transpose_quant_batch_matmul/scripts/gen_data.py` 生成（`--dtype=hifp8`）：

- `input/input_a.bin`: A 矩阵（hifloat8，每元素 1 字节）
- `input/input_b.bin`: B 矩阵（hifloat8，每元素 1 字节）
- `input/scale.bin`: x2Scale（uint64 per-channel，编码 1.0f = 0x3F80000000000000）
- `input/initial_c.bin`: 初始化 C（全零）

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `examples/acl/transpose_quant_batch_matmul/scripts/verify_result.py` 执行，支持 float16/bfloat16 两种输出 dtype 的尺寸与结构校验。

## 覆盖的场景

| 场景维度     | 用例                           |
| ------------ | ------------------------------ |
| A/B dtype    | hifloat8                       |
| C dtype      | float16, bfloat16             |
| Batch        | 1, 2, 4                        |
| K            | 128, 256, 512                  |

## 代码结构

```
tqbmm_fixpipe_quant/
├── tqbmm_fixpipe_quant.cpp       # kernel 实现
├── tqbmm_fixpipe_quant.conf      # 参数路由配置
├── tqbmm_fixpipe_quant.csv       # CSV 测试用例
└── README.md                     # 本文档
```

构建配置在 op 层 `examples/acl/transpose_quant_batch_matmul/CMakeLists.txt` 中统一管理；运行统一通过 `examples/acl/common/run.sh` 调度。数据生成和精度校验由 `examples/acl/transpose_quant_batch_matmul/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件            | 头文件                                                   | 职责                              |
| --------------- | -------------------------------------------------------- | --------------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_tqbmm.h`        | Transpose HIFP8 fixpipe quant BMM |
| Block MMAD      | `blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h`      | Block 级 fixpipe dequant 矩阵乘   |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h`               | QBMM V3 调度器                    |
| Epilogue        | `blaze/epilogue/block/block_epilogue_empty.h`           | 空 epilogue（dequant 在 fixpipe） |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`                   | MatmulWithScaleFixpipeQuant 策略  |

## 与 tqbmm_mx 的区别

| 项目              | tqbmm_mx                         | tqbmm_fixpipe_quant             |
| ----------------- | -------------------------------- | ------------------------------- |
| A/B dtype         | fp8_e4m3 / fp4_e2m1              | hifloat8                        |
| Scale 类型        | E8M0（MX 分组量化）              | uint64（per-channel fixpipe）   |
| Scale shape       | [M, B, scaleK] / [scaleK, N]     | [N]                             |
| 计算模式          | AIC-only                         | AIC-only                        |
| 反量化位置        | Cube MMAD 内联                   | Cube fixpipe                    |
| ScheduleType      | KernelMmadMultiBlockTQBMM        | KernelMmadMultiBlockTQBMMFixpipeQuant |
| Dispatch Policy   | MatmulWithScaleMx               | MatmulWithScaleFixpipeQuant     |
