# 权重反量化矩阵乘样例

## 概述

本样例使用 WQMM `GemmUniversal`，由 AIV 对 INT8 权重 `weight` 反量化，
再由 AIC 计算输入 `x` 与反量化权重的矩阵乘，写回输出 `y`。
可选参数包括反量化偏移 `antiquantOffset` 和输出偏置 `bias`。

```text
weightDequant = (cast(weight) + antiquantOffset) * antiquantScale
            y = x @ weightDequant + bias
```

`x`、`weight`、`y` 的逻辑形状分别为 `(M,K)`、`(K,N)`、`(M,N)`。
未启用偏移或偏置时省略对应加法。转换、反量化加法和乘法分别按 FP16/BF16 舍入，
矩阵乘使用 FP32 累加，输出转换为输入类型。

## 支持架构

Ascend 950 / DAV_3510，使用设备 0。

## 使用约束

- 输入 `x` 和输出 `y` 为相同的 FP16 或 BF16 类型；权重固定为 INT8。
  `antiquantScale`、`antiquantOffset` 和 `bias` 与输入类型相同。
- M、K、N 均为正，样例限制 `M <= 256`、`K <= 4096`、`N <= 1024`。
  这些是样例的配置范围，组件的类型与容量约束见 [Kernel 文档](../../../../docs/API/gemm/kernel/kernel_wqmm_mix_antiquant.md)。
- 样例采用固定分块：M/N/K 基本块为 32/64/64，输入和权重的 L1 K 分块长度均为 128，
  UB 输入行距为 512 字节，配置两份输入缓冲区和两个 AIC。分块是每次处理的子区域，
  缓冲区是其片上存储空间，行距是相邻物理行起点间的距离。
- 完整 N 轮次分配给两个 AIC，剩余 N 拆成两段尾块。M 超过 32 时分多轮处理，
  K 超过 128 时交替使用两份 L1 权重缓冲区。样例关闭输入预取，采用普通双缓冲装载输入。
- 尾块搬运包含对齐所需的填充空间，矩阵乘使用真实有效长度。

组件还支持 FP8 E4M3、HiFloat8 权重及 FP32 `bias`，这些类型不属于本样例的验证范围。

## 执行命令

在仓库根目录执行，先配置 CANN 环境，并安装 [examples 依赖](../../../requirements.txt)：

```bash
source /path/to/cann/set_env.sh
bash build.sh --examples --ops=weight_quant_batch_matmul
```

也可使用公共样例入口，仅运行指定样例：

```bash
bash examples/acl/common/run.sh --ops=weight_quant_batch_matmul --target=weight_quant_batch_matmul_antiquant
```

仅编译时添加 `--build-only`；使用已编译的可执行文件时添加 `--skip-build`。
公共入口按 CSV 逐条生成输入、运行 Kernel 并验证输出。

## CSV 参数

用例配置见 [weight_quant_batch_matmul_antiquant.csv](./weight_quant_batch_matmul_antiquant.csv)。

| 字段 | 含义与取值 |
| --- | --- |
| `casename` | 用例名称 |
| `m,k,n` | M、K、N 逻辑长度，均以元素计 |
| `x_dtype` | `fp16` 或 `bf16` |
| `transpose_weight` | 1：GM 权重按 `(N,K)` 存储；0：按 `(K,N)` 存储 |
| `antiquant_per_tensor` | 1：整张权重使用一个反量化缩放值和可选偏移值；0：每个 N 通道各使用一组参数 |
| `has_antiquant_offset` | 1：启用 `antiquantOffset`；0：关闭 |
| `has_bias` | 1：启用 `bias`；0：关闭 |

[weight_quant_batch_matmul_antiquant.conf](./weight_quant_batch_matmul_antiquant.conf)
为公共运行入口提供参数映射：`[gen_data]` 将 CSV 字段映射为数据生成脚本的命名参数，
`[kernel]` 指定可执行文件的位置参数顺序，`[verify]` 指定预期结果、设备结果及数据类型。
`$OUTPUT_DIR` 由运行入口提供，作为当前用例的数据目录。

## 输入输出文件

以下二进制文件存放在当前用例的数据目录中。表中的形状表示元素排列，无额外文件头。

| 文件 | 内容与形状 |
| --- | --- |
| `a.bin` | 输入 `x`，`(M,K)` |
| `b.bin` | INT8 权重 `weight`，按 `transpose_weight` 选择 `(N,K)` 或 `(K,N)` 存储 |
| `scale.bin` | `antiquantScale`，per-tensor 为 1 个元素，per-channel 为 N 个元素 |
| `offset.bin` | `antiquantOffset`，形状与 `antiquantScale` 相同；启用偏移时读取 |
| `bias.bin` | `bias`，N 个元素；启用偏置时读取 |
| `golden_c.bin` | CPU 生成的预期输出 `y`，`(M,N)` |
| `npu_out.bin` | 设备计算的输出 `y`，`(M,N)` |

数据生成脚本固定随机种子为 20260918，并生成可选参数文件；Kernel 仅在对应开关启用时读取。
验证脚本在数据目录写入 `verify_metrics.json`，公共入口汇总生成
`weight_quant_batch_matmul_antiquant_result.csv`。临时数据目录的保留方式由公共入口管理。

## 验证标准

CSV 中的 8 个用例覆盖 FP16/BF16、两种权重存储方向、per-channel/per-tensor、
偏移和偏置开关，以及 M/N/K 尾块和多轮分块。
CPU 预期结果逐步模拟反量化舍入，再执行 FP32 矩阵乘及输出类型转换。

每个输出元素均须满足：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

FP16 的 `atol`、`rtol` 均为 0.001；BF16 均为 0.008。
非有限值、空输出或长度错误均判定为失败。本样例验证数值结果，不采集性能数据。
