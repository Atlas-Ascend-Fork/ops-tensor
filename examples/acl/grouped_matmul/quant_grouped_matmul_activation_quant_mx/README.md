# Quant Grouped MatMul Activation Quant MX 样例

## 概述

本样例演示 GMMAQ（Grouped MatMul + GeluTanh + MX 动态量化）在 Ascend950
上的 Blaze 模板调用流程，并通过 CSV 用例完成输入生成、Kernel 执行和输出校验。

计算流程为：

```text
Grouped MX MatMul -> GeluTanh 激活 -> MX 动态量化 -> Y/YScale 输出
```

样例使用固定的、容易阅读的 tiling 参数。`baseM/baseN/baseK`、L1 切分、L0C
双缓冲和输出量化参数直接由 CSV 传入，Host 侧只计算输入文件的字节数和地址，
不在示例中重新实现完整的 tiling 搜索。

## 支持场景

当前样例覆盖以下 MXFP8/MXFP4 场景：

- MXFP8 E4M3 输入和 WeightNZ，`scaleAlg=1`，输出 MXFP8 E4M3；
- MXFP4 E2M1 输入和 WeightNZ，`scaleAlg=2`，输出 MXFP4 E2M1；
- 单 group、双 group、四 group 的 M 轴分组；
- `dbL0C=1/2`；
- `l1BufferStage=2/3`；
- MXFP4 `dstTypeMax=0/6/9/12`；
- 小 M、小 N、尾 M 以及较大 N 的固定 NZ 测试形状。

样例固定使用以下布局和参数，生产代码及 Kernel UT 覆盖更多组合：

- A 使用 ND 布局，按 M 轴分组（`groupType=0`）；
- Weight 使用单张量 FRACTAL_NZ 布局（`singleW=1`）；
- `groupListType=1`，CSV 中的 `groupList` 填每个 group 的 M 长度；
- 激活函数固定为 `gelu_tanh`，偏置关闭；
- K 为 64，MXFP8 的 N 为 32 的倍数，MXFP4 的 N 为 64 的倍数；
- 为匹配当前 Ascend950 Blaze tile 约束，`baseM` 使用不小于 16 的值；
- 当前数据生成脚本采用全零 payload 和 E8M0 scale=1，便于对比 Y/YScale 的字节级 golden。

## CSV 字段

测试用例定义在 `quant_grouped_matmul_activation_quant_mx.csv` 中：

| 字段 | 含义 |
| --- | --- |
| `caseName` | 用例名称 |
| `groupNum` | group 数量 |
| `m/n/k` | 每个 group 的矩阵 M/N/K |
| `baseM/baseN/baseK` | L0 计算块大小 |
| `kAL1/kBL1` | A、B 在 L1 中的 K 切分 |
| `scaleKAL1/scaleKBL1` | ScaleA、ScaleB 在 L1 中的 K 切分 |
| `dbL0C` | L0C buffer 数量，支持 1 或 2 |
| `l1BufferStage` | L1 buffer 数量，支持 2 或 3 |
| `dtype` | `mxfp8_e4m3` 或 `mxfp4_e2m1` |
| `scaleAlg` | MX 量化算法，FP8 使用 1，FP4 E2M1 使用 2 |
| `dstTypeMax` | FP4 动态量化的目标类型最大值，样例覆盖 0、6、9、12 |
| `groupList` | Length 类型的 group 列表，以分号分隔 |

## 编译和运行

在仓库根目录执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_activation_quant_mx
```

仅编译：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_activation_quant_mx --build-only
```

跳过编译直接执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_activation_quant_mx --skip-build
```

运行器会按 CSV 逐条执行：

1. `scripts/gen_gmmaq_data.py` 生成输入和 CPU golden；
2. `quant_grouped_matmul_activation_quant_mx` 读取输入、调用 Blaze Kernel 并写出 Y/YScale；
3. `scripts/verify_gmmaq_result.py` 对比 NPU 输出和 golden。

结果保存在样例 CSV 对应的 `_result.csv` 文件中，单条用例的输入、golden 和 NPU
输出保存在运行器生成的 `input/`、`output/` 目录中。

## 目录结构

```text
quant_grouped_matmul_activation_quant_mx/
├── quant_grouped_matmul_activation_quant_mx.cpp
├── quant_grouped_matmul_activation_quant_mx.conf
├── quant_grouped_matmul_activation_quant_mx.csv
└── README.md
```

样例可执行文件由 `examples/acl/grouped_matmul/CMakeLists.txt` 统一注册，避免同一
target 在父子 CMake 中重复定义；运行器通过 `examples/acl/common/run.sh` 统一调度。
