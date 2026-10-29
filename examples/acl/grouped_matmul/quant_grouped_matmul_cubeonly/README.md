# Quant Grouped MatMul CubeOnly 样例

## 概述

本样例演示 `kernel_qgmm_cube.h`（QGMM Cube / AIC_ONLY Fixpipe 量化 Grouped MatMul）在 NPU 上的调用方式。

样例采用与 `quant_grouped_matmul_mx` 类似的 CSV 驱动结构：

1. `scripts/gen_data_cubeonly.py` 生成输入数据和 CPU golden；
2. `quant_grouped_matmul_cubeonly.cpp` 读取输入，在 NPU 上调用 Blaze QGMM Cube Kernel；
3. `scripts/verify_result_cubeonly.py` 校验 NPU 输出。

## 支持场景

当前样例覆盖简单 INT8/FP8 场景：

- INT8 × INT8 → FP16，可选 INT32 bias；
- FP8 E4M3 × FP8 E4M3 → FP32；
- PERTENSOR / PERCHANNEL / double-scale；
- Weight 支持 ND / DN / NZ / ZN；
- M 轴分组（`groupType=0`）；
- LENGTH / OFFSET / SPARSE Group List；
- single tensor 权重（`singleW=1`）。

## CSV 字段

| 字段 | 说明 |
|------|------|
| `caseName` | 用例名称 |
| `groupNum/m/n/k` | 分组数及每组的 M/N/K |
| `baseM/baseN/baseK` | L0 block 大小 |
| `kAL1/kBL1` | A/B 的 L1 K 切分 |
| `x1QuantMode/x2QuantMode` | 0/1/2：DEFAULT/PERTENSOR/PERCHANNEL |
| `isBias/dbL0C/groupType/groupListType/singleW` | bias、L0C 缓冲、分组类型等 |
| `aType/bType/cType/biasType/x2ScaleType` | 类型选择 |
| `layoutA/layoutB` | 数据排布 |
| `groupList` | group list，分号分隔 |
| `dtype` | 数据生成类型：`int8` / `fp8_e4m3` |

## 编译与运行

在仓库根目录执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_cubeonly
```

仅编译：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_cubeonly --build-only
```

## 目录结构

```text
grouped_matmul/
├── scripts/
│   ├── gen_data_cubeonly.py
│   └── verify_result_cubeonly.py
└── quant_grouped_matmul_cubeonly/
    ├── quant_grouped_matmul_cubeonly.cpp
    ├── quant_grouped_matmul_cubeonly.conf
    ├── quant_grouped_matmul_cubeonly.csv
    └── README.md
```

### 输出精度校验

CSV 的 `cType` 同时控制数据生成、Kernel 输出类型和校验器。INT8 输入支持示例输出 FP16/BF16，FP8 输入支持 FP32/BF16；BF16 golden 直接从计算结果转换为 `ml_dtypes.bfloat16`，不经过 FP16。

与同仓 MX 校验器一致：FP16/BF16 的单点绝对误差阈值和超阈值点占比阈值均为 `1e-3`，FP32 均为 `1e-4`。非有限值计为错误点；错误比例大于阈值时失败，等于阈值时通过。CPU 冒烟测试不代表 NPU 数值精度验证。
