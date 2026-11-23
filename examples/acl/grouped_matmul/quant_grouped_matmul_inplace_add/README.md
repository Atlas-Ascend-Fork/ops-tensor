# Quant Grouped MatMul InplaceAdd 样例

## 概述

本样例在 Ascend 950 上按量化场景选择 Kernel，并在同一个 FP32 YRef 缓冲区上连续执行两次原地累加。

## 支持场景

| `scenario` | A/B 类型 | Scale1 / Scale2 |
|------------|----------|-----------------|
| `hif8_tc` | HiFloat8 | FP32 `[G]` / `[G, N]` |
| `hif8_tt` | HiFloat8 | FP32 `[G]` / `[G]` |
| `mxfp8` | MXFP8 E4M3FN | E8M0 `[K / 64 + G, M, 2]` / `[K / 64 + G, N, 2]` |
| `mxfp4` | MXFP4 E2M1 | E8M0 `[K / 64 + G, M, 2]` / `[K / 64 + G, N, 2]` |

- A 为 DN，B 为 ND，按 K 分组。`K / 64` 为整数除法，MX scale 每个元素覆盖 32 个 K 元素，空组保留对应的 scale 行。
- MXFP4 沿 A 的 M 轴和 B 的 N 轴将两个元素打包成一个字节，因此 M 和 N 必须为偶数。
- YRef 为 FP32 ND `[G, M, N]`，每组累加至对应的 `[M, N]` 输出切片。
- 样例固定使用 `32 × 32 × 64` 的 L0 tile，A/B 在 L1 中的 K 切分长度均为 64。用例覆盖两种组表编码、空组、M/N 尾块和跨多个 K tile 的组。

## CSV 字段

用例由 `quant_grouped_matmul_inplace_add.csv` 配置。

| 字段 | 说明 |
|------|------|
| `caseName` | 用例名称 |
| `scenario` | 量化场景，取值见上表 |
| `m` | 每组的 M 维度 |
| `n` | 每组的 N 维度 |
| `k` | 各组 K 长度之和 |
| `groupNum` | 组数 G |
| `groupListType` | 0 为累计 K，1 为每组 K 长度 |
| `groupLengths` | 以分号分隔的 G 个组长度，各项非负且总和等于 `k`，0 表示空组 |

`m`、`n`、`k` 和 `groupNum` 必须为正整数。`groupLengths` 始终按组填写长度，由数据生成脚本按 `groupListType` 写出累计 K 或组长度。

## 编译与运行

按[样例环境要求](../../../README.md)准备 CANN 环境。数据生成和结果校验依赖 NumPy。以下命令在仓库根目录执行。

```bash
bash build.sh --examples --ops=grouped_matmul --target=quant_grouped_matmul_inplace_add
```

仅编译时使用以下命令。

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=quant_grouped_matmul_inplace_add --build-only
```

## 目录结构

```text
grouped_matmul/
├── scripts/
│   ├── gen_qgmm_inplace_add_data.py
│   └── verify_qgmm_inplace_add_result.py
└── quant_grouped_matmul_inplace_add/
    ├── quant_grouped_matmul_inplace_add.cpp
    ├── quant_grouped_matmul_inplace_add.conf
    ├── quant_grouped_matmul_inplace_add.csv
    └── README.md
```

## 输出精度校验

1. `gen_qgmm_inplace_add_data.py` 按场景生成量化 A/B、scale、每个元素均为非零的初始 YRef，以及两次调用的 CPU golden。
2. `quant_grouped_matmul_inplace_add.cpp` 将初始 YRef 复制到设备，在同一 GM 缓冲区上连续执行两次 Kernel，输出 `npu_out_1.bin` 和 `npu_out_2.bin`。
3. `verify_qgmm_inplace_add_result.py` 逐元素校验第一次输出为 `YRef_initial + delta`，第二次为 `golden_1 + delta`。`delta` 是单次调用的反量化结果。两次都单独检查空组是否保留初始值。
