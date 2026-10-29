# Quant Grouped MatMul SwiGLU Quant MX 样例

## 概述

本样例演示 Ascend950 上基于 Blaze 的 MXFP8 Grouped MatMul、SwiGLU mode 2
和 MX 动态量化融合路径。权重采用 FRACTAL_NZ/NZ 或其转置 ZN 布局，Kernel
在 L1 中拼接权重左右两个 N 半区，再输出量化后的 `Y` 和 `YScale`。

计算流程为：

```text
MXFP8 Grouped MatMul (full 2N) -> SwiGLU mode 2 -> MXFP8 E4M3FN Y + E8M0 YScale
```

SwiGLU mode 2 的计算公式为：

```text
act  = min(left, clampLimit)
gate = clamp(right, -clampLimit, clampLimit) + gluBias
Yfp  = act / (1 + exp(-gluAlpha * act)) * gate
```

本样例不包含 SwiGLU mode 0/1 或其他模式。

## 支持场景

样例固定使用以下类型和公共约束：

- `X`、`Weight` 和 `Y`：MXFP8 E4M3FN；
- `XScale`、`WeightScale` 和 `YScale`：E8M0；
- `LayoutA`：ND；
- `LayoutB`：NZ（非转置）或 ZN（转置）；
- `groupType=0`、`groupListType=0`（累计 offset）、`singleW=1`；
- 无 Bias，`swigluMode=2`；
- 支持 `scaleAlg=0`（OCP）和 `scaleAlg=1`（cuBLAS）。

样例使用固定 tiling 配置；增加 CSV 用例时须遵守以下约束，不支持修改 tiling 参数：

- `baseM/baseN/baseK` 固定为 `16/64/64`，`kAL1/kBL1/scaleKAL1/scaleKBL1`
  固定为 `64/64/64/64`，`dbL0C` 固定为 `2`；
- `groupNum` 为 `[1, 8]`，`m` 为正整数，`k >= 2`，SwiGLU 前的完整 `n`
  为 64 的正整数倍；`layoutB` 仅支持 `nz` 或 `zn`，`scaleAlg` 仅支持 0 或 1；
- `clampLimit` 为可表示为 float32 的有限正数，`gluAlpha` 和 `gluBias`
  为可表示为 float32 的有限数，`dstTypeMax` 固定为 `0.0`；
- `groupList` 须包含 `groupNum` 个位于 `[0, m]` 的非降累计 offset，
  最后一项等于 `m`；重复 offset 表示空 group；
- 输入文件大小须与对应张量的预期字节数一致，整数参数及张量大小不得溢出。

CSV 中的 `n` 是 SwiGLU 前的完整宽度 `2N`。输出 `Y` 的形状为
`[m, n/2]`，输出 `YScale` 的形状为 `[m, ceil((n/2)/64), 2]`。

### 输入输出张量约束

| 张量 | 逻辑 shape | dtype / layout | 空指针语义 |
| --- | --- | --- | --- |
| `X` | `[m, k]` | E4M3FN / ND | 必须非空 |
| `Weight`（NZ） | `[groupNum, k, n]` | E4M3FN / NZ，按 expert 分别 padding | 必须非空 |
| `Weight`（ZN） | `[groupNum, n, k]` | E4M3FN / ZN，按 expert 分别 padding | 必须非空 |
| `XScale` | `[m, ceil(k/64), 2]` | E8M0 / ScaleA-ND | 必须非空 |
| `WeightScale`（NZ） | `[groupNum, ceil(k/64), n, 2]` | E8M0 / ScaleB-ND | 必须非空 |
| `WeightScale`（ZN） | `[groupNum, n, ceil(k/64), 2]` | E8M0 / ScaleB-DN | 必须非空 |
| `groupList` | `[groupNum]` | int64 / ND，累计 offset | 必须非空 |
| `Y` | `[m, n/2]` | E4M3FN / ND | 必须非空 |
| `YScale` | `[m, ceil((n/2)/64), 2]` | E8M0 / ND | 必须非空 |

样例固定 `singleW=1`，一个连续 Weight/WeightScale Tensor 承载全部 expert；
内部 `cGmAddr` 与 `biasGmAddr` 槽位不使用并传 `nullptr`。低层 Blaze
Kernel 返回 `void`，不负责返回 ACLNN 错误码；CANN 算子集成必须在
ACLNN/tiling 层完成必选空指针及 shape/layout 校验，并按接口规范对必选空指针
返回 `161001`。

内置用例覆盖：

- NZ 非转置与 ZN 转置；
- `scaleAlg=0/1`；
- 多 group 的非零累计 offset；
- K=68 的第二个 L1/Scale 窗口和 NZ/ZN 物理 padding；
- 超过两个 `baseM` 的 group，覆盖后续 M tile 的 XScale 地址；
- M 尾块和输出 N 尾块；
- 左右 N 半区、相邻列及首尾 K scale group 可区分的 WeightScale；
- `Y/YScale` 双输出。

数据生成器使用非零、稀疏且可精确复现的 E4M3FN 输入：每行的首、尾 K
位置非零，并按行及 K scale group 设置不同的 XScale。WeightScale 的逻辑值
先参与 Golden 计算，再分别按 NZ 的 ScaleBND 和 ZN 的 ScaleBDN 布局落盘。
这既覆盖多 group/M tile 地址偏移，也让第二 K 窗口和左右 WeightScale slice
实际影响输出。Golden 按设备实现先转换为 BF16，再执行对应的 MX 量化算法；
生成阶段会拒绝超出有限 E4M3FN 范围的数据。结果校验按下面的数值规则执行，
不将 FP8 编码当整数比较，也不要求正负零的二进制编码一致。
OCP 的零判断使用 BF16 原始指数：全为零/次正规数的组使用零 scale 和零倒数；
乘法仍保留正负零的符号。cuBLAS 使用绝对值码点判断非零。

### 精度校验规则

`verify_gmmsq_result.py` 分别检查两个输出：

- `Y` 按 E4M3FN 解码为 FP32 后，使用双千分之一标准：
  `abs(actual - golden) / max(abs(actual), abs(golden)) > 1e-3` 的点数
  不超过总点数的 `1e-3`。
  两者均为零时误差为零，正负零视为数值相同；仅一者为零时相对误差为 1。
- `YScale` 按 E8M0 解码为 FP32 后逐点要求数值相同。它是独立的离散输出，
  合法相邻 scale 相差 2 倍；编码 0 表示 `2^-127`，不是浮点零。
  不仅比较反量化后的 `Y * YScale`，以免错误的 Y 和 scale 相互抵消。
- 任一输出包含 NaN/Inf、文件为空或 actual/golden 元素数不同，均直接失败。

这里的 FP32 是校验解码类型，不是新增算子输出类型。本样例输出始终为
E4M3FN/E8M0；Y 使用上述双千分之一标准，但不以浮点容差放宽 scale 算法
或布局语义。CPU golden 的 BF16 舍入和 MX 量化过程保持不变。
CPU 回归覆盖正负零、容差内外、非有限值、scale 错误及 Y/scale 抵消反例；
CPU 测试通过不代表 NPU 精度验证。验证依赖版本为 NumPy 1.26.4、ml_dtypes 0.5.4。

## CSV 字段

| 字段 | 含义 |
| --- | --- |
| `caseName` | 用例名称 |
| `groupNum` | group 数量 |
| `m/n/k` | 总 M、SwiGLU 前完整 2N、K |
| `baseM/baseN/baseK` | L0 计算块大小，`baseN` 按输出 N 定义 |
| `kAL1/kBL1` | A、B 在 L1 中的 K 切分 |
| `scaleKAL1/scaleKBL1` | ScaleA、ScaleB 在 L1 中的 K 切分 |
| `dbL0C` | L0C buffer 数量 |
| `layoutB` | `nz` 或 `zn` |
| `scaleAlg` | MX 输出量化算法，支持 0 或 1 |
| `clampLimit` | SwiGLU 输入裁剪上限 |
| `gluAlpha/gluBias` | SwiGLU mode 2 参数 |
| `dstTypeMax` | 动态 dtype range 保留参数；本样例为 0 |
| `groupList` | 累计 M offset，以分号分隔，最后一个值必须等于 `m` |

## 编译和运行

在仓库根目录执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_swiglu_quant_mx
```

仅编译：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_swiglu_quant_mx --build-only
```

运行单条用例：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul \
    --target=quant_grouped_matmul_swiglu_quant_mx --ti=0
```

完整的 grouped_matmul 样例回归可执行：

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul
```

仅验证样例数据生成器、结果校验脚本及 NZ/ZN 搬运地址模型的 CPU 自检，可执行：

```bash
python3 -m unittest discover -s examples/acl/grouped_matmul/quant_grouped_matmul_swiglu_quant_mx/tests -p 'test_*.py'
```

其中搬运地址用例只验证 Python 地址模型，不执行 NPU DMA；上述自检均不验证 NPU Kernel 精度，算子输出须运行样例并与 golden 比较。

## 目录结构

```text
quant_grouped_matmul_swiglu_quant_mx/
├── quant_grouped_matmul_swiglu_quant_mx.cpp
├── quant_grouped_matmul_swiglu_quant_mx.conf
├── quant_grouped_matmul_swiglu_quant_mx.csv
├── tests/
│   ├── test_example_data.py
│   ├── test_result_verifier.py
│   ├── test_swiglu_weight_nz_concat_addresses.py
│   └── test_swiglu_weight_nz_scalar_equivalence.py
└── README.md
```

样例由 `examples/acl/grouped_matmul/CMakeLists.txt` 注册，数据生成与结果校验脚本
位于 `examples/acl/grouped_matmul/scripts/`。
