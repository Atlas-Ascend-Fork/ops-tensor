# RotateQuant Example

本样例演示 Ascend950（`dav-3510`，AIC:AIV=1:2）上 RotateQuant 的 Blaze 模板组装，
直接使用 `BlockSchedulerRotateQuant`、`GemmUniversal`、`BlockMmad` 和
`BlockEpilogueRotateQuant`，不依赖 ops-nn 仓或 ACLNN 接口。

## 计算语义

记 `X` 为 `[M,N]`，`Q=N/K`，将每行划分为 Q 个连续的 K 元素片段：

```text
Z[m,q*K:(q+1)*K] = X[m,q*K:(q+1)*K] @ rotation[e, b]
```

- 普通模式：`groupList=none`，忽略公式中的 e。
- 分组模式：按 `groupList` 沿 X 的 M 轴分组，e 为原始组下标；空组仍占用一个 rotation 下标。
- `B=1`：所有 N 轴片段共享该组的同一个 `[K,K]` 旋转矩阵。
- `B=Q`：第 q 个片段使用该组的第 q 个旋转矩阵，即 `b=q`。

| 模式 | rotation 的逻辑 shape |
| --- | --- |
| 普通，B=1 | `[K,K]` |
| 普通，B=Q | `[Q,K,K]` |
| 分组，B=1 | `[E,K,K]` |
| 分组，B=Q | `[E,Q,K,K]` |

Cube 用 FP16/BF16 输入和 FP32 累加，Fixpipe 将完整结果转为 BF16 后送到 AIV。
对于每连续 32 个元素组成的 MX 量化组，若 BF16 alpha 满足 `0<alpha<1`，
先计算 BF16 阈值 `limit=alpha*max(abs(Z))`，再截断到 `[-limit,limit]`。
随后计算 E8M0 scale 并量化为 MX-FP4 E2M1。两个 AIV 按轮次交替处理完整块。

## 本样例配置

| 项目 | 配置 |
| --- | --- |
| 输入 | 连续 ND；X 为 `[M,N]`，X 与 rotation 同为 FP16 或 BF16 |
| K | 32、64、128、256，且 N 能被 K 整除 |
| 输出 | Y 逻辑 shape 为 `[M,N]`；FP4 打包存储为 M 行、每行 N/2 字节 |
| scale | E8M0 字节，逻辑 shape `[M,ceil(N/64),2]`；有效区每行 N/32 字节，padding 为 0 |
| 量化属性 | `axis=-1`、`roundMode=rint`、`scaleAlg=2`、`dstTypeMax=6`、`trans=false` |
| alpha | BF16 标量；`none` 不传入 alpha；0 和 1 不启用 clamp |
| groupList | `none` 为普通模式；分组模式为 1～1024 个 INT64 值，用分号分隔 |
| groupListType | 0 为 cumsum，1 为 count；允许空组、全部为空及只覆盖 M 的前缀 |

本样例要求 `M,N>0`、`M*N<=INT32_MAX`。这些是样例的输入限制；本样例没有穷举
RotateQuant 模板支持的全部输出 dtype、量化属性或输入维度。

每块最多处理 16 个 K 元素行。Host 查询设备容量，校验 L1/L0/UB 空间，并按 L0B
双缓冲容量选择 `baseK`；K=256 时可分多段 MMAD，累加完成后再 Fixpipe。
最多启动 8 个 AIC 任务组，实际有效核和分组尾块由 Scheduler 决定。
这是便于阅读的样例分块，不复刻 ops-nn 的性能 Tiling 策略。

## 编译与执行

使用仓库 examples 的公共环境：CANN Toolkit >=9.1.0、Ascend950、`bisheng`、
CMake 和初始化后的 Tensor API 子模块。Python 参考环境为
`numpy==2.5.3`、`ml_dtypes==0.6.0`，不需要 PyTorch。
数据生成使用 `ml_dtypes.bfloat16`；校验脚本按固定的 FP4 E2M1 编码表解码，
仅依赖 NumPy，不要求 `ml_dtypes` 提供 `float4_e2m1fn` 类型。

在 ops-tensor 仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install numpy==2.5.3 ml_dtypes==0.6.0
bash examples/common/run.sh --ops=rotate_quant --target=rotate_quant
```

也可以使用仓库入口：

```bash
./build.sh --examples --ops=rotate_quant --target=rotate_quant
```

仅编译或选择第 0～3 条用例：

```bash
bash examples/common/run.sh --ops=rotate_quant --target=rotate_quant --build-only
bash examples/common/run.sh --ops=rotate_quant --target=rotate_quant --skip-build --ti=0-3
```

单独运行一条分组用例：

```bash
data_dir=examples/rotate_quant/rotate_quant/output
python3 examples/rotate_quant/scripts/gen_data.py \
  --m 65 --n 384 --k 128 --b 3 --dtype float16 \
  --group-list-type 0 --group-list '17;17;40;65' --alpha 0.5 --output-dir "$data_dir"
examples/build/rotate_quant/rotate_quant \
  65 384 128 3 float16 0 '17;17;40;65' 0.5 "$data_dir"
python3 examples/rotate_quant/scripts/verify_result.py \
  --golden-dir "$data_dir" --actual-dir "$data_dir" --m 65 --n 384
```

## 数据与校验

`gen_data.py` 使用固定 seed 20261026，生成随机输入和 QR 分解得到的旋转矩阵。
NumPy 用实际 FP16/BF16 输入的 FP32 值计算矩阵乘，再转 BF16 并执行可选 clamp。
本样例使用有限、正常范围随机数据，不覆盖 NaN/Inf/subnormal 输入。

| 文件 | 内容 |
| --- | --- |
| `input_x.bin`、`input_rotation.bin` | 按输入 dtype 存储的连续 16 位输入数据 |
| `input_alpha.bin` | BF16 标量；alpha=none 时该文件仍存在，但 kernel 不读取 |
| `input_group_list.bin` | INT64 分组列表；普通模式写一个占位 0，kernel 不读取 |
| `golden_ref.bin` | 转 BF16 并执行 clamp 后的参考值，以 FP32 存储 |
| `golden_scale.bin` | 根据参考值独立计算的 DYN scale 字节，含每行 padding |
| `case.json` | shape、分组、有效行数、属性、seed 和 Python 库版本 |
| `npu_y.bin`、`npu_scale.bin` | 设备输出 |
| `verify_metrics.json` | 公共批跑框架读取的校验结果 |

校验方式参考 FlatQuant example，属于样例级检查：

- 将 FP4 打包值解码，以设备输出的 scale 反量化，与参考值比较；
  误差界为 `1.1*scale+1e-9`。FP4 E2M1 最大半 ULP 为一个 scale，额外余量覆盖 BF16 舍入。
- scale 与 CPU 参考的指数编码最多相差 1，覆盖矩阵乘累加差异使 BF16 amax 跨过指数阈值的情况。
  有限输入不允许输出 scale=255。
- 有效行的 scale padding 必须为 0。
- Host 在启动前将 Y 和 scale 填为 `0xa5`。未覆盖行的全部输出字节必须保留该值；
  全空分组时所有输出都应保持不变。

这套检查不替代需求 STC 的全场景精度和性能验收，也不提供性能对比结论。

## 用例与目录

`rotate_quant.csv` 提供 14 条用例，覆盖两种输入 dtype、四种 K、四种 rotation shape、
单轮/多轮、M 尾块、K=32 的 scale padding、count/cumsum、空组、部分覆盖和可选 clamp。

```text
examples/rotate_quant/
├── CMakeLists.txt
├── scripts/
│   ├── gen_data.py
│   └── verify_result.py
└── rotate_quant/
    ├── rotate_quant.cpp
    ├── rotate_quant.conf
    ├── rotate_quant.csv
    └── README.md
```
