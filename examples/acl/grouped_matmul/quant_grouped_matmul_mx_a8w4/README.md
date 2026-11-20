# Grouped Matmul MXA8W4 示例

本示例在 Ascend 960（`dav-9201`）上调用 Blaze 的 `GroupedMatmulWithScaleMx<0>`，按 A 的行分组。每组使用独立权重和缩放系数，输出按输入行顺序写入 FP32 矩阵。可选 bias 按组和输出通道广播。原生路径的类型与布局约束见 [Kernel 文档](../../../../docs/API/gemm/kernel/kernel_qgmm_mx_basic.md)。

## 编译与运行

加载支持 Ascend 960 的 CANN 环境。在 ops-tensor 仓库根目录安装数据脚本依赖。

```bash
python3 -m pip install numpy ml_dtypes
```

运行下列命令。统一 runner 按 [conf](quant_grouped_matmul_mx_a8w4.conf) 读取 [CSV](quant_grouped_matmul_mx_a8w4.csv)，逐条生成输入和 NumPy 参考结果，在 device 0 启动 AIC Kernel，再校验完整输出。

```bash
bash examples/acl/common/run.sh --soc=ascend960 --ops=grouped_matmul --target=quant_grouped_matmul_mx_a8w4
```

添加 `--build-only` 仅编译。添加 `--ti=0` 运行首条用例。数据与设备输出保存在本示例的 `output/` 目录。

## 输入与校验

E 为专家数，M 为输入总行数，K 为累加维度，N 为输出通道数。A 和 C 按组拼接行。B、ScaleB、bias 按专家顺序连续存储。

| 文件 | 类型 | 存储布局 |
| --- | --- | --- |
| `input_a.bin` | FP8 E4M3FN | ND `[M,K]` |
| `input_b.bin` | FP4 E2M1 | 逻辑 `[E,K,N]`，每组按 ZN 打包，K 补齐到 64，N 补齐到 16，每字节存两个 FP4 |
| `scale_a.bin` | E8M0 | `[M,S]`，每 32 个 K 元素共用一个缩放系数 |
| `scale_b.bin` | E8M0 | DN `[E,N,S]`，缩放维连续 |
| `bias.bin` | FP16、BF16 或 FP32 | 可选，`[E,N]` |
| `group_list.bin` | INT64 | `[E]` |
| `golden_c.bin`、`npu_out.bin` | FP32 | ND `[M,N]` |

S 为 `2 × ceil(K / 64)`，即缩放系数按两个一组存储。生成器用 `seed` 固定输入，NumPy 按每 32 个 K 元素应用对应的 A、B 缩放系数，以 FP32 做分组矩阵乘并加 bias。校验脚本检查文件长度、有限值和每个元素的误差 `abs(actual - golden) <= atol + rtol * abs(golden)`。

`listType=1` 使用分组行数，例如 `3;0;9`。同一分组在 `listType=0` 时写作累计行数 `3;3;12`。空组保留专家位置，所有组共处理 M 行。CSV 以 `biasDtype=none` 省略 bias，`dbL0C` 选择单缓冲（1）或双缓冲（2），`cores` 指定启动的 AIC 数，`outputDtype` 固定为 `float32`。

随附用例包含两种 groupList、空组、三种 bias、L0C 单双缓冲、K=258 的尾块及多核多 K 窗口。设备分块与参数构造见 [示例源码](quant_grouped_matmul_mx_a8w4.cpp)。
