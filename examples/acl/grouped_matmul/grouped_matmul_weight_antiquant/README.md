# Grouped Matmul 权重反量化示例

## 功能说明

本示例在 Ascend 960（`NPU_ARCH=9201`）上调用 [Blaze Tensor API Kernel](../../../../docs/API/gemm/kernel/kernel_wqgmm_mix_antiquant.md)，完成按输入行分组的矩阵乘：每组使用独立的量化权重，先按输出通道反量化，再与 FP16/BF16 输入相乘，最后加上可选 bias。

向量核 AIV 负责权重反量化，矩阵核 AIC 负责矩阵乘加（MMAD）；每个 AIC 配对一个 AIV。示例程序完成设备内存分配、参数构造和 Kernel 启动，Python 脚本提供确定性输入、CPU 参考结果（golden）和输出比较。

### 维度与支持范围

`E` 是分组数，也称专家数；`M` 是所有组的总行数；`K` 是输入通道数和矩阵乘累加维度；`N` 是输出通道数。各专家共享 K/N，每组行数由 groupList 指定。

| 参数 | 类型 | 连续存储形状 | 说明 |
| --- | --- | --- | --- |
| x | FP16 / BF16 | `[M,K]` | 所有组的输入行按顺序拼接 |
| weight | INT8 / FP8 E4M3FN / FP8 E5M2 / HiFloat8 | `[E,K,N]` 或 `[E,N,K]` | 每组独立权重，方向由 trans 指定 |
| scale | 与 x 相同 | `[E,N]` | 每专家、每输出通道一个缩放系数，沿 K 共用 |
| offset | 与 x 相同 | `[E,N]` | 可选，仅 INT8 权重支持 |
| bias | FP16 x 配 FP16；BF16 x 配 BF16 或 FP32 | `[E,N]` | 可选，沿输入行广播 |
| groupList | INT64 | `[E]` | 每组行数或累计行数 |
| y | 与 x 相同 | `[M,N]` | 与输入的行顺序一致 |

### 权重方向：NN 与 NT

NN/NT 的第一个字母描述 x，第二个描述 weight；N 表示不转置，T 表示转置。x 始终为不转置的 `[M,K]`：

| CSV 的 trans | 方向 | weight 文件的形状 | 每组的矩阵乘 |
| --- | --- | --- | --- |
| 0 | NN：x 和 weight 均不转置 | `[E,K,N]`，N 连续 | `x_e × weight_e` |
| 1 | NT：x 不转置，weight 转置 | `[E,N,K]`，K 连续 | `x_e × weight_eᵀ` |

生成器先生成逻辑 `[E,K,N]` 权重，再按 trans 写入文件。例如单组逻辑权重 `[[1,2,3],[4,5,6]]`，NN 写入 `1,2,3,4,5,6`，NT 写入 `1,4,2,5,3,6`。程序将文件原样复制到设备，Kernel 用对应 layout 读取。

### groupList 的填写方法

例如三组分别处理 2、0、3 行，总 M=5：

| listType | 含义 | CSV 的 groupList |
| --- | --- | --- |
| 1 | count：每个专家的行数 | `2;0;3` |
| 0 | cumsum：截至当前专家的累计行数 | `2;2;5` |

分号分隔 E 个非负 INT64 值。count 的和等于 M；cumsum 非递减且最后一项等于 M。空组保留在列表中，对应专家的权重和参数仍占一行专家维。

## 编译与运行

先加载支持 Ascend 960 的 CANN 环境，在 ops-tensor 仓库根目录执行：

```bash
python3 -m pip install -r examples/requirements.txt
bash examples/acl/common/run.sh \
  --soc=ascend960 --ops=grouped_matmul \
  --target=grouped_matmul_weight_antiquant
```

统一 runner 根据 [conf](./grouped_matmul_weight_antiquant.conf) 读取 [CSV](./grouped_matmul_weight_antiquant.csv)，对每条记录依次执行：

```text
CSV 配置 → 生成输入和 golden → 启动设备 Kernel → 比较完整输出
```

只编译示例，或选择 CSV 中第 0 条用例：

```bash
bash examples/acl/common/run.sh \
  --soc=ascend960 --ops=grouped_matmul \
  --target=grouped_matmul_weight_antiquant --build-only

bash examples/acl/common/run.sh \
  --soc=ascend960 --ops=grouped_matmul \
  --target=grouped_matmul_weight_antiquant --ti=0
```

已有构建产物时可添加 `--skip-build`。设备程序使用 device 0，启动核数由 CSV 的 cores 指定，取值为 1–32 且应不超过设备可用 AIC 数。Python 数据处理使用 numpy、ml_dtypes，HiFloat8 编码使用 en_dtypes。

### 参数构造与资源生命周期

示例按照 [Kernel Params 调用契约](../../../../docs/API/gemm/kernel/kernel_wqgmm_mix_antiquant.md#调用契约) 设置完整 K、调度基本块和相等的 Ka/Kb，并在读取 groupList 后检查其非负性、累计顺序和总行数。

设备内存及 TensorList 描述符由 `DeviceBuffer` 管理，stream 由 `AclStream` 管理，ACL 初始化和设备选择由 `AclRuntime` 管理。正常路径在每次 Kernel 启动后等待 stream 完成，比较两次输出，关闭 stream 并写出结果；设备内存释放后重置设备，再结束 ACL。

发生异常时，stream 析构先尝试同步已提交任务，再销毁 stream，随后释放其使用的设备内存。ACL 环境在外层作用域结束时依次尝试重置设备和结束 ACL；设备选择失败时，构造过程释放已经初始化的 ACL 环境。清理接口的错误记录接口名和错误码，析构过程不抛异常，原始异常继续交给 main 报告。正常路径显式检查关闭结果，关闭失败时程序返回非零退出码。

## 文件与参数

| 文件 | 用途 |
| --- | --- |
| [grouped_matmul_weight_antiquant.cpp](./grouped_matmul_weight_antiquant.cpp) | 分配内存，构造单元素 TensorList 描述符和 Kernel Params，启动混合核 |
| [grouped_matmul_weight_antiquant.csv](./grouped_matmul_weight_antiquant.csv) | 用例配置和校验容差 |
| [grouped_matmul_weight_antiquant.conf](./grouped_matmul_weight_antiquant.conf) | CSV 列到生成器、设备程序和校验器参数的映射 |
| [gen_grouped_weight_antiquant.py](../scripts/gen_grouped_weight_antiquant.py) | 生成输入二进制文件和 CPU golden |
| [verify_grouped_weight_antiquant.py](../scripts/verify_grouped_weight_antiquant.py) | 逐元素比较 golden 与设备输出 |
| [test_weight_antiquant_data.py](./tests/test_weight_antiquant_data.py) | 输入序列化、分阶段舍入和校验器的 Python 单元测试 |

### CSV 字段

```text
caseName,groups,m,n,k,xDtype,weightDtype,trans,hasOffset,biasDtype,listType,groupList,cores,seed,atol,rtol
```

| 字段 | 含义及取值 |
| --- | --- |
| caseName | 用例名称，便于识别配置 |
| groups | 专家数 E，1–1024 |
| m | 总行数 M，1–1048576 |
| n、k | 输出通道数 N、输入通道数 K，均为 1–4096 |
| xDtype | `float16` 或 `bfloat16`，同时决定 scale/offset 和输出类型 |
| weightDtype | `int8`、`float8_e4m3fn`、`float8_e5m2` 或 `hifloat8` |
| trans | 0 为 NN，1 为 NT，含义见上文 |
| hasOffset | 0 省略 offset，1 使用 offset；1 仅配 int8 |
| biasDtype | `none` 表示无 bias；其余按支持范围填写 `float16`、`bfloat16` 或 `float32` |
| listType | 0 为累计行数，1 为每组行数 |
| groupList | 分号分隔的 E 个值 |
| cores | 启动的 AIC 数，每个 AIC 配一个 AIV |
| seed | 数据生成随机种子，相同配置和种子可复现输入 |
| atol、rtol | 逐元素比较的绝对容差和相对容差 |

例如，下面一行描述 FP16 输入、INT8 转置权重、带 offset 和 FP16 bias 的三专家配置：

```csv
custom_fp16_int8_nt,3,5,19,17,float16,int8,1,1,float16,1,2;0;3,4,20261105,0.003,0.003
```

此时 x 为 `[5,17]`，weight 文件为 `[3,19,17]`，scale/offset/bias 均为 `[3,19]`，输出为 `[5,19]`。

### 生成的数据

runner 将数据保存在本示例的 `output/` 目录，目录内容对应最近执行的用例：

| 文件 | 内容 |
| --- | --- |
| x.bin、weight.bin、scale.bin | 输入、物理布局权重、缩放系数 |
| offset.bin、bias.bin | 启用相应参数时生成的数据 |
| group_list.bin | E 个 INT64 值 |
| golden_weight.bin | 反量化后的逻辑 `[E,K,N]` 权重，类型与 x 相同 |
| golden_y.bin | CPU 参考输出 `[M,N]` |
| npu_out.bin | 设备输出 `[M,N]` |

## 数据生成和精度校验

### 输入与参考计算

生成器固定每条用例的随机种子。x 从 `[-0.5,0.5)` 生成，INT8 weight 从整数 `[-16,16]` 生成，浮点低精度 weight 从 `[-4,4)` 生成后编码；scale、offset、bias 分别从 `[0.03,0.12)`、`[-0.5,0.5)`、`[-0.125,0.125)` 生成并转为指定类型。

golden 从实际写入文件的字节重新解码。NT 权重读回后恢复为逻辑 `[E,K,N]`，随后对每个专家计算：

```text
w0 = Cast到x类型(weight)
w1 = 舍入到x类型(w0 + offset)       // 未启用 offset 时 w1 = w0
w2 = 舍入到x类型(w1 * scale)
y  = 转为输出类型(FP32矩阵乘(x, w2) + bias)
```

INT8/HiFloat8 转 BF16 经过 FP16 中间值；E4M3FN/E5M2 经过 FP32 中间值。Add 和 Mul 分开舍入，scale/offset 按专家和输出通道广播。CPU 矩阵乘使用 FP32 累加，设备输出按下述容差与其比较。

### 判定方法

设备程序对同一输入连续启动两次 Kernel，每次先初始化输出缓冲区，比较两次完整输出是否一致。校验器进一步检查输出文件长度、每个元素是否为有限值，以及：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

| 输出类型 | CSV 示例的 atol | CSV 示例的 rtol |
| --- | ---: | ---: |
| FP16 | 0.003 | 0.003 |
| BF16 | 0.02 | 0.02 |

校验日志给出元素数 `elements`、不匹配数 `mismatches` 和最大绝对误差 `max_abs`。失败时返回非零退出码，并输出首个不匹配元素的行列坐标、参考值和实际值；成功时 mismatches 为 0。

### 用例覆盖

CSV 共 50 条，覆盖 x/weight 类型、合法 bias 类型、offset 开关和两种权重方向的全部配置。每对 NN/NT 使用相同种子和逻辑数据。维度组合覆盖小 K/N、16 元素对齐边界、两次 Vector 计算拼接权重块，以及 K=257/513 的多段累加；分组包含空组和专家数多于启动核数的情况。

设备侧采用无 bias 时 baseM=256、有 bias 时 baseM=240；baseN=`min(256,Align16(N))`，kL0=`min(64,Align16(K))`，kL1=4×kL0。这里 `Align16` 表示向上补齐为 16 的倍数，分块与缓冲区说明见 [Kernel 文档](../../../../docs/API/gemm/kernel/kernel_wqgmm_mix_antiquant.md#计算流程与缓冲区)。

## 辅助脚本测试

在仓库根目录运行以下命令，可检查生成器的 NN/NT 存储等价、分步舍入、错误数据拒绝和校验器行为：

```bash
python3 examples/acl/grouped_matmul/grouped_matmul_weight_antiquant/tests/test_weight_antiquant_data.py -v
```

调度和 Tensor 视图的 C++ 测试入口见 [UT 说明](../../../../tests/ut/op_kernel/weight_quant_grouped_matmul/README.md)。
