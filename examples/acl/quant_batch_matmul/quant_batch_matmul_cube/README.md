# quant_batch_matmul_cube 样例

本样例直接组装并调用 `kernel_qbmm_cube.h`、`kernel_qbmm_cube_without_batch.h` 与
`kernel_qbmm_pertensor_streamk.h`，演示 Quant Batch Matmul 的 Cube/Fixpipe 路径：

- `kernel_variant=batch`：支持 Batch 广播；
- `kernel_variant=without_batch`：Batch 固定为 1，不执行多 Batch 广播和地址换算；
- `kernel_variant=pertensor_streamk`：Batch 固定为 1，使用 INT8 输入、单个 FP32 scale 和
  StreamK workspace。

数据生成和校验逻辑参考 `ops-samples` 的 `quant_matmul_hifp8/scripts`：

- `x1quantmode=pertensor`、`x2quantmode=pertensor`：A/B scale 均为 FP32 标量；
- `x1quantmode=default`、`x2quantmode=perchannel`：仅 B 使用 per-channel scale。scale 文件为
  FP32 `[N]`，Host 侧将其转换为 Fixpipe 使用的 `uint64_t[N]`；
- `x1quantmode=default`、`x2quantmode=pertensor`：仅 B 使用一个 per-tensor scale。

`examples/acl/quant_batch_matmul/scripts/gen_data_cube.py` 支持生成 HiFloat8/BF16、Int8/Int32 和 per-tensor
StreamK 场景的 Int8/FP16 数据。HiFloat8 Golden 按 Fixpipe 精度清除 scale 的低 13 位并保存为 BF16；
非 StreamK 的 Int8 Matmul 使用 Int32 累加和 Int32 Golden；per-tensor StreamK 使用 FP32 标量
scale 并保存 FP16 Golden。公共的 `verify_result_cube.py` 对 BF16/FP16 使用误差标准，对 Int32
逐元素精确比较。

Kernel 的 `AType/BType/CType/BiasType/X2ScaleType` 均为模板参数；Host 根据 CSV dtype 选择对应的
预编译实例，buffer 大小也按实例类型计算。

## CSV 参数

CSV 每行包含：

```text
batch,M,K,N,AType,BType,CType,bias,biasType,transA,transB,x1quantmode,x2quantmode,x2ScaleType,
baseM,baseN,baseK,kL1,l1BufferNum,aFullLoad,kernel_variant
```

其中：

- `bias` 为 `0` 或 `N`；
- `transA/transB` 支持 `true/false` 或 `1/0`；
- 当前 HiFloat8 样例支持 `AType/BType=hifloat8_t`、`CType=bfloat16_t`、`biasType=float`；
- Int8 样例支持 `AType/BType=int8_t`、`CType=int32_t`、`biasType=int32_t`，量化模式为
  `default/default`，`x2ScaleType=float` 仅作为未使用的模板类型占位；
- A/B 均使用 per-tensor scale 时，`x2ScaleType=float`；仅 B 使用 per-channel scale 时，
  `x2ScaleType=uint64_t`。per-channel scale 文件仍为 FP32，Host 将其截断并编码为传给 Kernel 的
  设备侧 `uint64_t` scale；
- `baseM <= 256`、`baseK <= 128`、`baseN <= 256`，最大基本块为 `256 * 128 * 256`；
- `l1BufferNum` 支持 `2`、`3` 或 `4`；
- `aFullLoad=true` 使用 A 全载调度，否则使用非全载调度；
- `kernel_variant` 支持 `batch`、`without_batch` 和 `pertensor_streamk`；后两者要求 `batch=1`；
- `pertensor_streamk` 固定为 Int8×Int8→FP16、`default/pertensor`、FP32 scale、ND 输入、无转置、
  无 Bias 且 `aFullLoad=false`，运行时按 AIC core 数分配并清零 workspace；
- batch/without-batch 的 `blockNum` 固定为 `32`；per-tensor StreamK 使用运行时 AIC core 数，均不在
  CSV 中配置。

CSV 还包含用于输出和数据目录命名的 `casename` 列。当前样例仅使用 ND 输入，默认用例覆盖 HiFloat8
的 per-tensor 和 per-channel scale、without-batch、三缓冲非全载与 A 全载场景，以及 Int8 的普通
ND、transA+transB、transB+batch 和 per-tensor StreamK 场景。
两个三缓冲用例均使用 `K=288`、`kL1=64`，覆盖五轮 L1 buffer 轮转（0→1→2→0→1）和 32 元素
K 尾块；非全载用例同时覆盖 M/N 尾块、B 的 per-channel scale 和 Bias；A 全载用例设置
`transA=true`，并通过 32 个 M 基本块和 3 个 N 基本块覆盖转置 A 的三缓冲布局及 A tile 复用。

## 在 NPU 上编译运行

安装 Python 依赖并初始化 CANN 环境后，通过统一入口编译并运行：

```bash
source /path/to/cann/set_env.sh
bash examples/acl/common/run.sh --ops=quant_batch_matmul --target=quant_batch_matmul_cube
```

仅编译、不启动 Device Kernel：

```bash
bash examples/acl/common/run.sh --ops=quant_batch_matmul --target=quant_batch_matmul_cube --build-only
```

复用已有构建结果：

```bash
bash examples/acl/common/run.sh --ops=quant_batch_matmul --target=quant_batch_matmul_cube --skip-build
```

## 代码结构

```
quant_batch_matmul_cube/
├── quant_batch_matmul_cube.cpp     # kernel 实现
├── quant_batch_matmul_cube.conf    # 参数路由配置
├── quant_batch_matmul_cube.csv     # CSV 测试用例
└── README.md                       # 本文档
```

构建配置在 op 层 `examples/acl/quant_batch_matmul/CMakeLists.txt` 中统一管理；运行通过 `examples/acl/common/run.sh` 统一调度，数据生成和精度校验由 `examples/acl/quant_batch_matmul/scripts/` 下的 `gen_data_cube.py` 和 `verify_result_cube.py` 执行。

单独生成 `default/per-channel` Batch 用例的输入：

```bash
python3 examples/acl/quant_batch_matmul/scripts/gen_data_cube.py \
  --batch 2 --m 128 --k 256 --n 128 --bias 0 \
  --a-type hifloat8_t --b-type hifloat8_t --c-type bfloat16_t --bias-type float \
  --trans-a false --trans-b false \
  --x1-quant-mode default --x2-quant-mode perchannel \
  --x2-scale-type uint64_t --kernel-variant batch \
  --output-dir data/tc_nd_batch
```

对应的可执行文件调用为：

```bash
./examples/build/acl/quant_batch_matmul/quant_batch_matmul_cube \
  2 128 256 128 hifloat8_t hifloat8_t bfloat16_t 0 float \
  false false default perchannel uint64_t \
  256 256 128 128 2 false batch data/tc_nd_batch
```
