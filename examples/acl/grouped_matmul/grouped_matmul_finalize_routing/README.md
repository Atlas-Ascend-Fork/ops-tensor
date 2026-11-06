# GroupedMatmulFinalizeRouting 示例

## 功能与目录

本示例面向 Ascend950（DAV_3510、arch35），通过 CSV 参数选择量化模式、权重格式和数据类型，直接组装 ops-tensor 的 Kernel、BlockMmad、Epilogue 和 Scheduler。它是 device 组件的调用示例，不依赖 ops-transformer 的 aclnn 入口。

本目录使用单一目标 `grouped_matmul_finalize_routing`，包含同名 CPP、CSV、CONF，以及组件组装头 `gmmfr_kernel.h`。数据生成和精度校验脚本为 `examples/acl/grouped_matmul/scripts/gen_gmmfr_quant_data.py`、`verify_gmmfr_quant_result.py`，遵循统一执行器的目录约定。

## 支持组合

| quant_mode | weight_format | dtype（A） | B | output_dtype | logit_dtype |
| --- | --- | --- | --- | --- | --- |
| pertoken | NZ | int8、fp8_e4m3fn、hifloat8 | 与 A 同类型，转置 WeightNZ | float32 | float32 |
| mx | ND | fp8_e4m3fn、fp8_e5m2、fp4_e2m1 | 与 A 同类型，非转置 ND | float32 | float32 |
| mx | NZ | fp4_e2m1 | 同类型，非转置 WeightNZ | bfloat16 | bfloat16 |
| mx | NZ | fp8_e4m3fn | 同类型，非转置 WeightNZ | bfloat16、float32 | float32 |
| a8w4 | NZ | fp8_e4m3fn | packed FP4 E2M1，转置 WeightNZ | float32 | float32 |

A 为 ND，bias/shared input 为 BF16。FP32 输出使用 INT64 rowIndex，BF16 输出使用 INT32 rowIndex。本表列出本示例提供的组合，不代表组件模板或 aclnn 接口的全部能力。

CSV 共 22 条：pertoken 的三个类型、MX WeightND 的三个类型及 A8W4 各有对齐与 M 尾块场景，共 14 条；另有 3 条 MX WeightNZ FP32/BF16 输出场景，以及 5 条 INT8 确定性场景（尾块、大 N、单专家跨窗口、跨组窗口及恰好写满窗口）。尾块场景覆盖可选 token scale、非零 bias/shared input、空专家和累计分组列表。

## 构建与执行

从 ops-tensor 根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
git submodule update --init include/tensor_api
python3 -m pip install numpy ml_dtypes
bash examples/acl/common/run.sh --ops=grouped_matmul --target=grouped_matmul_finalize_routing -j1
```

统一 runner 依次完成编译、生成输入、执行 NPU kernel 和 CPU golden 对比。`--ti=0` 选择首条场景；`--build-only` 仅构建，`--skip-build` 复用可执行文件。输入与输出位于本目录的 `input/`、`output/`；runner 执行后清理生成数据，结果汇总写入同目录的 `grouped_matmul_finalize_routing_result.csv`。

本示例使用仓库固定版本的 tensor_api 子模块及配套 CANN 编译器。Kernel 启动类型为 `KERNEL_TYPE_MIX_AIC_1_2`，一个 AIC 配两个 AIV。

## 配置字段与约束

- `quant_mode,weight_format,dtype,output_dtype,logit_dtype`：取上表组合。A8W4 的 B 固定为 FP4 E2M1。
- `e,m,k,n,batch`：正整数，m 是所有专家的输入总行数；示例要求 K/N 为 64 的倍数。非转置 NZ 的 K0=16，FP8/FP4 的 N0 分别为 32/64；转置 NZ 的 K0=32、N0=16。
- `bias,tokenScale`：0/1；tokenScale 仅控制 pertoken 可选的逐 token FP32 scale。MX/A8W4 的 E8M0 ScaleA 始终必需。
- `sharedLen,sharedOffset`：行数，非负且和不超过 batch；shared input 缩放权重固定为 0.5。
- `groupType`：0 为 INT64 累计行数列表，1 为 INT64 行数列表。`emptyGroup=1` 生成一个空专家。
- `deterministic`：仅 pertoken INT8 支持。`workspaceRows=0` 为完整 M 分配 workspace；非零值必须是 baseM 的正整数倍，可用于验证单专家跨窗口处理。
- pertoken ScaleB 为 FP32 `[E,N]`，可选 ScaleA 为 FP32 `[M]`。MX/A8W4 的 ScaleA/ScaleB 为 E8M0，每 32 个 K 元素一组。

pertoken/MX 全量化使用 baseM=32、baseN=64、baseK=64；K 操作数与 MX scale 的 L1 窗口均按完整 K 配置。这是示例的固定 tiling，扩展大 K 时调用方须重新检查 L1 等资源容量。A8W4 使用配套动态 K 分块和 N 重切分调度。上述约束不等同于算子 Host 的完整支持范围。

可执行文件接收以下位置参数，由 CONF 提供：

```text
quant_mode weight_format dtype output_dtype logit_dtype E M K N batch bias tokenScale sharedLen sharedOffset groupType dataDir outputPath [deterministic [workspaceRows]]
```

dataDir 内包含 `x.bin,weight.bin,scale_b.bin,bias.bin,scale_a.bin,groups.bin,shared.bin,logit.bin,rows.bin`；程序检查配置组合和输入字节数。可选输入保留占位文件，由开关决定是否传入有效地址。Kernel 启动前将输出置零。

## 数据与精度验证

生成脚本使用明确的有限低精度编码构造非零输入；HIFLOAT8 取值为 ±1、±0.5，FP4 每两个存储元素打包成一个字节。CPU reference 独立执行分组矩阵乘、scale、bias、logit 和路由累加；输出初值包含 shared input 的贡献。

BF16 reference 在矩阵乘结果转 BF16、logit 转 BF16、BF16 乘法和每次路由加法后分别舍入，不能替换成全部 FP32 累加结束后只转换一次。

- pertoken、MX WeightND、A8W4：FP32 逐元素 atol=rtol=1e-4，不允许错误元素。
- MX WeightNZ：FP32 atol=rtol=4e-3，错误元素比例不超过 4e-3；BF16 atol=rtol=2e-2，错误元素比例不超过 2e-2。

所有模式均拒绝 NaN/Inf、空输出和长度不一致。阈值用于本示例的验证，不是组件的通用精度承诺。

## 组件组装

`gmmfr_kernel.h` 的 `Run<Flow,A,B,Deterministic,Out,Logit,RowIndex>` 封装示例组装，不属于 Blaze 公共 API。CSV 中的确定性 INT8 用例同时覆盖完整 workspace 和有限窗口配置。

[组件使用指南](../../../../docs/API/gemm/kernel/grouped_matmul_finalize_routing.md)；[Kernel UT](../../../../tests/ut/op_kernel/grouped_matmul_finalize_routing/README.md)。

输出 y 由示例调用端在启动 kernel 前清零；kernel 仅写入 shared input 初值并累加路由结果。

`deterministic=1` 仅用于 pertoken INT8，调用方提供按 M 向上对齐到 baseM 后的 FP32 workspace，验证有序路由累加。
