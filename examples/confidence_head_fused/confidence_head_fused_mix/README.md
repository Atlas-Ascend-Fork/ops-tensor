# ConfidenceHeadFused Mix 示例

## 功能说明

组装 GEMV Cube Block、Scheduler 和 Epilogue，实现 hidden/markov 投影、可选 FP32 bias、温度缩放、截断和 Sigmoid。输入为连续 FP16/BF16，输出为 FP32；bias[N] 在 batch 间广播。

温度缺省或为正负零时按 1 计算，其他值使用当前值。clamp 作用于指数输入，先下限、后上限截断，none 表示省略该侧。K=0 时以 bias 或零作为 logits；空 batch/N 返回空输出。

## 执行方式

在仓库根目录加载 CANN 环境后执行：

```bash
bash examples/common/run.sh --ops=confidence_head_fused --target=confidence_head_fused_mix
```

公共入口根据 [CSV 用例](confidence_head_fused_mix.csv) 和 [参数映射](confidence_head_fused_mix.conf) 生成数据、执行设备程序并校验结果。

| CSV 字段 | 说明 |
| --- | --- |
| n / batch | 每个 batch 的行数与 batch 数。 |
| k_embed / k_markov | hidden 和 markov 的 K。 |
| dtype | float16 或 bfloat16。 |
| has_markov | 是否传入 markov；false 时 k_markov 须为零。 |
| temperature | none 或 FP32 数值，生成对应设备输入。 |
| bias | 是否生成 [N] FP32 bias。 |
| clamp_low / clamp_high | none 或 FP32 截断阈值。 |

保留 8 条代表用例，覆盖 FP16/BF16、双段多轮 K、batch bias、markov-only、空 markov、零 K、温度零值、单侧截断和空输出。

## 代码结构

| 文件 | 说明 |
| --- | --- |
| [confidence_head_fused_mix.cpp](confidence_head_fused_mix.cpp) | 参数解析、设备容量检查、组件组装与 Kernel 启动。 |
| [gen_data.py](../scripts/gen_data.py) | 生成量化后的 FP16/BF16 输入及 FP32 Golden。 |
| [reference.py](../scripts/reference.py) | CPU 投影与后处理计算。 |
| [verify_result.py](../scripts/verify_result.py) | 校验 FP32 输出长度与精度，生成校验结果。 |

## Blaze 组件

组件接口见 [Kernel ConfidenceHeadFused Mix](../../../docs/API/gemm/kernel/kernel_confidence_head_fused_mix.md)。示例配置 baseN=baseK=128、kL1=256、L1 四缓冲和 L0C 双缓冲。启动前检查 L1、L0、BiasTable 和 UB 容量，仅支持权重全载 L1 的输入。

## 精度校验

Golden 使用量化后的输入，以 FP32 完成投影与后处理。设备在温度缩放和 clamp 后内联执行 `Exp/Adds/Div`，最后一步保留默认的非精确 Div；存量 Sigmoid 使用精确 Div，两者不要求逐位一致。本示例保持现有 `atol=1e-4`、`rtol=1e-3`，不因实现拆分放宽阈值。逐元素满足 `abs(actual-golden) <= 1e-4 + 1e-3*abs(golden)`，并检查输出长度与非有限值；空输出按长度零校验。设备输出和 Golden 分别为 output/npu_out.bin、output/cpu_output.bin，校验结果写入 output/verify_metrics.json。
