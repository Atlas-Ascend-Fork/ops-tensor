# QuantBatchMatmul MIX Example

本样例对应 UT wrapper `qbmm_mix.h`，覆盖两个 Kernel：

- `variant=batch`：`kernel_qbmm_mix.h`；
- `variant=without_batch`：`kernel_qbmm_mix_without_batch.h`。

A/B 为 INT8，A scale 为 per-token FP32 `[M]`，B scale 为 per-channel FP32 `[N]`。AIC 计算 INT32
累加结果，AIV 执行反量化并输出 FP16。scale 使用非单位值，以验证其地址和布局是否正确。

## 执行

```bash
bash examples/acl/common/run.sh \
    --ops=quant_batch_matmul \
    --target=quant_batch_matmul_mix
```

CSV 中的 `variant` 用于选择支持 Batch 广播的 `batch` 实现或 Batch 固定为 1 的
`without_batch` 实现。
