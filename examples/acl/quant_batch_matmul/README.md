# QuantBatchMatmul Examples

本目录与 `tests/ut/op_kernel/quant_batch_matmul` 的实现分类保持一致。每个 example target 对应一类
可独立编译和运行的 Kernel：

| UT wrapper | Example target | 覆盖的主要实现 |
| --- | --- | --- |
| `qbmm_cube.h` | `quant_batch_matmul_cube` | Cube batch / without-batch / per-tensor StreamK |
| `qbmm_mix.h` | `quant_batch_matmul_mix` | MIX batch / without-batch |
| `qbmm_mx.h` | `quant_batch_matmul_mx` | MX batch / without-batch / L0C ping-pong / StreamK |

运行全部 QBMM examples：

```bash
bash examples/acl/common/run.sh --ops=quant_batch_matmul
```

运行单个实现族：

```bash
bash examples/acl/common/run.sh \
    --ops=quant_batch_matmul \
    --target=quant_batch_matmul_mx
```

每个 target 目录均包含同名 `.cpp`、`.csv`、`.conf` 和 README。CSV 通过 `variant` 或
`kernel_variant` 字段选择该 target 下的具体 Kernel 实现；公共数据脚本位于 `scripts/`。
