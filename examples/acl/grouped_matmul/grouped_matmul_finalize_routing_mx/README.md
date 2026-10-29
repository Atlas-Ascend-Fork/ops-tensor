# GroupedMatmulFinalizeRouting MX Example

This CSV-driven example exercises the TensorAPI GMMFinalizeRouting path with
WeightNZ weights. It covers:

- MXFP4 E2M1 with BF16 output and BF16 logit;
- MXFP8 E4M3/E4M3 with BF16 output and FP32 logit;
- MXFP8 E4M3/E4M3 with FP32 output and FP32 logit.

Each case generates input binaries and a CPU golden result, runs the NPU
kernel, and compares the output. FP32 uses `atol=rtol=4e-3`; BF16 uses
`atol=rtol=2e-2`, with the same value used as the allowed mismatch ratio.

Build and run from the ops-tensor root:

```bash
bash examples/acl/common/run.sh --ops=grouped_matmul --target=grouped_matmul_finalize_routing_mx
```
