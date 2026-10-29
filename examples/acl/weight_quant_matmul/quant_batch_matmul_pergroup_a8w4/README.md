# QuantBatchMatmul Pergroup A8W4 Example

本样例对应 `kernel_wqmm_mix_pergroup.h`（T-CG per-group 权重反量化 Kernel）：AIV 把 packed
FP4 权重乘 per-group BF16 scale 反量化成 FP8 写入共享 L1，AIC 执行 FP8×FP8 MMAD，Fixpipe
乘 per-channel yScale（UINT64 编码）输出 INT8：

```text
y = (x1 @ (x2 * x2Scale)) * yScale
```

- `variant=ND`：Weight ND（转置，`dn_ext_layout_ptn`），x2 物理 `(n, k)` 行主序、x2Scale
  物理 `(n, k/32)` 行主序；
- `variant=NZ`：Weight NZ（`nz_layout_ptn` fractal），x2 为 `((16, k1), (32, n1))` fractal、
  nibble 沿 N 打包，x2Scale 物理 `(k/32, n)` 行主序。

tiling 为固定单核配置（`stepM == stepN == 1`，baseM=32/baseN=64/baseK=128，BL1 四缓冲 +
两 AIV N 二分），要求 `m <= 32`、`n <= 64`、`k` 为 32 的倍数。

## 执行

```bash
bash examples/acl/common/run.sh \
    --ops=weight_quant_matmul \
    --target=quant_batch_matmul_pergroup_a8w4
```

CSV 中的 `variant` 选择 ND/NZ 输入布局。数据生成与 golden 参见
`scripts/gen_data_pergroup_a8w4.py`（yScale 的 UINT64 打包与 Fixpipe 尾数截断均按硬件语义
模拟），校验脚本 `scripts/verify_result_pergroup_a8w4.py` 按 ±1 容差比对 INT8 输出。
