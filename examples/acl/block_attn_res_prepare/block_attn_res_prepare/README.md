# BlockAttnResPrepare Example

本样例演示 `Blaze::Attention::Kernel::KernelBlockAttnResPrepare` 的完整可执行用法，包括：

- host 侧输入、tiling、workspace 和输出 GM 管理；
- `KERNEL_TYPE_MIX_AIC_1_2` Kernel entry；
- 公开 `Kernel::Params` 的完整字段映射；
- 空 `validBlocks` 与完整 `validBlocks` 两个 CSV 用例；
- NumPy golden 对 `weightedOutput`、`softmaxMax` 和 `softmaxSum` 的联合校验。

## 执行

```bash
bash examples/acl/common/run.sh \
    --ops=block_attn_res_prepare \
    --target=block_attn_res_prepare
```

用例定义见 `block_attn_res_prepare.csv`。样例固定使用 `[T,N,D]=[1,8,32]` 的 residual 和
`[S,D]=[2,32]` 的 query，以便清晰展示 Kernel 组装和输出语义。
