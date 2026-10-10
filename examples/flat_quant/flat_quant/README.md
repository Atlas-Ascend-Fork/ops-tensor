# FlatQuant Example

本样例演示 `Blaze::Attention::Kernel::AttentionUniversal`（FlatQuant 特化，
`kernel_flat_quant.h`）的完整可执行用法。算子语义：

- 输入：`x [K, M, N]`（BF16）、`P1 [M, M]`、`P2 [N, N]`（BF16，可选）；
- 计算：`y[b] = P1 @ x[b] @ P2`（`hasP2=false` 时退化为 `P1 @ x[b]`）；
- 输出：MMAD 结果经 AIV epilogue 做 MX 动态量化——FP4 (E2M1) 打包值
  `npu_y.bin`（每字节两个 nibble，低半字节为偶数下标元素）+ 每 32 元素一组的
  E8M0 scale `npu_scale.bin`。

样例覆盖：

- AIC/AIV 混合执行（`KERNEL_TYPE_MIX_AIC_1_2`，块内 cube/vector 握手同步）；
- `iterBatch`（单 tile 内多 batch 迭代，UB 中级联多个 batch 结果）；
- 多轮 tile 调度（`K / iterBatch > blockNum` 时的 AIC↔AIV 双向同步轮转）；
- 尾轮混合 `iterBatch`（tail tile 的 batch 数自适应）；
- `hasP2=false` 路径（跳过第二个矩阵乘）。

## 形状约束

`M`、`N` 为 16 的倍数，`M*N` 为 64 的倍数且不超过 4096（epilogue UB 中
scaleBlock 缓冲的上限），`iterBatch*M*N <= 32768`（L0C 上限）。Kernel 以固定
8 个 block 启动，实际参与计算的 block 数由 `BlockSchedulerFlatQuant` 决定。

## 执行

```bash
bash examples/common/run.sh --ops=flat_quant
```

用例定义见 `flat_quant.csv`。数据生成（`scripts/gen_data.py`）使用近正交矩阵
（QR 分解）作为 `P1/P2`，并以 FP32 einsum 生成量化前的 golden 参考；
校验（`scripts/verify_result.py`）将 NPU 输出按其自身 scale 反量化后与 golden
比对（容差为 1.1 倍组 scale，覆盖 FP4 舍入与 BF16 舍入），scale 字节与 golden
推导的 DYN_DTYPE_RANGE 指数允许 ±1 偏差（组内 amax 的 1 ulp BF16 差异）。

## 代码结构

```
flat_quant/
├── flat_quant.cpp     # kernel entry + host 侧 GM 管理
├── flat_quant.conf    # 参数路由配置
├── flat_quant.csv     # CSV 测试用例
└── README.md          # 本文档
```
