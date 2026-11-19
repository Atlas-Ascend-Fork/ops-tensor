# GroupedMatmul SwiGLU MX A8W4 示例

使用 Blaze 实例化 MX A8W4 分组矩阵乘、SwiGLU mode 2、可选专家 smoothScale 和 MX 输出量化。输入 x 为 E4M3FN，转置 WeightNZ 为 E2M1，输入及输出 scale 为 E8M0，输出 y 为 E4M3FN。

## 运行

在仓库根目录初始化依赖，加载 CANN 环境后使用仓库统一入口：

```bash
git submodule update --init --recursive
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash examples/acl/common/run.sh --ops=grouped_matmul --target=grouped_matmul_swiglu_mx_a8w4 --soc=ascend950
```

流程为 CSV → 数据生成及 CPU golden → 编译、运行 kernel → 校验 y/yScale。复用 `gen_gmmsq_data.py` 的 SwiGLU、MX 量化参考计算和 `verify_gmmsq_result.py` 的数值校验。

## 配置与覆盖

- CSV 中 n 是 SwiGLU 前的完整宽度；输出形状为 `(m,n/2)`。
- 仅支持转置 WeightNZ、单个连续权重存放所有专家；示例要求 K/N 不小于 256，且为 128 的整数倍。
- weight 的物理存储为 `(E,K/64,N/16,16,64)`，FP4 每字节两个元素；weightScale 为 `(E,N,K/64,2)`。
- 固定基础块 M=256、N=128，无 bias，clampLimit=7、gluAlpha=1.702、gluBias=1。
- 覆盖无 smoothScale、按专家及列变化的 smoothScale、count/cumsum、scaleAlg=0/1、多专家非均匀分组及 M 尾块。
- x 的首尾 K 位置非零，权重左右半区及相邻列有不同数值；部分输入触发 clamp。smoothScale 采用 2 的幂，使复用参考计算中的 BF16 舍入与 smoothing 的先后顺序数值等价。

新增 CSV 用例需满足程序中的范围检查：专家数为 1～8，M/K/N 不超过 4096，M 不小于专家数。样例范围不代表算子的完整支持范围。
