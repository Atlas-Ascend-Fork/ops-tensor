# Block Epilogue GELU MX Quant
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_gelu_mx_quant.h)

## 功能说明
在 AIV 上读取 AIC 经 Fixpipe 写入 UB 的 FP32/BF16 结果，执行 GELU 激活、BF16 舍入和动态 MX 量化，输出到 GM。支持 tanh、erf 两种激活算法，以及下述三种量化算法。
<details>

<summary><strong>激活计算公式</strong></summary>

- GELU（tanh 近似）：

    $$
    out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
    $$

- GELU（erf）：

    $$
    out=GELU(self)=0.5 * self * (1 + erf(self / \sqrt{2}))
    $$

</details>

<details>

<summary><strong>动态量化计算公式</strong></summary>

- 场景1，当quantAlg为0时：
    - 将输入x在axis维度上按k = blocksize个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

    - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

- 场景2，当quantAlg为1时，只涉及FP8类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值:

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$

    - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值:

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
        $$

    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
- 场景3，当quantAlg为2时，只涉及FP4_E2M1类型：
    - 当dstTypeMax = 0.0/6.0/7.0时：
        - 将输入x在axis维度上按k = blocksize个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize：
        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
        $$
        $$
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
        $$
        - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
    - 当dstTypeMax != 0.0/6.0/7.0时：
        - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$
        - 将FP32映射到目标数据类型可表示的范围内，其中当dst_max_value=0时，$Amax(DType)$是目标精度能表示的最大值；当dst_max_value!=0时，$Amax(DType)$是dst_max_value传入值。
        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$
        - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
        - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
        - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
        $$
        - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
        - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
        - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

</details>

输出 y 支持 FLOAT8_E4M3FN / FLOAT8_E5M2 / FLOAT4_E2M1 / FLOAT4_E1M2，scale 类型为 FLOAT8_E8M0。GELU 和 MX 量化由 Tile 组件的 VF 实现；两个 AIV 沿 M 分工，每个 AIV 最多处理 128×256 个元素。

**关联框架**：[Block Epilogue 基础框架](./block_epilogue.md)
**上游计算**：[kernel_qbmm_mx_activation_quant](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)（AIC 写 L0C→UB）

## 特殊约束

### 计算位置
仅在 AIV 上执行；由 [Kernel](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md) 在 `WaitForCube()` 后调用。

### 输入 UB 复用通知

本组件在 GELU 结果写入 BF16 区后执行 `NotifyCube()`。
该通知允许 Fixpipe 复用从 UB 偏移 0 开始的 `MAX_SINGLE_MN*sizeof(DataTypeIn)` 字节输入区，FP32 输入时为 128 KiB。
每个 AIV/tile 通知一次，包括零行分支。Kernel 负责数据就绪通知及输入复用等待，详见 [Kernel 同步约定](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md#输入-ub-复用通知的所有权)。

输出与 scale 的搬出及复用由 `V_MTE3` / `MTE3_V` 保护，析构时等待最后一次搬出完成。

### 激活算法（GeluAlg）
| 算法 | 值 | 说明 |
|------|----|------|
| TANH | 0 | 默认，tanh 近似 |
| ERF | 1 | erf 算法 |

### 量化算法（QuantAlg）
| 算法 | 值 | 说明 |
|------|----|------|
| OCP | 0 | 默认，对应上述量化场景1 |
| BLAS | 1 | 对应上述量化场景2 |
| DYN_DTYPE_RANGE | 2 | 对应上述量化场景3 |

- BLAS模式只支持量化目的类型为FLOAT8_E4M3FN / FLOAT8_E5M2。

### 舍入模式（ROUND_MODE_FP4）
| 模式 | 值 | 说明 |
|------|----|------|
| RINT | 0 | 默认，四舍六入五成双舍入|
| FLOOR | 1 | 向负无穷舍入 |
| ROUND | 2 | 四舍五入 |

- 仅在量化目的类型为FLOAT4_E2M1 / FLOAT4_E1M2的时候生效，否则总是RINT模式。

### 对齐要求

- 输入 UB 从偏移 0 开始，行距为 `Align32(singleN)` 个元素；`operator()` 接收 tile 的实际 M/N。
- 非末尾 N tile 的起点和宽度须按 32 个元素对齐，末尾 tile 可不足 32 个元素。
- GELU 将 BF16 激活结果的无效尾部补零到 32 元素分组边界。
- 输出 scale 布局为 `[M, ceil(N/64), 2]`。有效 scale 列数为奇数时，由末尾 N tile 调用 `PadScaleRows` 写入配对补零字节。

## 特殊静态常量
| 常量 | 说明 |
|------|------|
| BLOCK_SIZE | 每32个元素做量化 |
| MAX_SINGLE_MN | 每个 AIV 的最大元素数，128×256 |
| MAX_SINGLE_SCALE_NUM | 每个 AIV 的最大 scale 数，MAX_SINGLE_MN / 32 |

## 特殊类型别名
| 类型 | 说明 |
|------|------|
| DataTypeOut | 输出类型（FLOAT8_E4M3FN / FLOAT8_E5M2 / FLOAT4_E2M1 / FLOAT4_E1M2） |
| DataTypeIn | 输入类型（FLOAT32 / BFLOAT16） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    uint32_t baseM{0};
    uint32_t baseN{0};
    GeluAlg geluAlg{GeluAlg::TANH};
    QuantAlg quantAlg{QuantAlg::OCP};
    ROUND_MODE_FP4 fp4RoundMode{ROUND_MODE_FP4::RINT};
    float dtypeMax{0.0F};  // FP4 自定义最大值
};
```

## 特殊成员方法

### 构造/析构函数
```
__aicore__ inline BlockEpilogueGeluMxQuant();
__aicore__ inline ~BlockEpilogueGeluMxQuant();
```

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
缓存激活、量化参数并设置输出地址。UB 使用固定偏移，每个 AIV 最多处理 128×256 个元素。

### operator函数
```
__aicore__ inline void operator()(
    const BlockShape& blockShape,   // 当前 tile 的实际 shape
    const BlockCoord& blockCoord);  // 输出偏移 {yOffset, yScaleOffset}
```
执行流程：

1. 两个 AIV 沿 M 分工；零行 AIV 通知输入区可复用后返回。
2. 计算 GELU，将结果舍入为 BF16 并补零尾组，然后通知输入区可复用。
3. 按每 32 个 BF16 元素生成 scale，执行 MX 量化。
4. 按需转换 FP4 输出布局，为末尾 scale 补零，写回 y 和 scale。

### UpdateGlobalAddr函数
```
__aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset)
```
按 `{yOffset, yScaleOffset}` 更新当前 batch 的 GM 输出基址。

### UpdateNextProblem函数
```
__aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape)
```
读取完整 N，更新每行 scale 数量和配对补零标记。

## 数据流
```
UB(fp32/bf16) ─Cast→ fp32 ─[GELU] ─Cast→ bf16 ─[mx quant] ─Cast→ OutType ─→ GM
```

## 适用场景

矩阵乘结果的 GELU 激活与动态 MX 量化后处理。
