# Block Epilogue FMM With Scale Add
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h)

## 功能说明
FusedMatMul scale_add场景的向量后处理Block，在 **AIV** 上执行。读取AIC经Fixpipe搬到UB的fp32累加结果和GM中的x3，完成缩放、相加和类型转换：

$$ out = alpha \times (x1@x2) + beta \times x3 $$

x1@x2累加器从UB读取，x3从GM读取；缩放和相加均以VF（vector-function）MicroAPI风格（`RegTensor` / `MaskReg` / `Cast` / `Interleave`）完成，最终输出bf16或fp16。

`alpha`和`beta`在`Params`中均为float，由调用方在进入Kernel前完成属性解析和类型转换。

**关联框架**：[Block Epilogue基础框架](./block_epilogue.md)
**上游计算**：[BlockMmadMatmulFixpipeOpti](../../gemm/block/block_mmad_matmul_fixpipe_opti.md)（AIC写L0C→UB）

## 特殊约束

### 计算位置
仅在AIV上执行，由[KernelMatmulWithScaleAdd](../../gemm/kernel/kernel_matmul_with_scale_add.md)（`GemmUniversal` 按 `KernelMmadFmmWithScaleAdd` 调度策略特化）在AIC完成当前Block的矩阵乘后调用。

### 数据类型
- L0C累加类型和向量计算类型固定为fp32。
- x3和输出使用相同的`ElementType_`，FusedMatMul场景支持bf16和fp16。
- x3的shape与输出一致，不支持广播。

### splitM子块
一个AIC对应两个AIV时，两个AIV按M轴切分任务。M为奇数时，AIV0比AIV1多处理一行；某个AIV没有有效行时仍完成ready/free同步，避免AIC等待同步信号时卡死。

Fixpipe在`DUAL_DST_SPLIT_M`前会对M轴补齐。累加器寻址使用补齐后的物理行跨度，x3和输出寻址使用各AIV负责的逻辑行偏移。

### UB复用
fp32累加器位于UB起始位置，x3和输出复用累加器后的同一块UB空间。组件通过MTE2/V/MTE3事件保证x3搬入、向量计算和输出写回的顺序。

## 特殊静态常量
| 常量 | 说明 |
|------|------|
| SPLIT_M_ALIGN | 非splitM场景M轴对齐粒度（2） |
| DATA_BLOCK | 数据块大小（32字节） |
| ACC_ALIGN | fp32累加器N轴对齐元素数 |
| ELEMENT_ALIGN | x3和输出N轴对齐元素数 |
| AIC_SYNC_AIV_FLAG / AIV_SYNC_AIC_FLAG | AIC与AIV的ready/free同步标志 |

## 特殊类型别名
| 类型 | 说明 |
|------|------|
| DispatchPolicy | BlockMmad调度策略 |
| L0CDataType | L0C累加类型，固定为float |
| X3Type / OutputType | x3和输出类型 |
| ComputeType | 向量计算类型，固定为float |
| BlockShape | 当前Block的(M, N, K, Batch)形状 |

## 特殊数据结构

### Params
```cpp
struct Params {
    GM_ADDR x3GmAddr{nullptr};
    GM_ADDR outputGmAddr{nullptr};
    float alpha{1.0F};
    float beta{1.0F};
};
```

## 特殊成员方法

### 构造/析构函数
```cpp
__aicore__ inline BlockEpilogueFmmWithScaleAdd()
__aicore__ inline ~BlockEpilogueFmmWithScaleAdd()
```
功能：构造时初始化x3/output UB复用事件；析构时等待最后一次UB→GM搬运完成。

### Init函数
```cpp
__aicore__ inline void Init(const Params& params, const ProblemShape& problemShape)
```
功能：记录x3和输出地址、alpha、beta及全局N；根据alpha、beta是否为1.0选择对应的向量计算分支。

### operator函数
```cpp
template <typename TensorC>
__aicore__ inline void operator()(
    TensorC& ubTensor,
    const BlockShape& blockShape,
    int64_t dstOffset,
    bool splitM,
    int64_t baseM,
    int64_t baseN,
    int64_t stagingBaseOffset = -1,
    int64_t stagingBytes = -1)
```
说明：`stagingBaseOffset`/`stagingBytes` 指定 x3 与输出复用的 staging 窗口（UB 偏移与字节数）；默认 `-1` 时窗口紧随 fp32 累加器之后并占用剩余 UB 空间，UB 中存放多个累加器镜像时由调用方显式传入。

执行流程：
1. 根据`splitM`和`subBlockIdx`计算当前AIV负责的行数及行偏移。
2. 按`baseN`遍历当前Block的N tile，并等待AIC Fixpipe就绪。
3. 根据剩余UB容量对M轴分段，将x3从GM搬入UB。
4. 以fp32执行alpha × accumulator + beta × x3，并转换为`ElementType_`。
5. 将输出写回GM，通知AIC当前缓冲区可复用。

## 数据流
```text
UB(fp32 accumulator) ─[× alpha]─┐
                                ├─Add─Cast→ bf16/fp16 ─→ GM
GM(x3) ─→ UB ─Cast→ fp32 ─[× beta]─┘
```

## 设计说明
- alpha和beta分别为默认值时，通过编译期模板分支跳过对应的标量乘法。
- x3和输出原地复用同一块UB，减少额外UB占用。
- `localRows=0`时只执行AIC/AIV同步，不发起无效的向量计算和搬运。

## 适用场景
- arch35 FusedMatMul的bf16/fp16 Batch Matmul + scale_add融合后处理。
- x1、x2、x3和输出均为连续ND格式，且Batch维度一致。
