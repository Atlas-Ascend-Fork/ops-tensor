# grouped_matmul_no_quant

非量化 Grouped MatMul 样例，直接调用 Blaze Tensor API kernel（`blaze/gemm/kernel/kernel_grouped_matmul.h`），
覆盖 `aclnnGroupedMatmulV5`（Ascend 950）非量化场景矩阵。

## 1. 功能说明

| 维度 | 取值 | 说明 |
| ---- | ---- | ---- |
| groupType | -1 / 0 / 2 | 不分组（m-m-m）/ M 轴分组 / K 轴分组 |
| groupListType | 0 / 1 / 2 | 累积和（cumsum）/ 各组大小（count）/ 稀疏 `[E, 2]` 对（仅 M 轴分组） |
| weightFormat | ND / NZ | ND 连续排布 / FRACTAL_NZ 亲和排布 |
| dtype | float16 / bfloat16 / float32 | x、weight、bias、y 同 dtype |

### 场景组合（对齐 aclnnGroupedMatmulV5 Ascend 950 非量化支持表）

| groupType | x | weight | y | 说明 |
| :---: | :---: | :---: | :---: | ---- |
| -1 | 多 | 多 | 多 | 不分组，groupList 必须为空 |
| 0 | 单 | 单 | 单 | M 轴分组 s-s-s，weight 为 `[G, K, N]` 单 tensor |
| 0 | 单 | 多 | 单 | M 轴分组 s-m-s，weight 为 2D tensor list |
| 0 | 多 | 多 | 单 | M 轴分组 m-m-s，tiling 侧归一化为不分组语义（kernel 不读 groupList，M 取自各 x 描述符） |
| 2 | 单 | 单 | 单 | K 轴分组 s-s-s，x 转置为 `[K, M]`，y 为 `[G, M, N]`，不支持 bias |
| 2 | 单 | 多 | 多 | K 轴分组 s-m-m，y 为 2D tensor list，不支持 bias |

groupListType=2（稀疏 `[E, 2]`，非零组前置）仅支持 M 轴分组；M 轴分组的 s-s-s / s-m-s 按第一列实际组索引
定位 weight/bias，m-m-s 则完全走 dense 寻址。K 轴分组为空组（size=0）时由 AIV 路径补零输出。

## 2. 约束

- NZ weight 要求 `k % 16 == 0` 且 `n % (32 / dtypeSize) == 0`；**K 轴分组不支持 NZ weight**。
- K 轴分组不支持 bias；不分组（m-m-m）不传 groupList。
- M 轴分组 groupList 各组尺寸之和须等于 x 第一维；K 轴分组须等于 K。

## 3. 执行方法

```bash
# 从仓库根目录
./build.sh --examples --ops=grouped_matmul --target=grouped_matmul_no_quant

# 或直接使用统一入口
bash examples/acl/common/run.sh --ops=grouped_matmul --target=grouped_matmul_no_quant
```

## 4. 用例表（CSV）

| 用例 | 场景 |
| ---- | ---- |
| mgroup_sss_offset_nd_fp16_bias | M 分组 s-s-s，cumsum，ND，fp16，bias |
| mgroup_sss_count_nd_fp32 | M 分组 s-s-s，count，ND，fp32 |
| mgroup_sss_sparse_nd_fp16 | M 分组 s-s-s，稀疏（含空组截断），ND，fp16 |
| mgroup_sss_sparse_nz_bf16_bias | M 分组 s-s-s，稀疏，NZ weight，bf16，bias |
| mgroup_sms_count_nd_bf16_bias | M 分组 s-m-s，count，ND，bf16，bias |
| mgroup_sms_sparse_nd_fp16 | M 分组 s-m-s，稀疏（按实际组索引选 weight），ND，fp16 |
| mgroup_sms_offset_nz_fp16 | M 分组 s-m-s，cumsum，NZ weight，fp16 |
| mgroup_mms_count_nd_fp16 | M 分组 m-m-s，count，ND，fp16 |
| mgroup_mms_sparse_nd_fp16 | M 分组 m-m-s，稀疏 groupList（kernel 不读取，dense 回退），ND，fp16 |
| mgroup_mms_sparse_nz_bf16_bias | M 分组 m-m-s，稀疏 groupList，NZ weight，bf16，bias |
| kgroup_sss_offset_nd_fp16 | K 分组 s-s-s，cumsum，ND，fp16 |
| kgroup_sss_count_nd_bf16_empty | K 分组 s-s-s，count（含空组 AIV 补零），ND，bf16 |
| kgroup_smm_count_nd_fp16 | K 分组 s-m-m，count，ND，fp16 |
| nosplit_mmm_nd_fp16 | 不分组 m-m-m，ND，fp16 |
| nosplit_mmm_nz_bf16_bias | 不分组 m-m-m，NZ weight，bf16，bias |

## 5. 数据流

`gen_gmm_no_quant_data.py` 生成输入（含 NZ 帧重排、转置 x 存储、稀疏 groupList）与 NumPy golden；
样例可执行文件按 CSV 参数构造 tensor list（携带 shape 描述符）与 tiling 后 `<<<coreNum>>>` 直接启动
Blaze kernel；`verify_result.py` 将 NPU 输出与 golden 逐元素比对（混合绝对容差 + 错误率阈值）。
