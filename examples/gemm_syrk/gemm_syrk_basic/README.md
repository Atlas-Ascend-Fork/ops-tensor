# gemm_syrk_basic Example

## 概述

本示例演示基于 Blaze 框架的对称秩k更新（syrk）算子在昇腾 NPU 上的单次搬运实现：

```
C = alpha * (A @ A^T) + beta * C
```

其中 A 的 shape 为 `[batch, m, k]`，C 为 `[batch, m, m]` 的对称矩阵，输入输出同地址原地更新（cIn 与 c 绑定同一 device buffer，与 GemmSyrk aclnn 层的同地址绑定一致）。

- **算子**: gemm_syrk
- **场景**: gemm_syrk_basic（单次搬运 syrk 组装，MIX 1 AIC : 2 AIV）
- **算法特点**:
  - **单次搬运**：利用分形对偶性 `NZ(X)(m,k) ≡ ZN(X^T)(k,m)`（字节级相等），每个 A 行块每 k-chunk 仅一次 GM→L1 搬运，同一份 L1 镜像同时供给 L0A 与 L0B，总搬运量约为通用矩阵乘组合的一半；
  - **计算减半**：仅遍历紧凑上三角槽位（`BlockSchedulerSyrkTriangular`），每槽位单条 Mmad 链产出 `C[i,j]`，AIV sub0 以 nz2nd fixpipe 写出 `(i,j)`、AIV sub1 以 nz2dn fixpipe（硬件转置）写出镜像 `(j,i)`，无 GM 回读、无软件转置；
  - **转置布局**：`trans=true` 时 A 以转置的 `[batch, k, m]` 布局存储，kernel 绑定 DNExt `(m, k)` GM 视图，计算 `C = alpha * (A^T @ A) + beta * C`。
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_syrk.h`（对应 ops-nn 仓 `matmul/gemm_syrk` 算子）

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[batch, m, k]`（trans=false）或 `[batch, k, m]`（trans=true）
- 输入/输出 C shape: `[batch, m, m]`，**必须对称**（原地更新语义要求，数据生成脚本会自动生成对称 C）
- 数据类型: float16, bfloat16
- 设备要求 MIX 1 AIC : 2 AIV（`aivNum == aicNum * 2`）
- 对称方块 tiling 契约（host 侧 tiling 已强制）：`baseBlock == baseM == baseN == mL1 == nL1`（16 对齐）、
  `baseBlock^2 * 4B <= L0C_SIZE / 2`、`2 * baseBlock * kL1 * dtypeSize <= L1_SIZE / 2`、
  `baseBlock * baseK * dtypeSize <= L0A_SIZE / 2`、每轴单尾块（不做核间尾块切分）

## CSV 驱动测试

### 执行方式

```bash
bash examples/common/run.sh --ops=gemm_syrk --target=gemm_syrk_basic
```

### 测试用例定义

```csv
casename,m,k,batch,alpha,beta,dtype,trans
gemm_syrk_basic_fp16,128,256,1,1.0,1.0,float16,false
gemm_syrk_basic_bf16,128,256,1,1.0,1.0,bfloat16,false
gemm_syrk_alpha_beta_fp16,256,512,1,2.0,0.5,float16,false
gemm_syrk_batch_fp16,64,128,4,1.0,1.0,float16,false
gemm_syrk_tail_fp16,200,40,2,1.5,0.5,float16,false
gemm_syrk_trans_fp16,128,256,1,1.0,1.0,float16,true
gemm_syrk_trans_bf16,64,128,2,2.0,1.0,bfloat16,true
```

## 代码结构

```
gemm_syrk/
├── CMakeLists.txt                  # 构建配置
├── scripts/
│   ├── gen_data.py                 # 数据生成（对称 C + fp32 域 golden）
│   └── verify_result.py            # 精度 + 对称性校验
└── gemm_syrk_basic/
    ├── gemm_syrk_basic.cpp         # kernel + host 实现
    ├── gemm_syrk_basic.conf        # 参数路由配置
    ├── gemm_syrk_basic.csv         # CSV 测试用例
    └── README.md                   # 本文档
```

运行通过 `examples/common/run.sh` 统一调度，数据生成和精度校验由 `examples/gemm_syrk/scripts/` 下的脚本执行。

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_syrk.h` | syrk kernel（上三角槽位遍历 + 双 AIV epilogue） |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_syrk.h` | 单次搬运 Block 级矩阵乘（L1 双视图） |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_syrk.h` | 紧凑上三角调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h` | `alpha * acc + beta * C` 缩放累加 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulSyrk（路由到 GemmUniversal syrk 特化） |

## Tiling 说明

host 侧 `ComputeTiling` 按硬件预算推导对称方块（`examples/gemm_syrk/gemm_syrk_basic/gemm_syrk_basic.cpp`）：

- `l0cBlockMax = floorSqrt16(L0C_SIZE / 2 / sizeof(float))`（单累加器位于一个 L0C 槽）
- `l1BlockMax = floor16(L1_SIZE / 2 / (2 * dtypeSize * kL1))`（A+B 镜像适配 L1 双缓冲半区）
- `l0BlockMax = floor16(L0A_SIZE / 2 / (dtypeSize * baseK))`（L0A ping-pong 槽）
- `baseBlock = clamp(min(128, min(mAlign, 各硬件上限)), 16, ...)`

固定参数：`baseK = 64`、`kL1 = 128`、L1 双缓冲（`l1Stages = 2`）。
