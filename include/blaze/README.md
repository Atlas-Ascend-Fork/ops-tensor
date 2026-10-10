# Blaze

**Blaze**（**B**asic **L**inear **A**lgebra **O**ptimized **E**ngine）基础线性代数加速引擎，构建 NPU 高性能 CUBE 底座, 为算子开发提供 CUBE 公共能力。

## 定位与边界

- **聚焦 Kernel 端**：Blaze 只负责矩阵乘类算子的 Kernel 端计算组件（数据搬运、MMAD、调度等），不涉及 aclnn 入口与 Host 端逻辑。
- **职责分工**：算子的 Tiling 计算、内存规划、解决方案注册等 Host 端工作由各算子自身负责，Blaze 与之配合而非替代。
- **依赖关系**：Blaze 依赖 `include/tensor_api/`（编译期外部依赖，不在本仓内）提供的张量结构抽象（Layout / Shape / Coord 等），并直接对接 AscendC Kernel 接口。
- **目标算子**：服务于使用到矩阵乘计算的相关算子，包括 Matmul、GroupedMatmul、MC2 等。

## 设计理念

- **分层抽象**：从 Kernel（完整内核）到 Block（基本块计算）再到 Tile（细粒度搬运/计算指令），逐层下沉，关注点分离。
- **策略驱动**：通过 `DispatchPolicy` 将算法变体（如全载 / 非全载、量化模式、是否带 scale 等）作为类型参数派发到不同的 Block 实现，编译期完成最优实现的选择。
- **类型安全的组合**：A/B/C/Bias 的 dtype 与 Layout（NDExt / DNExt / NZ / ZN 等）作为类型参数透传，编译期生成最优代码路径。
- **充分利用 Cube 架构**：直接对接 L1 / L0A / L0B / L0C 的存储层级与 MMAD 指令，结合 double-buffer、ND2NZ 自动补零等机制压榨硬件性能。

## 模块组成

物理结构如下（各子目录下的具体文件随算法扩展而增减，下面仅给出代表性示例）：

```
blaze/
├──epilogue/   # Epilogue 层：后处理策略
|   └── block/
├──attention/  # Attention 层：FA 类混合 AIC/AIV 公共模板，与 GEMM 体系隔离
|   ├── kernel/    # Kernel 层：完整注意力内核入口
|   ├── block/     # Block 层：注意力 Block 抽象与调度
|   └── policy/    # Dispatch Policy：派发策略定义
└── gemm/      # 工具库
    ├── kernel/      # Kernel 层：完整算子内核入口
    ├── block/      # Block 层：Block 级矩阵乘抽象与调度
    ├── tile/       # Tile 层：细粒度搬运与计算原语
    ├── policy/     # Dispatch Policy：派发策略定义
    └── utils/      # 通用工具与常量
```

各层职责（自上而下）：

| 子目录 | 命名空间 | 职责 |
| :--- | :--- | :--- |
| `gemm/kernel/` | `Blaze::Gemm::Kernel` | 完整算子内核入口，组合 Block + Epilogue + Scheduler 形成可启动的 Kernel |
| `gemm/block/` | `Blaze::Gemm::Block` | Block 级 Mmad 抽象及其针对不同 Policy 的实现, 以及 Block 调度器 |
| `epilogue/` | `Blaze::Epilogue::Block`（历史兼容别名 `Blaze::Gemm::Block`） | 后处理策略，可按需扩展 Bias / 激活 / 反量化等 |
| `attention/kernel/` | `Blaze::Attention::Kernel` | 完整注意力内核入口，与 GEMM 的 `Kernel` 体系隔离 |
| `attention/block/` | `Blaze::Attention::Block` | 注意力 Block 抽象及其调度器 |
| `attention/policy/` | `Blaze::Attention` | 注意力派发策略定义 |
| `gemm/tile/` | `Blaze::Gemm::Tile` | Tile 级原语：定制化 MMAD Trait、定制化 GM->L1/ L1->L0搬运, 如 K 方向补零等 |
| `gemm/policy/`| `Blaze::Gemm` | 派发策略定义，控制全载模式、量化模式等行为 |
| `gemm/utils/` | `Blaze::Gemm` | 通用工具与常量：CeilDiv、Layout 推导、量化模式常量等 |
|

## API 接口文档
详细接口文档参考 [blaze接口文档](../../docs/API/README.md)
