# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为"简单的bug修复"，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献新算子

如果您有全新的算子希望基于 NPU 进行设计与实现，欢迎在 Issue 中提出您的想法与设计方案。完整的贡献流程如下：

### 1. 新增 Issue，创建需求

新建 `Requirement|需求建议` 类 Issue，并在其中说明新增算子的设计方案。

Issue 需包含以下内容：

- **背景信息**
- **价值/作用**
- **设计方案**

同时，请在提交的 Issue 中评论`/assign @yourself` 认领该任务。

### 2. 需求评审

Sig组将指派Committer对您提交的 Issue 进行评审并给出修改意见。请在完成修改后，于 Issue 中@对应Committer。

若需求被接纳，sig成员将为您分配合适的算子分类路径，以便您将贡献的算子提交至对应目录。

### 3. 提交 PR

本仓为 header-only 的 Blaze GEMM 框架，算子实现分散在 `include/blaze/`（Kernel 侧组件）、`docs/API/`（接口文档）、`examples/`（端到端样例）和 `tests/ut/op_kernel/`（Kernel UT）四个目录中。新算子交付件如下：

```
include/blaze/                           # Blaze 框架 header-only 组件（Kernel 侧）
├── gemm/
│   ├── kernel/                          # Kernel 层：完整算子内核入口
│   │   └── kernel_${op}_${variant}.h    #   新算子的 kernel 头文件
│   ├── block/                           # Block 层：BlockMmad + BlockScheduler
│   │   ├── block_mmad_${op}_${variant}.h         #   新算子的 BlockMmad 实现
│   │   └── block_scheduler_${op}_${variant}.h    #   新算子的 BlockScheduler 实现
│   ├── tile/                            # Tile 层：细粒度搬运与计算原语
│   │   ├── ${op}_${tile_func}.h         #   新算子需要的 Tile 原语（按需）
│   │   └── arch35/                      #   Ascend950 特有 Tile 实现（按需）
│   ├── policy/
│   │   └── dispatch_policy.h            # DispatchPolicy：新增策略在此扩展
│   └── utils/                           # 通用工具（CeilDiv、Layout 推导等，按需扩展）
├── epilogue/
│   ├── block/                           # Epilogue 层：后处理策略
│   │   └── block_epilogue_${op}_${variant}.h  #   新算子的 Epilogue（按需，无后处理用 BlockEpilogueEmpty）
│   └── fusion/                          # Fusion 算子（按需）
└── README.md                            # Blaze 框架说明

docs/API/                                # Blaze 接口文档
├── gemm/
│   ├── kernel/                          # Kernel 层 API 文档（按新增组件补充）
│   ├── block/                           # Block 层 API 文档（按新增组件补充）
│   └── tile/                            # Tile 层 API 文档（按新增组件补充）
├── epilogue/                            # Epilogue 层 API 文档（按需）
└── README.md                            # Blaze 接口总览

examples/acl/                            # 端到端样例（Host 侧 runner + device kernel 实例化）
└── ${op}/                               # 算子级目录
    ├── CMakeLists.txt                   #   通过 ops_example_add_executable 注册样例
    ├── scripts/
    │   ├── gen_data.py                  #   数据生成 + CPU golden 计算
    │   └── verify_result.py             #   NPU 输出 vs CPU golden 精度比对
    └── ${op}_${variant}/                #   样例级目录
        ├── ${op}_${variant}.cpp         #   样例源码（host 侧 ACL 内存管理 + kernel 调度）
        ├── ${op}_${variant}.conf        #   执行参数配置（INI 格式，见 examples/acl/common/CONF_FORMAT.md）
        ├── ${op}_${variant}.csv         #   CSV 测试用例表
        └── README.md                    #   样例说明文档

tests/ut/op_kernel/                      # Kernel UT（基于 Google Test + tikicpulib 模拟器）
└── ${op}/                               # 算子级 UT 目录
    ├── CMakeLists.txt                   #   AddOpTestCase(${op} "ascend950pr_9599" "编译选项")
    ├── test_${op}.cpp                   #   gtest 测试用例（TEST_F + KERNEL_RUN_KF）
    ├── ${op}.cpp                        #   变体统一入口（if constexpr 按 OP_TYPE 分发）
    ├── ${op}_${variant}.h               #   变体 wrapper（host → kernel 桥接）
    └── ${op}_tiling_data.h              #   Tiling 数据结构定义
```

> 文件命名遵循《[编程规范](CODING_CONVENTIONS.md)》的 `层级_算子_任务类型_策略` 约定（如 `kernel_matmul_streamk.h`、`block_mmad_qbmm_mx.h`）。Blaze 为 header-only 库，Kernel 侧组件以 `.h` 头文件形式提供。

**各目录职责**：

| 目录                                            | 放什么                                                              | 必选/按需            |
| ----------------------------------------------- | ------------------------------------------------------------------- | -------------------- |
| `include/blaze/gemm/kernel/`                  | 新算子的 Kernel 层入口（组合 Block + Epilogue + Scheduler）         | 必选                 |
| `include/blaze/gemm/block/`                   | 新算子的 BlockMmad 和 BlockScheduler 实现                           | 必选                 |
| `include/blaze/gemm/policy/dispatch_policy.h` | 新算子的 DispatchPolicy 策略定义（扩展已有文件）                    | 必选                 |
| `include/blaze/gemm/tile/`                    | 新算子需要的 Tile 原语（数据搬运、补零、Scale 处理等）              | 按需                 |
| `include/blaze/epilogue/block/`               | 新算子的 Epilogue 后处理；无后处理时复用`BlockEpilogueEmpty`      | 按需                 |
| `include/blaze/gemm/utils/`                   | 新算子需要的通用工具（Layout 推导、常量等）                         | 按需                 |
| `docs/API/`                                   | 新增 Kernel/Block/Tile/Epilogue 组件的接口文档                      | 必选（有新增组件时） |
| `examples/acl/${op}/`                         | 端到端样例（含 gen_data.py、verify_result.py、CSV 用例、README.md） | 必选                 |
| `tests/ut/op_kernel/${op}/`                   | Kernel UT（gtest + tikicpulib 模拟器执行）                          | 必选                 |

代码上库要求：

- 代码交付件：需包含 Blaze 组件头文件（`include/blaze/`）、接口文档（`docs/API/`）、端到端样例（`examples/`，含 README.md 说明算子功能与参数）、Kernel UT（`tests/ut/op_kernel/`）
- 是否签署 CLA
- PR 是否已关联对应 Issue
- 代码是否符合《[C++ 编程规范](<https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md>)》
- 代码是否符合本仓《[编程规范](CODING_CONVENTIONS.md)》（命名约定、编码约束、头文件自包含等）
- 代码是否编译通过（`bash build.sh --examples --ops=${op}` 样例编译 + `bash build.sh --opkernel -u --ops=${op}` UT 编译）
- 新增功能是否补充对应 UT（见《编程规范》2.11 UT 要求：新增功能必须补充 UT，已有 UT 不能修改）

### 4. CI门禁

通过评论 `compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改，目前CI门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及codecheck误报，请提交给sig成员屏蔽）
- UT测试
- 冒烟测试

门禁通过后，请在关联的 Issue 中@指派的Committer。

### 5. Committer检视

Committer检视后将反馈检视意见，请完成所有修改后@指派的Committer。

### 6. Maintainer检视合入

Committer 检视通过后，标注 `/lgtm`标签。Maintainer 将在1天内进行最终审核，确认无问题后，将标注 `/approve` 标签合入PR。

## 二、算子Bug修复

如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行处理。

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行跟踪优化。

## 四、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您纠正对应文档描述。

## 五、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您，跟踪协助解决问题。
