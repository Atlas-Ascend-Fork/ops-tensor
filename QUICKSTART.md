# ops-tensor 快速入门

## 介绍

ops-tensor 内置了 **Blaze**（Basic Linear Algebra optimiZed Engine）基础线性代数加速引擎，为融合算子开发提供高性能 MM 公共能力，构建 NPU 通用高性能 CUBE 底座。

环境搭建一般分为如下场景，您可以按需安装：

- **编译态**：针对仅编译不运行本项目的场景，只需安装前置依赖和 CANN toolkit 包。
- **运行态**：针对运行本项目的场景（编译运行或纯运行），除了安装前置依赖和 CANN toolkit 包，还需安装驱动与固件、CANN ops 包。

## 环境准备

### 系统要求

ops-tensor 支持源码编译，进行源码编译前，请确保如下基础依赖、NPU 驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0（建议版本 <= 3.10）
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - make
   - googletest（仅执行 Kernel UT 时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）
2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作。

   单击[下载链接](https://www.hiascend.com/hardware/firmware-drivers/community)，根据实际产品型号和环境架构，获取对应的 `Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`、`Ascend-hdk-<chip_type>-npu-firmware_<version>.run` 包。

   安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》中"安装指南 > 安装NPU驱动和固件"。

### 支持的产品

- Ascend 950PR / Ascend 950DT

### 安装 CANN 包

#### 1. 下载软件包

单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新发布日期对应的目录，根据实际产品型号和环境架构，获取 `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`。

#### 2. 安装软件包

**安装社区版 CANN Toolkit 包**

```bash
# 需要确保安装目录权限至少为 755
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run

# 安装命令
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- `${cann_version}`：表示 CANN 包版本号。
- `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
- `${install_path}`：表示指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

## 环境验证

安装完 CANN 包或进入 Docker 容器后，需验证环境和驱动是否正常。

- **检查 NPU 设备**（仿真执行，跳过此步骤）：

  ```bash
  # 运行 npu-smi，若能正常显示设备信息，则驱动正常
  npu-smi info
  ```
- **检查 CANN 安装**：

  ```bash
  # 查看 CANN Toolkit 版本信息（非 root 用户，将 /usr/local 替换为 ${HOME}）
  cat /usr/local/Ascend/cann/opp/version.info
  ```

## 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以 root 用户为例（非 root 用户，将 /usr/local 替换为 ${HOME}）
source /usr/local/Ascend/cann/set_env.sh

# 指定路径安装
source ${install_path}/cann/set_env.sh
```

## 源码下载

```bash
# 下载项目源码
git clone -b master https://gitcode.com/cann/ops-tensor.git
cd ops-tensor
```

> [!NOTE] ✏️ 注意
> gitcode 平台在使用 SSH 协议时，请在本地生成 SSH 公钥进行克隆、推送等操作。

## 编译执行

### 编译项目

ops-tensor 提供一键式编译能力，使用 `build.sh` 脚本：

```bash
# 编译所有算子
./build.sh

# 指定编译线程数（默认 8）
./build.sh -j16

# 查看完整帮助信息
./build.sh --help
```

## Kernel-UT测试

Kernel UT（单元测试）用于验证算子内核的正确性。

### 1. 构建并执行所有 Kernel UT

```bash
./build.sh --opkernel -u
```

### 2. 执行指定算子的 Kernel UT

```bash
# 单个算子
./build.sh --opkernel -u --ops=mat_mul

# 多个算子
./build.sh --opkernel -u --ops=mat_mul,quant_batch_matmul
```

### 3. 指定 SoC 执行 Kernel UT

```bash
./build.sh --opkernel -u --soc=ascend950 --ops=mat_mul
```

### 4. 设置测试超时时间

```bash
# 默认超时 300 秒
./build.sh --opkernel -u --test-timeout=600
```

**支持的 Kernel UT 算子**（位于 `tests/ut/op_kernel/`）：

- `mat_mul` - 矩阵乘算子
- `fused_mat_mul` - 融合矩阵乘算子（MatMul + Scale + Add）
- `transpose_batch_mat_mul` - 转置批量矩阵乘算子
- `quant_batch_matmul` - 量化批量矩阵乘算子
- `quant_grouped_matmul` - 量化分组矩阵乘算子
- `weight_quant_batch_matmul_mx` - MXA8W4 权重量化矩阵乘组件

若提示如下信息，则说明 Kernel UT 测试通过：

```bash
[SUCCESS] Kernel UT all SoC tests passed
```

### 5. 未联网编译

未联网环境下编译执行 Kernel UT，需提前在联网环境备齐 `cann-cmake`、`gtest`、`tensor_api` 三项依赖并拷贝到离线环境。

#### 1. 准备 cann-cmake

`fetch_cann_cmake.cmake` 优先识别 `third_party/cann-cmake/` 目录，存在则直接使用（否则联网 git clone）。离线环境需手动克隆后整目录拷贝：

```bash
# 联网环境
git clone -b master-015 https://gitcode.com/cann/cmake.git cann-cmake
# 将 cann-cmake/ 整目录拷贝到离线环境 ops-tensor/third_party/cann-cmake/
```

#### 2. 准备 gtest

`gtest.cmake` 查找顺序：从 `third_party/gtest/googletest-1.14.0.tar.gz` 构建；若仅 `third_party/googletest-1.14.0.tar.gz` 存在则自动移入 `gtest/` 再构建；否则联网下载。离线环境放源码 tarball，首次执行 UT 时自动构建：

```bash
# 联网环境下载
wget -O googletest-1.14.0.tar.gz https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/googletest/googletest-1.14.0.tar.gz
# 拷贝到离线环境 ops-tensor/third_party/googletest-1.14.0.tar.gz
# 首次执行 UT 时会自动移入 third_party/gtest/ 并完成编译
```

#### 3. 手动拉取 tensor_api

`tensor_api` 是 `include/tensor_api` 下的 git 子模块，源仓库 `https://gitcode.com/cann/asc-devkit.git`。离线环境无法 `git submodule update`，需在联网环境拉取**当前 ops-tensor 提交固定的子模块版本**，再拷贝到离线环境。只克隆分支最新提交可能得到与本仓固定版本不同的源码：

```bash
# 联网环境，在 ops-tensor 仓库根目录执行
TENSOR_API_REV="$(git ls-tree HEAD include/tensor_api | awk '{print $3}')"
git clone -b feature/tensor_api_from_9.0.0 https://gitcode.com/cann/asc-devkit.git include/tensor_api
git -C include/tensor_api checkout "$TENSOR_API_REV"
test "$(git -C include/tensor_api rev-parse HEAD)" = "$TENSOR_API_REV"
# 拷贝到离线环境 ops-tensor/include/tensor_api/（注意路径在 include/ 下，非 third_party/）
```

或在联网环境通过子模块方式拉取后整目录拷贝：

```bash
cd ops-tensor
git submodule update --init --recursive include/tensor_api
# 将 include/tensor_api/ 整目录拷贝到离线环境对应位置
```

`build.sh`（`--opkernel -u` 或 `--examples`）执行前会检测 `include/tensor_api/` 是否已存在且非空，**已存在则自动跳过初始化和版本校验**。因此手动拉取时应先按上文切换到固定提交，再拷贝和编译。

#### 4. 离线编译执行

```bash
source /usr/local/Ascend/cann/set_env.sh

./build.sh --opkernel -u
# 或指定算子
./build.sh --opkernel -u --ops=mat_mul
```

> 若第三方依赖未放在默认的 `third_party/` 目录，可通过 `--cann_3rd_lib_path` 指定（`tensor_api` 仍须置于 `include/tensor_api/`）：
>
> ```bash
> ./build.sh --opkernel -u --cann_3rd_lib_path=/path/to/cann_3rd_lib_path
> ```

成功标志同上文：出现 `[SUCCESS] Kernel UT all SoC tests passed`。

更详细的 Kernel UT 开发流程请参阅 [算子开发指南](./QUICK_OP_INVOCATION.md)。

## Samples测试

Samples 测试用于验证算子样例在 NPU 上的编译、执行和精度正确性。通过 CSV 测试用例表驱动，自动完成编译、数据生成、kernel 执行和精度验证。

### 1. 执行所有样例

```bash
./build.sh --examples
```

`build.sh` 会自动发现 `examples/` 下所有样例目录中的同名 CSV 文件（`{example}.csv`），以 `--case` 参数委托各样例的 `run.sh` 执行。

### 2. 执行指定算子的样例

```bash
# 单个算子下的所有样例
./build.sh --examples --ops=mat_mul

# 指定样例
./build.sh --examples --ops=mat_mul --target=mat_mul_streamk
```

### 3. 通过 run.sh 执行

```bash
bash examples/common/run.sh --ops=mat_mul --target=mat_mul_streamk
```

### 4. CSV 测试用例格式

每个样例目录下有一个与样例同名的 CSV 文件，定义所有测试用例：

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,format
mat_mul_streamk_fp16,100,8192,100,100,float16,false,false,false,"(ND,ND)"
mat_mul_streamk_fp32,100,8192,100,100,float32,false,false,false,"(ND,ND)"
```

若提示如下信息，则说明 Samples 测试通过：

```bash
[SUCCESS] All examples operations completed!
```

### 5. 未联网编译

未联网环境下编译执行 Samples，仅需手动拉取 `tensor_api` 子模块。

#### 1. 手动拉取 tensor_api

`tensor_api` 的手动拉取步骤与 Kernel UT 一致，详见 [Kernel UT 未联网编译 · 手动拉取 tensor_api](#3-手动拉取-tensor_api)。

#### 2. 离线编译执行

```bash
source /usr/local/Ascend/cann/set_env.sh

./build.sh --examples
# 或指定算子/样例
./build.sh --examples --ops=mat_mul --target=mat_mul_streamk
```

成功标志同上文：出现 `[SUCCESS] All examples operations completed!`。

更详细的 Samples 使用说明请参阅 [Examples 文档](./examples/README.md)。

## 相关文档

- [项目整体介绍](./README.md) - 项目介绍
- [算子开发指南](./QUICK_OP_INVOCATION.md) - 详细的开发流程
- [Blaze 模块介绍](./include/blaze/README.md) - Blaze 引擎说明
- [API 文档](./docs/API/README.md) - API 详细说明
