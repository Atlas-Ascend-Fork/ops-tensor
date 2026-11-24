// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "platform/platform_ascendc.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/epilogue/block/block_epilogue_dual_level_scale.h"

constexpr uint32_t LEVEL1_SCALE_L1_WINDOW_K = 4096;

// Example-local file protocol. All dimensions count logical elements, never packed bytes.
struct Tiling {
    uint64_t version{1};
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t experts{0};
    // Protocol v1 metadata; device addressing is derived from the logical shape.
    uint64_t weightStrideBytes{0};
    // Scheduler/L1 tile; BlockMmad defines the dual-level L0 limits.
    uint64_t baseM{32};
    uint64_t baseN{64};
    // Number of K elements per level0 scale group.
    uint64_t level0Size{512};
    uint64_t coreNum{0};
    uint64_t transposeWeight{0};
    uint64_t groupListType{0};
    uint64_t hasBias{0};
    uint64_t outputBf16{0};
};
static_assert(sizeof(Tiling) == 14 * sizeof(uint64_t));
static_assert(offsetof(Tiling, level0Size) == 64);
static_assert(offsetof(Tiling, outputBf16) == 104);

template <class Output, bool TransposeWeight>
__global__ __aicore__ void DualLevelGmmExample(GM_ADDR x, GM_ADDR a0, GM_ADDR a1, GM_ADDR w, GM_ADDR b0, GM_ADDR b1,
                                               GM_ADDR groups, GM_ADDR bias, GM_ADDR y, Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    if (t.version != 1) {
        return;
    }
    using Nd = asc::te::nd_ext_layout_ptn;
    using WeightLayout = AscendC::Std::conditional_t<TransposeWeight, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithDualLevelScaleMx>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp4x2_e2m1_t, Nd, fp4x2_e2m1_t, WeightLayout, Output, Nd, float,
                                               Nd>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDualLevelScale<Output>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename Mmad::ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    typename Kernel::Params params{};
    params.problemShape = {t.m, t.n, t.k, 0};
    params.mmadParams.aGmAddr = x;
    params.mmadParams.bGmAddr = w;
    params.mmadParams.cGmAddr = y;
    params.mmadParams.biasGmAddr = bias;
    params.mmadParams.scaleAGmAddr = a1;
    params.mmadParams.scaleBGmAddr = b1;
    params.epilogueParams.level0ScaleAGmAddr = a0;
    params.epilogueParams.level0ScaleBGmAddr = b0;
    params.groupListGmAddr = groups;
    auto& gmm = params.gmmParams;
    gmm.groupNum = t.experts;
    gmm.m = t.m;
    gmm.n = t.n;
    gmm.k = t.k;
    gmm.baseM = t.baseM;
    gmm.baseN = t.baseN;
    gmm.baseK = t.level0Size;
    gmm.kAL1 = t.level0Size;
    gmm.kBL1 = t.level0Size;
    gmm.scaleKAL1 = LEVEL1_SCALE_L1_WINDOW_K;
    gmm.scaleKBL1 = gmm.scaleKAL1;
    gmm.isBias = t.hasBias;
    gmm.groupType = 0;
    gmm.groupListType = t.groupListType;
    gmm.singleW = 1;
    Kernel kernel;
    kernel(params);
}

void Check(aclError e)
{
    if (e != ACL_SUCCESS) {
        throw std::runtime_error("ACL error " + std::to_string(e));
    }
}
struct Runtime {
    aclrtStream stream{nullptr};
    bool initialized{false};
    bool deviceSet{false};

    void Init()
    {
        Check(aclInit(nullptr));
        initialized = true;
        Check(aclrtSetDevice(0));
        deviceSet = true;
        Check(aclrtCreateStream(&stream));
    }

    void Close()
    {
        Check(aclrtDestroyStream(stream));
        stream = nullptr;
        Check(aclrtResetDevice(0));
        deviceSet = false;
        Check(aclFinalize());
        initialized = false;
    }

    ~Runtime()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

std::vector<uint8_t> Read(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot read " + path);
    }
    const auto n = file.tellg();
    if (n < 0) {
        throw std::runtime_error("Cannot stat " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(n));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size())) {
        throw std::runtime_error("Short read " + path);
    }
    return data;
}
struct Buffers {
    std::vector<void*> pointers;
    ~Buffers()
    {
        for (auto p : pointers) {
            aclrtFree(p);
        }
    }
    uint8_t* Allocate(size_t n)
    {
        void* p = nullptr;
        Check(aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_FIRST));
        pointers.push_back(p);
        return static_cast<uint8_t*>(p);
    }
    uint8_t* Load(const std::string& path)
    {
        const auto bytes = Read(path);
        auto p = Allocate(bytes.size());
        Check(aclrtMemcpy(p, bytes.size(), bytes.data(), bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        return p;
    }
};
void RunExample(const std::string& dir, const Tiling& t, aclrtStream stream)
{
    Buffers buffers;
    auto x = buffers.Load(dir + "/x.bin"), a0 = buffers.Load(dir + "/a0.bin");
    auto a1 = buffers.Load(dir + "/a1.bin"), w = buffers.Load(dir + "/weight.bin");
    auto b0 = buffers.Load(dir + "/b0.bin"), b1 = buffers.Load(dir + "/b1.bin");
    auto groups = buffers.Load(dir + "/groups.bin");
    auto bias = t.hasBias ? buffers.Load(dir + "/bias.bin") : nullptr;
    const size_t outBytes = t.m * t.n * 2;
    auto y = buffers.Allocate(outBytes);
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        throw std::runtime_error("No platform");
    }
    const auto availableCores = platform->GetCoreNumAic();
    if (t.coreNum == 0 || t.coreNum > availableCores) {
        throw std::runtime_error("Requested AIC cores exceed the available range");
    }
#define LAUNCH(Output, Trans) \
    DualLevelGmmExample<Output, Trans><<<t.coreNum, nullptr, stream>>>(x, a0, a1, w, b0, b1, groups, bias, y, t)
    if (t.outputBf16) {
        if (t.transposeWeight) {
            LAUNCH(bfloat16_t, true);
        } else {
            LAUNCH(bfloat16_t, false);
        }
    } else {
        if (t.transposeWeight) {
            LAUNCH(half, true);
        } else {
            LAUNCH(half, false);
        }
    }
#undef LAUNCH
    Check(aclrtSynchronizeStream(stream));
    std::vector<uint8_t> output(outBytes);
    Check(aclrtMemcpy(output.data(), output.size(), y, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    std::ofstream file(dir + "/output.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(output.data()), output.size());
    if (!file) {
        throw std::runtime_error("Cannot write output");
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: dual_level_grouped_matmul CASE_DIR\n";
        return 2;
    }
    try {
        const std::string dir = argv[1];
        Tiling t;
        auto td = Read(dir + "/tiling.bin");
        if (td.size() != sizeof(t)) {
            throw std::runtime_error("Invalid tiling size");
        }
        std::memcpy(&t, td.data(), sizeof(t));
        Runtime runtime;
        runtime.Init();
        RunExample(dir, t, runtime.stream);
        runtime.Close();
        std::cout << "kernel completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
