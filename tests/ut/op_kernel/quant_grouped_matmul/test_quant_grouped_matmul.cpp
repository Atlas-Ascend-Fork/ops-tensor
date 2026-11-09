/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * \file test_quant_grouped_matmul.cpp
 * \brief QGMM MX kernel smoke tests.
 */
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "quant_grouped_matmul_mx.h"

namespace {
constexpr int64_t M0 = 16;
constexpr int64_t M1 = 24;
constexpr int64_t N = 64;
constexpr int64_t K = 64;
constexpr uint32_t GROUP_NUM = 2;
constexpr uint32_t BLOCK_NUM = 1;
constexpr size_t SCALE_K = 2;

template <typename T>
inline constexpr bool IS_FP4_TYPE = std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;

void GenerateData(const std::string& dataDir, const std::string& dtypeA, const std::string& dtypeB,
                  const std::string& weightFormat, bool multiTensor, bool withBias, const std::string& groupListType)
{
    const std::string cleanCmd = "cd " + dataDir + " && rm -f *.bin";
    ASSERT_EQ(system(cleanCmd.c_str()), 0) << "Failed to clean QGMM test data";
    std::string genCmd = "cd " + dataDir + " && python3 gen_data.py --group_m " + std::to_string(M0) + " " +
                         std::to_string(M1) + " --n " + std::to_string(N) + " --k " + std::to_string(K) + " --seed 42" +
                         " --dtype_a " + dtypeA + " --dtype_b " + dtypeB + " --weight_format " + weightFormat +
                         " --group_list_type " + groupListType;
    if (multiTensor) {
        genCmd += " --multi_tensor";
    }
    if (withBias) {
        genCmd += " --with_bias";
    }
    ASSERT_EQ(system(genCmd.c_str()), 0) << "Failed to generate QGMM test data";
}

void ReadBinToGm(const std::string& path, GM_ADDR addr, size_t size)
{
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << path;
    file.read(reinterpret_cast<char*>(addr), static_cast<std::streamsize>(size));
    ASSERT_EQ(file.gcount(), static_cast<std::streamsize>(size)) << "Unexpected file size: " << path;
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (ptr_ != nullptr)
            AscendC::GmFree(ptr_);
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_{};
};

template <typename T>
size_t MxDataSize(size_t elementCount)
{
    if constexpr (IS_FP4_TYPE<T>) {
        return (elementCount + 1UL) / 2UL;
    }
    return elementCount * sizeof(T);
}

struct WeightBuffers {
    std::unique_ptr<GmBuffer> b;
    std::unique_ptr<GmBuffer> scaleB;
    std::vector<std::unique_ptr<GmBuffer>> bTensors;
    std::vector<std::unique_ptr<GmBuffer>> scaleBTensors;
};

void LoadCommonCaseData(const std::string& dataDir, GmBuffer& a, GmBuffer& scaleA, GmBuffer& bias, GmBuffer& c,
                        GmBuffer& groupList, size_t aSize, size_t scaleASize, size_t groupListItems)
{
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", a.Get(), aSize));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale_a.bin", scaleA.Get(), scaleASize));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", bias.Get(), GROUP_NUM * N * sizeof(float)));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/output_c.bin", c.Get(), (M0 + M1) * N * sizeof(float)));
    ASSERT_NO_FATAL_FAILURE(
        ReadBinToGm(dataDir + "/group_list.bin", groupList.Get(), groupListItems * sizeof(int64_t)));
}

template <bool MultiTensor>
void PrepareWeightData(const std::string& dataDir, size_t bGroupSize, size_t scaleBGroupSize, WeightBuffers& buffers)
{
    if constexpr (!MultiTensor) {
        buffers.b = std::make_unique<GmBuffer>(GROUP_NUM * bGroupSize);
        buffers.scaleB = std::make_unique<GmBuffer>(GROUP_NUM * scaleBGroupSize);
        ASSERT_NE(buffers.b->Get(), nullptr);
        ASSERT_NE(buffers.scaleB->Get(), nullptr);
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", buffers.b->Get(), GROUP_NUM * bGroupSize));
        ASSERT_NO_FATAL_FAILURE(
            ReadBinToGm(dataDir + "/scale_b.bin", buffers.scaleB->Get(), GROUP_NUM * scaleBGroupSize));
        return;
    }
    buffers.b = std::make_unique<GmBuffer>((GROUP_NUM + 1UL) * sizeof(uint64_t));
    buffers.scaleB = std::make_unique<GmBuffer>((GROUP_NUM + 1UL) * sizeof(uint64_t));
    ASSERT_NE(buffers.b->Get(), nullptr);
    ASSERT_NE(buffers.scaleB->Get(), nullptr);
    auto* bList = reinterpret_cast<uint64_t*>(buffers.b->Get());
    auto* scaleBList = reinterpret_cast<uint64_t*>(buffers.scaleB->Get());
    bList[0] = sizeof(uint64_t);
    scaleBList[0] = sizeof(uint64_t);
    for (uint32_t i = 0; i < GROUP_NUM; ++i) {
        buffers.bTensors.emplace_back(std::make_unique<GmBuffer>(bGroupSize));
        buffers.scaleBTensors.emplace_back(std::make_unique<GmBuffer>(scaleBGroupSize));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b_" + std::to_string(i) + ".bin",
                                            buffers.bTensors.back()->Get(), bGroupSize));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale_b_" + std::to_string(i) + ".bin",
                                            buffers.scaleBTensors.back()->Get(), scaleBGroupSize));
        bList[i + 1] = reinterpret_cast<uint64_t>(buffers.bTensors.back()->Get());
        scaleBList[i + 1] = reinterpret_cast<uint64_t>(buffers.scaleBTensors.back()->Get());
    }
}

template <bool MultiTensor>
void FillTiling(QGMMUT::QgmmTilingData& tiling, bool withBias, uint8_t dbL0C, uint8_t groupListType)
{
    tiling = {GROUP_NUM,
              M0,
              N,
              K,
              16,
              64,
              64,
              64,
              64,
              2,
              2,
              static_cast<uint8_t>(withBias),
              dbL0C,
              2,
              0,
              groupListType,
              static_cast<uint8_t>(MultiTensor ? 0U : 1U)};
}

template <typename AType, typename BType, typename LayoutB, bool MultiTensor = false>
void RunQgmmCase(const std::string& dtypeA, const std::string& dtypeB, const std::string& weightFormat,
                 uint8_t groupListType = 1, bool withBias = false, uint8_t dbL0C = 1)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    constexpr size_t totalM = M0 + M1;
    constexpr bool blockedWeight = std::is_same_v<LayoutB, asc::te::nz_layout_ptn> ||
                                   std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    constexpr size_t c0 = IS_FP4_TYPE<BType> ? 64UL : 32UL;
    constexpr size_t alignedN = blockedWeight ? ((N + 15UL) / 16UL * 16UL) : N;
    constexpr size_t alignedK = blockedWeight ? ((K + c0 - 1UL) / c0 * c0) : K;
    const size_t aSize = MxDataSize<AType>(totalM * K);
    const size_t bGroupSize = MxDataSize<BType>(alignedK * alignedN);
    const size_t scaleAGroupSize = totalM * SCALE_K * sizeof(fp8_e8m0_t);
    const size_t scaleBGroupSize = N * SCALE_K * sizeof(fp8_e8m0_t);

    GmBuffer a(aSize);
    GmBuffer bias(GROUP_NUM * N * sizeof(float));
    GmBuffer c(totalM * N * sizeof(float));
    const size_t groupListItems = groupListType == 2 ? GROUP_NUM * 2UL : GROUP_NUM;
    GmBuffer groupList(groupListItems * sizeof(int64_t));
    GmBuffer tiling(sizeof(QGMMUT::QgmmTilingData));
    GmBuffer scaleA(scaleAGroupSize);
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(scaleA.Get(), nullptr);
    ASSERT_NE(bias.Get(), nullptr);
    ASSERT_NE(c.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    const std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_grouped_matmul/quant_grouped_matmul_data";
    const std::string groupListName = groupListType == 0 ? "offset" : (groupListType == 2 ? "sparse" : "length");
    ASSERT_NO_FATAL_FAILURE(GenerateData(dataDir, dtypeA, dtypeB, weightFormat, MultiTensor, withBias, groupListName));
    ASSERT_NO_FATAL_FAILURE(
        LoadCommonCaseData(dataDir, a, scaleA, bias, c, groupList, aSize, scaleAGroupSize, groupListItems));
    WeightBuffers weights;
    ASSERT_NO_FATAL_FAILURE(PrepareWeightData<MultiTensor>(dataDir, bGroupSize, scaleBGroupSize, weights));

    auto* t = reinterpret_cast<QGMMUT::QgmmTilingData*>(tiling.Get());
    FillTiling<MultiTensor>(*t, withBias, dbL0C, groupListType);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = qgmm_mx_kernel_entry<AType, BType, float, float, LayoutA, LayoutB>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, a.Get(), weights.b->Get(), scaleA.Get(), weights.scaleB->Get(), bias.Get(),
                              c.Get(), groupList.Get(), tiling.Get()));
}

template <typename T>
constexpr uint8_t MxOneByte()
{
    if constexpr (IS_FP4_TYPE<T>) {
        return 0x22U;
    }
    if constexpr (std::is_same_v<T, fp8_e5m2_t>) {
        return 0x3CU;
    }
    return 0x38U;
}

void FillShapeGroupList(GmBuffer& groupList, uint32_t groupNum, int64_t splitSize, uint8_t groupListType)
{
    auto* data = reinterpret_cast<int64_t*>(groupList.Get());
    for (uint32_t i = 0; i < groupNum; ++i) {
        if (groupListType == 2) {
            data[i * 2UL] = i;
            data[i * 2UL + 1UL] = splitSize;
        } else {
            data[i] = groupListType == 0 ? static_cast<int64_t>(i + 1U) * splitSize : splitSize;
        }
    }
}

template <typename MxType, typename LayoutA, typename LayoutB>
void RunQgmmShapeCase(uint32_t e, int64_t m, int64_t n, int64_t k, uint8_t l1BufferStage = 2, uint8_t groupListType = 1)
{
    constexpr bool transA = std::is_same_v<LayoutA, asc::te::dn_ext_layout_ptn>;
    constexpr bool transB = std::is_same_v<LayoutB, asc::te::dn_ext_layout_ptn> ||
                            std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    constexpr bool weightNz = std::is_same_v<LayoutB, asc::te::nz_layout_ptn> ||
                              std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    const size_t c0 = IS_FP4_TYPE<MxType> ? 64UL : 32UL;
    const size_t storedK = weightNz ? (transB ? ((k + c0 - 1UL) / c0 * c0) : ((k + 15UL) / 16UL * 16UL)) : k;
    const size_t storedN = weightNz ? (transB ? ((n + 15UL) / 16UL * 16UL) : ((n + c0 - 1UL) / c0 * c0)) : n;
    const size_t scaleK = static_cast<size_t>((k + 63) / 64) * 2UL;
    const size_t aSize = MxDataSize<MxType>(static_cast<size_t>(e) * m * k);
    const size_t bSize = MxDataSize<MxType>(static_cast<size_t>(e) * storedK * storedN);
    const size_t scaleFactor = transA ? 2UL : 1UL;
    GmBuffer a(aSize);
    GmBuffer b(bSize);
    GmBuffer scaleA(static_cast<size_t>(e) * m * scaleK * scaleFactor);
    GmBuffer scaleB(static_cast<size_t>(e) * n * scaleK * scaleFactor);
    GmBuffer bias(static_cast<size_t>(e) * n * sizeof(float));
    GmBuffer c(static_cast<size_t>(e) * m * n * sizeof(float));
    const size_t groupListItems = groupListType == 2 ? static_cast<size_t>(e) * 2UL : e;
    GmBuffer groupList(groupListItems * sizeof(int64_t));
    GmBuffer tiling(sizeof(QGMMUT::QgmmTilingData));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(scaleA.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(c.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    std::fill_n(reinterpret_cast<uint8_t*>(a.Get()), aSize, MxOneByte<MxType>());
    std::fill_n(reinterpret_cast<uint8_t*>(b.Get()), bSize, MxOneByte<MxType>());
    std::fill_n(reinterpret_cast<uint8_t*>(scaleA.Get()), static_cast<size_t>(e) * m * scaleK * scaleFactor, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(scaleB.Get()), static_cast<size_t>(e) * n * scaleK * scaleFactor, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(c.Get()), static_cast<size_t>(e) * m * n * sizeof(float), 0U);
    const int64_t splitSize = transA ? k : m;
    FillShapeGroupList(groupList, e, splitSize, groupListType);
    auto* t = reinterpret_cast<QGMMUT::QgmmTilingData*>(tiling.Get());
    *t = {e, m, n, k, 16, 64, 64, 64, 64, 2, 2, 0, 1, l1BufferStage, 0, groupListType, 1};
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = qgmm_mx_kernel_entry<MxType, MxType, float, float, LayoutA, LayoutB>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, a.Get(), b.Get(), scaleA.Get(), scaleB.Get(), bias.Get(), c.Get(),
                              groupList.Get(), tiling.Get()));
}

// K-axis grouping: A is DN (physical [totalK, m]) and each group contributes its own k_g.
// The scale layout holds totalK/64 + groupNum rows, plus one spare row for the last group's
// tail over-read. The example validates numerics; this UT exercises the FP4 K-group branch,
// including the padded A L1 buffer allocation (baseM < C0).
template <typename MxType>
void RunQgmmKGroupCase(uint32_t e, int64_t m, int64_t n, const std::vector<int64_t>& kGroups, uint8_t l1BufferStage = 2)
{
    ASSERT_EQ(kGroups.size(), static_cast<size_t>(e));
    int64_t totalK = 0;
    for (int64_t groupK : kGroups) {
        ASSERT_GT(groupK, 0);
        totalK += groupK;
    }
    const size_t scaleSlots = static_cast<size_t>(totalK / 64 + static_cast<int64_t>(e) + 1);
    const size_t aSize = MxDataSize<MxType>(static_cast<size_t>(m) * static_cast<size_t>(totalK));
    const size_t bSize = MxDataSize<MxType>(static_cast<size_t>(totalK) * static_cast<size_t>(n));
    const size_t scaleASize = scaleSlots * static_cast<size_t>(m) * 2UL;
    const size_t scaleBSize = scaleSlots * static_cast<size_t>(n) * 2UL;
    GmBuffer a(aSize);
    GmBuffer b(bSize);
    GmBuffer scaleA(scaleASize);
    GmBuffer scaleB(scaleBSize);
    GmBuffer bias(static_cast<size_t>(e) * static_cast<size_t>(n) * sizeof(float));
    GmBuffer c(static_cast<size_t>(e) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float));
    GmBuffer groupList(static_cast<size_t>(e) * sizeof(int64_t));
    GmBuffer tiling(sizeof(QGMMUT::QgmmTilingData));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(scaleA.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(c.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    std::fill_n(reinterpret_cast<uint8_t*>(a.Get()), aSize, MxOneByte<MxType>());
    std::fill_n(reinterpret_cast<uint8_t*>(b.Get()), bSize, MxOneByte<MxType>());
    std::fill_n(reinterpret_cast<uint8_t*>(scaleA.Get()), scaleASize, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(scaleB.Get()), scaleBSize, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(c.Get()),
                static_cast<size_t>(e) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float), 0U);
    auto* groupListData = reinterpret_cast<int64_t*>(groupList.Get());
    for (uint32_t i = 0; i < e; ++i) {
        groupListData[i] = kGroups[i];
    }

    auto* t = reinterpret_cast<QGMMUT::QgmmTilingData*>(tiling.Get());
    // groupType=2 selects the K-split axis from LayoutA (DN); scaleKAL1/scaleKBL1 are K lengths.
    *t = {e, m, n, totalK, 16, 64, 128, 128, 128, 128, 128, 0, 1, l1BufferStage, 2, 1, 1};
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto
        fn = qgmm_mx_kernel_entry<MxType, MxType, float, float, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, a.Get(), b.Get(), scaleA.Get(), scaleB.Get(), bias.Get(), c.Get(),
                              groupList.Get(), tiling.Get()));
}

class QgmmMxKernelTest : public testing::Test {
protected:
    static void TearDownTestCase()
    {
        const std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_grouped_matmul/quant_grouped_matmul_data";
        const std::string cleanCmd = "cd " + dataDir + " && rm -f *.bin";
        (void)system(cleanCmd.c_str());
    }
};

TEST_F(QgmmMxKernelTest, MxFp8NdSingleTensor)
{
    RunQgmmCase<fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn>("mxfp8_e4m3", "mxfp8_e4m3", "nd");
}

TEST_F(QgmmMxKernelTest, MxFp8E5m2NdSingleTensor)
{
    RunQgmmCase<fp8_e5m2_t, fp8_e5m2_t, asc::te::nd_ext_layout_ptn>("mxfp8_e5m2", "mxfp8_e5m2", "nd");
}

TEST_F(QgmmMxKernelTest, MxFp4NdSingleTensor)
{
    RunQgmmCase<fp4x2_e2m1_t, fp4x2_e2m1_t, asc::te::nd_ext_layout_ptn>("mxfp4_e2m1", "mxfp4_e2m1", "nd");
}

TEST_F(QgmmMxKernelTest, MxFp4E1m2NzSingleTensor)
{
    RunQgmmCase<fp4x2_e1m2_t, fp4x2_e1m2_t, asc::te::nz_layout_ptn>("mxfp4_e1m2", "mxfp4_e1m2", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp4DnWeightOffsetListWithBias)
{
    RunQgmmCase<fp4x2_e2m1_t, fp4x2_e2m1_t, asc::te::dn_ext_layout_ptn>("mxfp4_e2m1", "mxfp4_e2m1", "dn", 0, true, 2);
}

TEST_F(QgmmMxKernelTest, MxFp8NdSparseList)
{
    RunQgmmCase<fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn>("mxfp8_e4m3", "mxfp8_e4m3", "nd", 2);
}

TEST_F(QgmmMxKernelTest, MxFp8NzSingleTensor)
{
    RunQgmmCase<fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::nz_layout_ptn>("mxfp8_e4m3", "mxfp8_e4m3", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp4NzSingleTensor)
{
    RunQgmmCase<fp4x2_e2m1_t, fp4x2_e2m1_t, asc::te::nz_layout_ptn>("mxfp4_e2m1", "mxfp4_e2m1", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp4MixedEncodingNzSingleTensor)
{
    RunQgmmCase<fp4x2_e2m1_t, fp4x2_e1m2_t, asc::te::nz_layout_ptn>("mxfp4_e2m1", "mxfp4_e1m2", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp8ZnSingleTensor)
{
    RunQgmmCase<fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::zn_layout_ptn>("mxfp8_e4m3", "mxfp8_e4m3", "zn");
}

TEST_F(QgmmMxKernelTest, MxFp8NzMultiTensor)
{
    RunQgmmCase<fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::nz_layout_ptn, true>("mxfp8_e4m3", "mxfp8_e4m3", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp4NzMultiTensor)
{
    RunQgmmCase<fp4x2_e2m1_t, fp4x2_e2m1_t, asc::te::nz_layout_ptn, true>("mxfp4_e2m1", "mxfp4_e2m1", "nz");
}

TEST_F(QgmmMxKernelTest, MxFp4E1m2NzMultiTensor)
{
    RunQgmmCase<fp4x2_e1m2_t, fp4x2_e1m2_t, asc::te::nz_layout_ptn, true>("mxfp4_e1m2", "mxfp4_e1m2", "nz");
}

TEST_F(QgmmMxKernelTest, BasicShapeE1M16N32K64)
{
    RunQgmmShapeCase<fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(1, 16, 32, 64);
}

TEST_F(QgmmMxKernelTest, BasicShapeE3M24N48K96Tail)
{
    RunQgmmShapeCase<fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(3, 24, 48, 96);
}

TEST_F(QgmmMxKernelTest, MxFp8NdL1TripleBuffer)
{
    RunQgmmShapeCase<fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(2, 64, 128, 192, 3);
}

TEST_F(QgmmMxKernelTest, TransB)
{
    RunQgmmShapeCase<fp8_e4m3fn_t, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>(2, 16, 64, 64);
}

TEST_F(QgmmMxKernelTest, MxFp4KGroupSplitK)
{
    // Unequal and non-64-aligned k_g, including the 64 boundary.
    RunQgmmKGroupCase<fp4x2_e2m1_t>(3, 64, 128, {63, 64, 65});
}

} // namespace
