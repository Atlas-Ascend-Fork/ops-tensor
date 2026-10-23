/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_matmul_activation_quant.cpp
 * \brief QuantMatmulActivationQuant assembly compile checks and CPU-debug epilogue setup tests.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "quant_matmul_activation_quant.h"
#include "tikicpulib.h"

namespace {

using ND = asc::te::nd_ext_layout_ptn;
using DN = asc::te::dn_ext_layout_ptn;
using NZ = asc::te::nz_layout_ptn;
using ZN = asc::te::zn_layout_ptn;

constexpr int64_t M = 16;
constexpr int64_t N = 64;
constexpr uint32_t BLOCK_NUM = 1U;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

// Assembly checks validate types and signatures without instantiating or executing operator() bodies.
template <typename AType, typename BType, typename OutputType, typename LayoutA, typename LayoutB,
          uint64_t FullLoadMode>
void CheckPublicAssembly()
{
    using Types = QuantMatmulActivationQuantTypes<AType, BType, OutputType, LayoutA, LayoutB, FullLoadMode>;
    using Kernel = typename Types::Kernel;
    using Epilogue = typename Types::BlockEpilogue;

    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelMmadWithScaleMxActivationQuant>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(Types::DispatchPolicy::L0C2UB_MODE == Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M);
    static_assert(std::is_same_v<typename Types::BlockMmad::AType, AType>);
    static_assert(std::is_same_v<typename Types::BlockMmad::BType, BType>);
    static_assert(std::is_same_v<typename Types::BlockMmad::CType, float>);
    static_assert(std::is_same_v<typename Types::BlockMmad::BiasType, float>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutC, ND>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeIn, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeOut, OutputType>);
    static_assert(
        std::is_same_v<decltype(std::declval<Epilogue&>().Init(std::declval<const typename Epilogue::Params&>())),
                       void>);
    static_assert(std::is_same_v<decltype(std::declval<Epilogue&>().UpdateNextProblem(
                                     std::declval<const typename Epilogue::ProblemShape&>())),
                                 void>);
    static_assert(std::is_same_v<decltype(std::declval<Epilogue&>().UpdateGlobalAddr(
                                     std::declval<const typename Epilogue::BlockCoord&>())),
                                 void>);
    static_assert(
        std::is_same_v<decltype(std::declval<Epilogue&>()(std::declval<const typename Epilogue::BlockShape&>(),
                                                          std::declval<const typename Epilogue::BlockCoord&>())),
                       void>);
    static_assert(
        std::is_same_v<decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>);

    typename Kernel::Params params{};
    params.problemShape = {128L, 256L, 512L, 2L};
    params.mmadParams = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    params.epilogueParams = {nullptr,
                             nullptr,
                             128U,
                             256U,
                             Blaze::Epilogue::Block::GeluAlg::ERF,
                             Blaze::Epilogue::Block::QuantAlg::DYN_DTYPE_RANGE,
                             Blaze::Epilogue::Block::ROUND_MODE_FP4::ROUND,
                             7.0F};
    params.l1Params = {512U, 512U, 2U};
    params.schParams = {128L, 256L, 1L, 1L, 1L, 1L, 0L, 0L};
    params.qbmmParams = {1U, 1U, 1U, 2U, 1U, 1U, 1U, 2U, 1U, 1U, 1U, 2U, 1U, 128U, 256U, 64U, 1U, 2U};

    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.epilogueParams.baseM, 128U);
    EXPECT_EQ(params.epilogueParams.baseN, 256U);
    EXPECT_EQ(params.epilogueParams.geluAlg, Blaze::Epilogue::Block::GeluAlg::ERF);
    EXPECT_EQ(params.epilogueParams.quantAlg, Blaze::Epilogue::Block::QuantAlg::DYN_DTYPE_RANGE);
    EXPECT_EQ(params.epilogueParams.fp4RoundMode, Blaze::Epilogue::Block::ROUND_MODE_FP4::ROUND);
    EXPECT_FLOAT_EQ(params.epilogueParams.dtypeMax, 7.0F);
    EXPECT_EQ(params.l1Params.kL1, 512U);
    EXPECT_EQ(params.l1Params.scaleKL1, 512U);
    EXPECT_EQ(params.l1Params.l1BufNum, 2U);
    EXPECT_EQ(params.schParams.mBaseTailSplitCnt, 1L);
    EXPECT_EQ(params.qbmmParams.batchC4, 2U);
    EXPECT_EQ(params.qbmmParams.biasThreeDim, 1U);
    EXPECT_EQ(params.qbmmParams.dbL0C, 2U);
}

template <typename LayoutA, typename LayoutB>
void CheckLayoutAndLoadModes()
{
    CheckPublicAssembly<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t, LayoutA, LayoutB, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPublicAssembly<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t, LayoutA, LayoutB, Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <typename LayoutB, uint64_t FullLoadMode>
void CheckSwiGluWeightNzAssembly()
{
    using Types = QuantMatmulSwiGluQuantTypes<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e5m2_t, LayoutB, FullLoadMode>;
    using BlockMmad = typename Types::BlockMmad;
    using Kernel = typename Types::Kernel;

    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelMmadWithScaleMxActivationQuant>);
    static_assert(BlockMmad::WEIGHT_NZ);
    static_assert(!BlockMmad::TRANS_A);
    static_assert(std::is_same_v<typename BlockMmad::LayoutB, LayoutB>);
    static_assert(BlockMmad::CONCAT_N);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::DataTypeOut, fp8_e5m2_t>);
    static_assert(
        std::is_same_v<decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>);

    EXPECT_GT(sizeof(Kernel), 0U);
}

template <typename LayoutA, typename LayoutB, uint64_t FullLoadMode>
void CheckWithoutBatchGeluAssembly()
{
    using Types = QuantMatmulActivationQuantWithoutBatchTypes<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e5m2_t, LayoutA, LayoutB,
                                                              FullLoadMode>;
    using Kernel = typename Types::Kernel;

    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelMmadWithScaleMxActivationQuant>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(
        std::is_same_v<decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>);

    typename Kernel::Params params{};
    params.problemShape = {64L, 128L, 128L, 1L};
    params.mmadParams = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    params.epilogueParams = {nullptr,
                             nullptr,
                             64U,
                             128U,
                             Blaze::Epilogue::Block::GeluAlg::TANH,
                             Blaze::Epilogue::Block::QuantAlg::OCP,
                             Blaze::Epilogue::Block::ROUND_MODE_FP4::RINT,
                             0.0F};
    params.l1Params = {64U, 64U, 2U};
    params.schParams = {64L, 128L, 1L, 1L, 1L, 1L, 0L, 0L};
    // Single-batch cases fill the batch parameters with defaults (all dims 1).
    params.qbmmParams = {1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 0U, 64U, 128U, 64U, 0U, 1U};

    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(asc::te::get<Blaze::Gemm::MNK_B>(params.problemShape), 1L);
    EXPECT_EQ(params.qbmmParams.baseN, 128U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 1U);
}

template <typename LayoutB, uint64_t FullLoadMode>
void CheckWithoutBatchSwiGluAssembly()
{
    using Types = QuantMatmulSwiGluQuantWithoutBatchTypes<fp8_e5m2_t, fp8_e4m3fn_t, fp8_e5m2_t, LayoutB, FullLoadMode>;
    using Kernel = typename Types::Kernel;

    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelMmadWithScaleMxActivationQuant>);
    static_assert(Types::BlockMmad::CONCAT_N);
    static_assert(!Types::BlockMmad::TRANS_A);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::DataTypeOut, fp8_e5m2_t>);
    static_assert(
        std::is_same_v<decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>);

    typename Kernel::Params params{};
    params.problemShape = {65L, 128L, 128L, 1L};
    params.mmadParams = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    params.epilogueParams.yGmAddr = nullptr;
    params.epilogueParams.yScaleGmAddr = nullptr;
    params.epilogueParams.baseM = 64U;
    params.epilogueParams.baseN = 64U;
    params.epilogueParams.scaleAlg = static_cast<uint8_t>(1U);
    params.l1Params = {64U, 64U, 2U};
    params.schParams = {64L, 128L, 1L, 1L, 1L, 1L, 0L, 0L};
    // Single-batch cases fill the batch parameters with defaults (all dims 1).
    params.qbmmParams = {1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 0U, 64U, 128U, 64U, 1U, 2U};

    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.epilogueParams.baseN, params.qbmmParams.baseN / 2U);
    EXPECT_EQ(params.epilogueParams.scaleAlg, static_cast<uint8_t>(1U));
    EXPECT_EQ(params.qbmmParams.isBias, 1U);
}

template <typename AType, typename BType, typename LayoutA, typename LayoutB, uint64_t FullLoadMode>
void CheckSwishAssembly()
{
    using Types = QuantMatmulSwishQuantTypes<AType, BType, LayoutA, LayoutB, FullLoadMode>;
    using ExplicitScalar = Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<
        AType, float, Blaze::Epilogue::Block::SwishBetaMode::Scalar>;
    static_assert(std::is_same_v<typename Types::BlockEpilogue, ExplicitScalar>);
    const typename ExplicitScalar::Params defaultParams{};
    EXPECT_FLOAT_EQ(defaultParams.beta, 1.0F);
    using Kernel = typename Types::Kernel;
    using Epilogue = typename Types::BlockEpilogue;
    using BlockMmad = typename Types::BlockMmad;
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelMmadWithScaleMxActivationQuant>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(Types::DispatchPolicy::L0C2UB_MODE == Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M);
    static_assert(!BlockMmad::CONCAT_N);
    static_assert(std::is_same_v<typename BlockMmad::AType, AType>);
    static_assert(std::is_same_v<typename BlockMmad::BType, BType>);
    static_assert(std::is_same_v<typename BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename BlockMmad::CType, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeIn, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeOut, AType>);
    static_assert(
        std::is_same_v<decltype(std::declval<Epilogue&>()(std::declval<const typename Epilogue::BlockShape&>(),
                                                          std::declval<const typename Epilogue::BlockCoord&>())),
                       void>);
    static_assert(
        std::is_same_v<decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>);
    EXPECT_GT(sizeof(Kernel), 0U);
}

template <typename AType, typename BType, typename LayoutA, typename LayoutB>
void CheckSwishLoadModes()
{
    CheckSwishAssembly<AType, BType, LayoutA, LayoutB, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckSwishAssembly<AType, BType, LayoutA, LayoutB, Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <typename AType, typename BType>
void CheckSwishNdLayouts()
{
    CheckSwishLoadModes<AType, BType, ND, ND>();
    CheckSwishLoadModes<AType, BType, ND, DN>();
    CheckSwishLoadModes<AType, BType, DN, ND>();
    CheckSwishLoadModes<AType, BType, DN, DN>();
}

template <typename OutputType, uint32_t ScaleAlg>
void RunSwishEpilogueSetup(float beta = 1.0F)
{
    GmBuffer y(static_cast<size_t>(M * N));
    GmBuffer yScale(static_cast<size_t>(M * 2));
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(yScale.Get(), nullptr);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = quant_matmul_swish_quant_epilogue_setup_entry<OutputType, ScaleAlg>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, BLOCK_NUM, y.Get(), yScale.Get(), beta)) << "Swish epilogue setup failed";
}

template <typename OutputType, Blaze::Epilogue::Block::GeluAlg GeluAlg, Blaze::Epilogue::Block::QuantAlg QuantAlg,
          Blaze::Epilogue::Block::ROUND_MODE_FP4 RoundMode>
void RunEpilogueSetup(float dstTypeMax)
{
    GmBuffer y(static_cast<size_t>(M * N));
    GmBuffer yScale(static_cast<size_t>(M * 2));
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(yScale.Get(), nullptr);
    std::fill_n(reinterpret_cast<uint8_t*>(y.Get()), static_cast<size_t>(M * N), 0U);
    std::fill_n(reinterpret_cast<uint8_t*>(yScale.Get()), static_cast<size_t>(M * 2), 0U);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = quant_matmul_activation_quant_epilogue_setup_entry<OutputType, GeluAlg, QuantAlg, RoundMode>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, BLOCK_NUM, y.Get(), yScale.Get(), dstTypeMax)) << "GELU epilogue setup failed";
}

} // namespace

TEST(QuantMatmulActivationQuantTest, PublicAssemblyCoversSupportedDtypeMatrix)
{
    CheckPublicAssembly<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPublicAssembly<fp8_e5m2_t, fp8_e4m3fn_t, fp8_e5m2_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPublicAssembly<fp8_e5m2_t, fp8_e5m2_t, fp8_e5m2_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPublicAssembly<fp8_e4m3fn_t, fp8_e5m2_t, fp8_e4m3fn_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPublicAssembly<fp4x2_e2m1_t, fp4x2_e2m1_t, fp4x2_e2m1_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

TEST(QuantMatmulActivationQuantTest, PublicAssemblyCoversAllTilingKeyLayoutsAndLoadModes)
{
    CheckLayoutAndLoadModes<ND, ND>();
    CheckLayoutAndLoadModes<ND, DN>();
    CheckLayoutAndLoadModes<ND, NZ>();
    CheckLayoutAndLoadModes<ND, ZN>();
    CheckLayoutAndLoadModes<DN, ND>();
    CheckLayoutAndLoadModes<DN, DN>();
    CheckLayoutAndLoadModes<DN, NZ>();
    CheckLayoutAndLoadModes<DN, ZN>();
}

TEST(QuantMatmulActivationQuantTest, SwiGluAssemblySupportsWeightNzAndBTranspose)
{
    CheckSwiGluWeightNzAssembly<NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckSwiGluWeightNzAssembly<NZ, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckSwiGluWeightNzAssembly<ZN, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckSwiGluWeightNzAssembly<ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
}

TEST(QuantMatmulActivationQuantTest, WithoutBatchAssemblyCoversGeluAndSwiGlu)
{
    CheckWithoutBatchGeluAssembly<ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckWithoutBatchGeluAssembly<DN, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckWithoutBatchSwiGluAssembly<ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckWithoutBatchSwiGluAssembly<DN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckWithoutBatchSwiGluAssembly<NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckWithoutBatchSwiGluAssembly<ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
}

TEST(QuantMatmulActivationQuantTest, EpilogueSetupCoversAlgorithmsAndRoundModes)
{
    using Blaze::Epilogue::Block::GeluAlg;
    using Blaze::Epilogue::Block::QuantAlg;
    using Blaze::Epilogue::Block::ROUND_MODE_FP4;
    RunEpilogueSetup<fp8_e4m3fn_t, GeluAlg::TANH, QuantAlg::OCP, ROUND_MODE_FP4::RINT>(0.0F);
    RunEpilogueSetup<fp8_e5m2_t, GeluAlg::ERF, QuantAlg::BLAS, ROUND_MODE_FP4::RINT>(0.0F);
    RunEpilogueSetup<fp4x2_e2m1_t, GeluAlg::ERF, QuantAlg::DYN_DTYPE_RANGE, ROUND_MODE_FP4::FLOOR>(7.0F);
    RunEpilogueSetup<fp4x2_e2m1_t, GeluAlg::TANH, QuantAlg::OCP, ROUND_MODE_FP4::ROUND>(0.0F);
}

TEST(QuantMatmulActivationQuantTest, SwishAssemblyCoversNdDtypesTransposesAndLoadModes)
{
    CheckSwishNdLayouts<fp8_e4m3fn_t, fp8_e4m3fn_t>();
    CheckSwishNdLayouts<fp8_e4m3fn_t, fp8_e5m2_t>();
    CheckSwishNdLayouts<fp8_e5m2_t, fp8_e4m3fn_t>();
    CheckSwishNdLayouts<fp8_e5m2_t, fp8_e5m2_t>();
}

TEST(QuantMatmulActivationQuantTest, SwishAssemblyCoversWeightNzTransposesAndLoadModes)
{
    CheckSwishLoadModes<fp8_e4m3fn_t, fp8_e4m3fn_t, ND, NZ>();
    CheckSwishLoadModes<fp8_e4m3fn_t, fp8_e4m3fn_t, ND, ZN>();
    CheckSwishLoadModes<fp8_e5m2_t, fp8_e4m3fn_t, ND, NZ>();
    CheckSwishLoadModes<fp8_e5m2_t, fp8_e4m3fn_t, ND, ZN>();
}

TEST(QuantMatmulActivationQuantTest, SwishEpilogueSetupCoversOutputDtypesAndScaleAlgorithms)
{
    RunSwishEpilogueSetup<fp8_e4m3fn_t, 0U>();
    RunSwishEpilogueSetup<fp8_e4m3fn_t, 1U>(0.5F);
    RunSwishEpilogueSetup<fp8_e5m2_t, 0U>(2.0F);
    RunSwishEpilogueSetup<fp8_e5m2_t, 1U>(-1.0F);
}
