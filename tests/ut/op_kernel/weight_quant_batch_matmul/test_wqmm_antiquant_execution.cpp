/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full license.
 */
#include "gtest/gtest.h"
#include "wqmm_test_utils.h"
#include "weight_quant_batch_matmul.h"

namespace {
using namespace WeightQuantBatchMatmulUT;
using Blaze::Gemm::QuantMode;
struct ExecutionCase {
    int64_t m, n, k, ka, kb;
    bool bias, preload;
};
template <class Component>
__global__ __aicore__ void AntiquantExecutionKernel(GM_ADDR config, GM_ADDR traces)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using Kernel = typename Component::Kernel;
    auto params = *reinterpret_cast<const typename Kernel::Params*>(config);
    uint64_t lane = 0;
    if ASCEND_IS_AIV {
        lane = 1 + AscendC::GetSubBlockIdx();
    }
    VectorTrace trace;
    vectorTrace = &trace;
    if ASCEND_IS_AIV {
        antiquantSyncTrace = &trace.sync;
    }
    Kernel{}(params);
    antiquantSyncTrace = nullptr;
    vectorTrace = nullptr;
    reinterpret_cast<VectorTrace*>(traces)[lane] = trace;
}

template <class Component>
void Execute(const ExecutionCase& c)
{
    using X = typename Component::BlockMmad::AType;
    using Bias = typename Component::BlockMmad::BiasType;
    using Kernel = typename Component::Kernel;
    using Policy = typename Component::DispatchPolicy;
    // Antiquant B describes physical (N,K): ND stores each K row contiguously.
    constexpr bool TRANS = !Blaze::Gemm::IsTrans<typename Component::LayoutB>::value;
    constexpr bool OFFSET = Policy::HAS_ANTIQUANT_OFFSET;
    constexpr bool CHANNEL = Policy::ANTIQUANT_TYPE == QuantMode::PERCHANNEL_MODE;
    GmBuffer a(c.m * c.k * sizeof(X)), b(c.n * c.k), scale(c.n * sizeof(X)), offset(c.n * sizeof(X)),
        bias(c.n * sizeof(Bias)), output(c.m * c.n * sizeof(X) + 64), config(sizeof(typename Kernel::Params)),
        traces(3 * sizeof(VectorTrace));
    std::vector<X> av(c.m * c.k, X(1)), sv(c.n, X(0.5f)), ov(c.n, X(1));
    a.Set(av);
    scale.Set(sv);
    offset.Set(ov);
    std::memset(b.Get(), 1, c.n * c.k);
    std::memset(output.Get(), 0xA5, c.m * c.n * sizeof(X) + 64);
    typename Kernel::Params params{};
    params.problemShape = asc::te::make_shape(c.m, c.n, c.k);
    params.mmadParams.aGmAddr = a.Get();
    params.mmadParams.cGmAddr = output.Get();
    params.mmadParams.biasGmAddr = bias.Get();
    params.mmadParams.hasBias = c.bias;
    params.mmadParams.kSize = c.k;
    params.mmadParams.l1TileShape = asc::te::make_shape(int64_t(16), int64_t(32), c.ka, c.kb);
    params.mmadParams.l0TileShape = asc::te::make_shape(int64_t(16), int64_t(32), int64_t(64));
    params.prologueParams.bGmAddr = b.Get();
    params.prologueParams.scaleGmAddr = scale.Get();
    params.prologueParams.offsetGmAddr = offset.Get();
    params.prologueParams.weightL2Cacheable = !c.preload;
    params.aElementCount = c.m * c.k;
    params.aPreloadSize = c.preload ? c.k : 0;
    // Full main + first tail + second tail; the smaller cases use only main.
    if (c.n == 88)
        params.schedulerParams = {16, 2, 1, 1, 32, 16, 8, 1, 1};
    else
        params.schedulerParams = {16, 1, 0, 0, uint64_t(c.n), 16, 8, 1, 1};
    config.Set(std::vector<typename Kernel::Params>{params});
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = AntiquantExecutionKernel<Component>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1, config.Get(), traces.Get()));
    auto* actual = reinterpret_cast<VectorTrace*>(traces.Get());
    EXPECT_EQ(actual[0].loads, 0U);
    EXPECT_EQ(actual[0].stores, 0U);
    std::vector<uint64_t> widths = c.n == 88 ? std::vector<uint64_t>{32, 32, 16, 8} :
                                               std::vector<uint64_t>{uint64_t(c.n)};
    for (uint64_t lane = 0; lane < 2; ++lane) {
        uint64_t loads = 0, stores = 0;
        for (auto width : widths) {
            for (int64_t start = 0; start < c.k; start += c.kb) {
                uint64_t lenK = std::min(c.kb, c.k - start);
                if (TRANS) {
                    uint64_t rows = lane == 0 ? width / 2 : width - width / 2;
                    uint64_t vfBlocks = (lenK + 255) / 256;
                    stores += vfBlocks * rows * 2;
                    loads += vfBlocks * rows * (2 + (CHANNEL ? 1 + OFFSET : 0));
                } else {
                    uint64_t halfK = c.kb / 2;
                    uint64_t localK = lane == 0 ? std::min(lenK, halfK) : (lenK > halfK ? lenK - halfK : 0);
                    for (uint64_t startK = 0; startK < localK; startK += 64) {
                        auto rows = std::min<uint64_t>(64, localK - startK);
                        uint64_t vfBlocks = (width + 255) / 256;
                        stores += vfBlocks * rows * 2;
                        loads += vfBlocks * (rows * 2 + (CHANNEL ? 2 * (1 + OFFSET) : 0));
                    }
                }
            }
        }
        uint64_t mTiles = (c.m + 15) / 16;
        uint64_t handoffs = mTiles * widths.size() * ((c.k + c.kb - 1) / c.kb);
        EXPECT_EQ(actual[lane + 1].sync.sets, handoffs);
        EXPECT_EQ(actual[lane + 1].sync.waits, handoffs);
        EXPECT_EQ(actual[lane + 1].sync.normalizedSets, lane == 1 ? handoffs : 0);
        EXPECT_EQ(actual[lane + 1].sync.normalizedWaits, lane == 1 ? handoffs : 0);
        EXPECT_EQ(actual[lane + 1].sync.unexpectedFlags, 0U);
        EXPECT_EQ(actual[lane + 1].stores, stores * mTiles);
        EXPECT_EQ(actual[lane + 1].loads, loads * mTiles);
    }
    // The PV memory shim does not implement MMAD numerics. Guard checking and
    // exact VF call counts remain observable and do not claim numerical coverage.
    for (uint64_t i = c.m * c.n * sizeof(X); i < c.m * c.n * sizeof(X) + 64; ++i)
        ASSERT_EQ(output.Get()[i], 0xA5);
}
using F16Channel = Components<half, int8_t, half, half, 512, 2, true, QuantMode::PERCHANNEL_MODE, true, false>;
using F16Tensor = Components<half, int8_t, half, half, 512, 4, false, QuantMode::PERTENSOR_MODE, true, true>;
using B16Channel = Components<bfloat16_t, int8_t, bfloat16_t, bfloat16_t, 1024, 2, false, QuantMode::PERCHANNEL_MODE,
                              false, false>;
using F16Fp8 = Components<half, float8_e4m3_t, half, half, 512, 4, true, QuantMode::PERTENSOR_MODE, false, false>;
using B16Fp8 = Components<bfloat16_t, float8_e4m3_t, bfloat16_t, bfloat16_t, 512, 2, true, QuantMode::PERCHANNEL_MODE,
                          true, true>;
using F16Hi8 = Components<half, hifloat8_t, half, half, 1024, 2, false, QuantMode::PERCHANNEL_MODE, true, false>;
using B16Hi8 = Components<bfloat16_t, hifloat8_t, bfloat16_t, bfloat16_t, 256, 4, true, QuantMode::PERTENSOR_MODE,
                          false, false>;
TEST(WqmmAntiquantExecutionTest, ChannelOffsetBiasAndThreeNRanges)
{
    Execute<F16Channel>({33, 88, 320, 128, 128, true, true});
}
TEST(WqmmAntiquantExecutionTest, TensorOffsetFullAInterleaved)
{
    Execute<F16Tensor>({16, 32, 512, 512, 128, true, false});
}
TEST(WqmmAntiquantExecutionTest, Bf16ChannelKTailAndEmptySecondAiv)
{
    Execute<B16Channel>({17, 32, 129, 256, 128, false, true});
}
TEST(WqmmAntiquantExecutionTest, Fp8PerTensorSingleKFullA) { Execute<F16Fp8>({16, 32, 64, 128, 128, false, false}); }
TEST(WqmmAntiquantExecutionTest, Bf16Fp8OffsetAndBias) { Execute<B16Fp8>({16, 32, 128, 128, 128, true, false}); }
TEST(WqmmAntiquantExecutionTest, HiFloat8ChannelKTail) { Execute<F16Hi8>({16, 32, 160, 256, 128, false, false}); }
TEST(WqmmAntiquantExecutionTest, Bf16HiFloat8TensorMultiUb) { Execute<B16Hi8>({16, 32, 640, 128, 128, false, false}); }
// Execute every production component emitted by the compilation matrix as
// well as the orientation/quantization variants above. All three shapes are
// needed for each type: blocked A, full interleaved A and contiguous full A
// with a K tail instantiate distinct helper functions in BlockMmad.
using ProductionComponents = testing::Types<Fp16Int8, Fp16Int8Offset, Bf16Int8, Fp16Fp8, Bf16Fp8, Fp16HiFloat8,
                                            Bf16HiFloat8, F16Channel, F16Tensor, B16Channel, F16Fp8, B16Fp8, F16Hi8,
                                            B16Hi8>;
template <class Component>
class WqmmFunctionCoverageTest : public testing::Test {};
TYPED_TEST_SUITE(WqmmFunctionCoverageTest, ProductionComponents);
TYPED_TEST(WqmmFunctionCoverageTest, BlockedAWithBiasPreloadAndAllNRanges)
{
    Execute<TypeParam>({17, 88, 640, 128, 128, true, true});
}
TYPED_TEST(WqmmFunctionCoverageTest, FullAInterleavedWithBias)
{
    Execute<TypeParam>({16, 32, 512, 512, 128, true, false});
}
TYPED_TEST(WqmmFunctionCoverageTest, FullAContiguousWithKTail)
{
    Execute<TypeParam>({17, 32, 257, 512, 128, false, true});
}
} // namespace
