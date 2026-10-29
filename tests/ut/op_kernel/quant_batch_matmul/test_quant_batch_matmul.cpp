/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_quant_batch_matmul.cpp
 * \brief QBMMV3 Kernel UT测试用例
 */

#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "qbmm_cube.h"
#include "qbmm_mix.h"
#include "qbmm_mx.h"
#include "blaze/gemm/utils/common_utils.h"

class QBMMV3Test : public testing::Test {
protected:
    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/quant_batch_matmul/qbmm_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }
};

namespace {

// QuantMode 编码（与 BlockEpilogueDequant::QuantMode 一致）
constexpr uint32_t QM_DEFAULT = 0;
constexpr uint32_t QM_PERTENSOR = 1;
constexpr uint32_t QM_PERCHANNEL = 2;
constexpr uint32_t QM_PERTOKEN = 4;
constexpr size_t STREAMK_WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(float);
constexpr size_t PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(int32_t);
constexpr size_t STREAMK_WORKSPACE_OVERHEAD = 20UL * 1024UL * 1024UL;

// ge::DataType 编码（bias 运行时 dtype）
constexpr uint32_t GE_DT_FLOAT = 0;
constexpr uint32_t GE_DT_FLOAT16 = 1;
constexpr uint32_t GE_DT_INT32 = 3;
constexpr uint32_t GE_DT_BF16 = 27;

struct MixCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t x1QuantMode; // 激活量化模式
    uint32_t x2QuantMode; // 权重量化模式
    bool isBias;
    uint32_t biasDtype;  // ge::DataType 编码，需与 bias.bin 元素类型一致
    size_t biasElemSize; // bias 单元素字节数（与 biasDtype 匹配）
    size_t outElemSize;  // 输出单元素字节数（half/bf16=2, float=4）
    const char* genArgs; // 传给 gen_data.py 的量化模式参数
    int64_t batch = 1;
    bool biasThreeDim = false;
    uint32_t kAL1 = 0;  // 0 keeps the legacy full-K L1 tile.
    uint32_t kBL1 = 0;  // 0 mirrors kAL1.
    uint32_t baseM = 0; // 0 keeps the legacy single M tile.
    uint32_t baseN = 0; // 0 keeps the legacy single N tile.
    uint32_t mTailTile = 1;
    uint32_t nTailTile = 1;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    int64_t mTailMain = 0;
    int64_t nTailMain = 0;
};

struct CubeCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    bool isBias;
    uint32_t biasDtype;
    size_t biasElemSize;
    size_t outElemSize;
    const char* genArgs;
    uint32_t kL1 = 0; // 0 keeps the legacy full-K L1 tile.
    uint32_t nBufferNum = 2;
    uint32_t baseK = 0; // 0 keeps the legacy full-K L0 tile.
    uint32_t baseM = 0; // 0 keeps the legacy single M/N tile.
    uint32_t baseN = 0;
    int64_t batch = 1;
    bool biasThreeDim = false;
    size_t scaleElemSize = sizeof(uint64_t);
    uint32_t dbL0C = 1;
    uint32_t kBL1 = 0; // 0 mirrors kL1; nonzero values exercise asymmetric A/B L1 K splits.
};

struct MxCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t nBufferNum;
    bool isBias;
    int64_t batch = 1;
    bool biasThreeDim = false;
};

struct L0CPingpongCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t nBufferNum;
    uint32_t blockNum;
    int64_t batch = 1;
};

struct StreamKCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
};

struct PertensorStreamKCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t baseM = 16U;
    uint32_t baseN = 16U;
    uint32_t baseK = 64U;
    uint32_t singleCoreK = 64U;
    uint32_t kL1 = 64U;
    bool hasX1Scale = false;
    bool isBias = false;
    uint32_t biasDtype = GE_DT_FLOAT;
    size_t biasElemSize = sizeof(float);
};

size_t GetMxScaleKLen(int64_t k) { return static_cast<size_t>((k + 63) / 64) * 2UL; }

template <typename T>
size_t GetMxInputSize(int64_t elementCount)
{
    if constexpr (Blaze::Gemm::IsFp4<T>()) {
        return ((static_cast<size_t>(elementCount) + 1UL) / 2UL) * sizeof(T);
    } else {
        return static_cast<size_t>(elementCount) * sizeof(T);
    }
}

void FillGmBuffer(GM_ADDR addr, size_t size, uint8_t value)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    std::fill_n(buffer, size, value);
}

template <typename T>
void FillGmValues(GM_ADDR addr, size_t count, T value)
{
    std::fill_n(reinterpret_cast<T*>(addr), count, value);
}

void ReplicateGmBatch(GM_ADDR addr, size_t singleBatchSize, int64_t batch)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    for (int64_t batchIdx = 1; batchIdx < batch; ++batchIdx) {
        std::memcpy(buffer + static_cast<size_t>(batchIdx) * singleBatchSize, buffer, singleBatchSize);
    }
}

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}

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

template <typename ScaleType>
void FillMixScaleOnes(GM_ADDR scaleGM, size_t count)
{
    if constexpr (std::is_same_v<ScaleType, float>) {
        FillGmValues<float>(scaleGM, count, 1.0F);
    } else if constexpr (std::is_same_v<ScaleType, bfloat16_t>) {
        constexpr uint16_t BF16_ONE_BITS = 0x3F80U;
        FillGmValues<uint16_t>(scaleGM, count, BF16_ONE_BITS);
    } else if constexpr (std::is_same_v<ScaleType, half>) {
        constexpr uint16_t FP16_ONE_BITS = 0x3C00U;
        FillGmValues<uint16_t>(scaleGM, count, FP16_ONE_BITS);
    }
}

template <typename ScaleType>
void FillPertensorStreamKScaleOne(GM_ADDR scaleGM)
{
    if constexpr (std::is_same_v<ScaleType, float>) {
        FillGmValues<float>(scaleGM, 1U, 1.0F);
    } else if constexpr (std::is_same_v<ScaleType, bfloat16_t>) {
        constexpr uint16_t BF16_ONE_BITS = 0x3F80U;
        FillGmValues<uint16_t>(scaleGM, 1U, BF16_ONE_BITS);
    } else if constexpr (std::is_same_v<ScaleType, uint64_t> || std::is_same_v<ScaleType, int64_t>) {
        constexpr uint64_t FIXPIPE_SCALE_ONE = 0x000000003F800000UL;
        FillGmValues<ScaleType>(scaleGM, 1U, static_cast<ScaleType>(FIXPIPE_SCALE_ONE));
    }
}

// 未指定切分时保留单 tile 默认值；显式配置用于多 tile 和 L1 buffer 回绕测试。
void FillCubeTiling(QBMMV3TilingData* tilingData, const CubeCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.baseM == 0 ? caseCfg.M : caseCfg.baseM;
    tilingData->baseN = caseCfg.baseN == 0 ? caseCfg.N : caseCfg.baseN;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = caseCfg.x1QuantMode;
    tilingData->x2QuantMode = caseCfg.x2QuantMode;
    tilingData->kAL1 = caseCfg.kL1 == 0 ? static_cast<uint32_t>(caseCfg.K) : caseCfg.kL1;
    tilingData->kBL1 = caseCfg.kBL1 == 0 ? tilingData->kAL1 : caseCfg.kBL1;
    tilingData->nBufferNum = caseCfg.nBufferNum;
    tilingData->baseM_qbmm = static_cast<uint32_t>(tilingData->baseM);
    tilingData->baseN_qbmm = static_cast<uint32_t>(tilingData->baseN);
    tilingData->baseK_qbmm = caseCfg.baseK == 0 ? static_cast<uint32_t>(caseCfg.K) : caseCfg.baseK;
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = caseCfg.dbL0C;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMixTiling(QBMMV3TilingData* tilingData, const MixCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.baseM == 0 ? caseCfg.M : caseCfg.baseM;
    tilingData->baseN = caseCfg.baseN == 0 ? caseCfg.N : caseCfg.baseN;
    tilingData->mTailTile = caseCfg.mTailTile;
    tilingData->nTailTile = caseCfg.nTailTile;
    tilingData->mBaseTailSplitCnt = caseCfg.mBaseTailSplitCnt;
    tilingData->nBaseTailSplitCnt = caseCfg.nBaseTailSplitCnt;
    tilingData->mTailMain = caseCfg.mTailMain;
    tilingData->nTailMain = caseCfg.nTailMain;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = caseCfg.x1QuantMode;
    tilingData->x2QuantMode = caseCfg.x2QuantMode;
    tilingData->kAL1 = caseCfg.kAL1 == 0 ? static_cast<uint32_t>(caseCfg.K) : caseCfg.kAL1;
    tilingData->kBL1 = caseCfg.kBL1 == 0 ? tilingData->kAL1 : caseCfg.kBL1;
    tilingData->nBufferNum = 2;
    tilingData->baseM_qbmm = static_cast<uint32_t>(tilingData->baseM);
    tilingData->baseN_qbmm = static_cast<uint32_t>(tilingData->baseN);
    tilingData->baseK_qbmm = static_cast<uint32_t>(caseCfg.K);
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMxTiling(QBMMV3TilingData* tilingData, const MxCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.baseM;
    tilingData->baseN = caseCfg.baseN;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = 0;
    tilingData->x2QuantMode = 0;
    tilingData->kAL1 = caseCfg.kL1;
    tilingData->kBL1 = caseCfg.scaleKL1;
    tilingData->nBufferNum = caseCfg.nBufferNum;
    tilingData->baseM_qbmm = caseCfg.baseM;
    tilingData->baseN_qbmm = caseCfg.baseN;
    tilingData->baseK_qbmm = caseCfg.baseK;
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = GE_DT_FLOAT;
}

// 清理旧 .bin 并调用 gen_data.py 生成输入数据；genArgs 为量化模式等附加参数（fixpipe 用例传 ""）。
// 供 RunMixSmoke 与 fixpipe 用例复用，避免数据生成命令块重复。
void RunGenData(const std::string& dataDir, int64_t M, int64_t N, int64_t K, const std::string& genArgs)
{
    std::string cleanCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " + std::to_string(M) +
                             " --n " + std::to_string(N) + " --k " + std::to_string(K) +
                             (genArgs.empty() ? std::string("") : (std::string(" ") + genArgs));
    int genRet = system(cleanCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in qbmm_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;
}

// 读取 .bin 到 GM 缓冲区，带打开/读取断言。供 RunMixSmoke 与 fixpipe 用例复用。
void ReadBinToGm(const std::string& path, GM_ADDR gm, size_t size, const char* what)
{
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "Failed to open " << what;
    f.read(reinterpret_cast<char*>(gm), size);
    ASSERT_TRUE(f.good()) << "Failed to read " << what << " (expected " << size << " bytes)";
}

// 通用 Cube smoke 执行体：准备数据 → 填 tiling → KERNEL_RUN_KF（仅崩溃检测，与 PR #61 一致）。
template <typename AType, typename BType, typename Func>
void RunTypedCubeSmoke(Func kernelFunc, const CubeCaseCfg& cfg)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;
    const size_t batchCount = static_cast<size_t>(cfg.batch);

    const size_t x1SingleBatchSize = static_cast<size_t>(M) * K * sizeof(AType);
    const size_t x2SingleBatchSize = static_cast<size_t>(K) * N * sizeof(BType);
    const size_t pertokenScaleCount = cfg.x1QuantMode == QM_PERTOKEN ? static_cast<size_t>(M) : 1UL;
    const size_t pertokenScaleSize = pertokenScaleCount * sizeof(float);
    const size_t scaleCount = cfg.x2QuantMode == QM_PERCHANNEL ? static_cast<size_t>(N) : 1UL;
    const size_t scaleSize = scaleCount * cfg.scaleElemSize;
    const size_t biasSingleBatchSize = static_cast<size_t>(N) * cfg.biasElemSize;
    const size_t x1Size = x1SingleBatchSize * batchCount;
    const size_t x2Size = x2SingleBatchSize * batchCount;
    const size_t biasSize = biasSingleBatchSize * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0U);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0U);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    if constexpr (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) {
        std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
        ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1SingleBatchSize, "input_a.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2SingleBatchSize, "input_b.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(), pertokenScaleSize,
                                            "pertoken_scale.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSize, "scale.bin"));
        ReplicateGmBatch(x1GM.Get(), x1SingleBatchSize, cfg.batch);
        ReplicateGmBatch(x2GM.Get(), x2SingleBatchSize, cfg.batch);

        if (cfg.isBias) {
            ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSingleBatchSize, "bias.bin"));
            if (cfg.biasThreeDim) {
                ReplicateGmBatch(biasGM.Get(), biasSingleBatchSize, cfg.batch);
            }
        }
    } else {
        // Zero is valid for FP8/HiFloat8 inputs. Keep typed smoke data deterministic and avoid interpreting
        // arbitrary int8 generator output as low-precision floating-point bit patterns.
        FillGmValues<float>(pertokenScaleGM.Get(), pertokenScaleCount, 1.0F);
        if (cfg.scaleElemSize == sizeof(float)) {
            FillGmValues<float>(scaleGM.Get(), scaleCount, 1.0F);
        } else {
            ASSERT_EQ(cfg.scaleElemSize, sizeof(uint64_t));
            constexpr uint64_t FIXPIPE_SCALE_ONE = 0x3F80000000000000UL;
            FillGmValues<uint64_t>(scaleGM.Get(), scaleCount, FIXPIPE_SCALE_ONE);
        }
    }

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillCubeTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

template <typename Func>
void RunCubeSmoke(Func kernelFunc, const CubeCaseCfg& cfg)
{
    RunTypedCubeSmoke<int8_t, int8_t>(kernelFunc, cfg);
}

template <typename AType, typename BType, typename X2ScaleType, typename Func>
void RunTypedMixSmoke(Func kernelFunc, const MixCaseCfg& cfg, bool loadGeneratedData = true)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;
    const size_t batchCount = static_cast<size_t>(cfg.batch);
    const size_t x1ScaleCount = cfg.x1QuantMode == QM_PERTOKEN ? static_cast<size_t>(M) : 1UL;
    const size_t x2ScaleCount = cfg.x2QuantMode == QM_PERCHANNEL ? static_cast<size_t>(N) : 1UL;
    const size_t x1SingleBatchSize = static_cast<size_t>(M) * K * sizeof(AType);
    const size_t x2SingleBatchSize = static_cast<size_t>(K) * N * sizeof(BType);
    const size_t pertokenScaleSingleBatchSize = x1ScaleCount * sizeof(float);
    const size_t scaleSingleBatchSize = x2ScaleCount * sizeof(X2ScaleType);
    const size_t biasSingleBatchSize = static_cast<size_t>(N) * cfg.biasElemSize;
    const size_t x1Size = x1SingleBatchSize * batchCount;
    const size_t x2Size = x2SingleBatchSize * batchCount;
    const size_t pertokenScaleSize = pertokenScaleSingleBatchSize * batchCount;
    const size_t scaleSize = scaleSingleBatchSize * batchCount;
    const size_t biasSize = biasSingleBatchSize * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmValues<float>(pertokenScaleGM.Get(), x1ScaleCount * batchCount, 1.0F);
    FillMixScaleOnes<X2ScaleType>(scaleGM.Get(), x2ScaleCount * batchCount);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    if (loadGeneratedData) {
        if constexpr (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) {
            std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
            ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

            ASSERT_NO_FATAL_FAILURE(
                ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1SingleBatchSize, "input_a.bin"));
            ASSERT_NO_FATAL_FAILURE(
                ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2SingleBatchSize, "input_b.bin"));
            ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(),
                                                pertokenScaleSingleBatchSize, "pertoken_scale.bin"));
            if constexpr (std::is_same_v<X2ScaleType, float>) {
                ASSERT_NO_FATAL_FAILURE(
                    ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSingleBatchSize, "scale.bin"));
            }
            if (cfg.isBias) {
                ASSERT_NO_FATAL_FAILURE(
                    ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSingleBatchSize, "bias.bin"));
            }
        }
    }

    ReplicateGmBatch(x1GM.Get(), x1SingleBatchSize, cfg.batch);
    ReplicateGmBatch(x2GM.Get(), x2SingleBatchSize, cfg.batch);
    ReplicateGmBatch(pertokenScaleGM.Get(), pertokenScaleSingleBatchSize, cfg.batch);
    ReplicateGmBatch(scaleGM.Get(), scaleSingleBatchSize, cfg.batch);

    if (cfg.isBias && cfg.biasThreeDim) {
        ReplicateGmBatch(biasGM.Get(), biasSingleBatchSize, cfg.batch);
    }

    QBMMV3TilingData* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMixTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

template <typename Func>
void RunMixSmoke(Func kernelFunc, const MixCaseCfg& cfg)
{
    RunTypedMixSmoke<int8_t, int8_t, float>(kernelFunc, cfg);
}

struct RuntimeBiasCase {
    uint32_t dtype;
    size_t elemSize;
};

template <typename AType, typename BType, typename X2ScaleType, typename Func>
void RunMixRuntimeModes(Func kernelFunc, int64_t k, size_t outElemSize)
{
    constexpr uint32_t X1_QUANT_MODES[] = {QM_DEFAULT, QM_PERTENSOR, QM_PERTOKEN};
    constexpr uint32_t X2_QUANT_MODES[] = {QM_PERTENSOR, QM_PERCHANNEL};
    constexpr RuntimeBiasCase BIAS_CASES[] = {
        {GE_DT_FLOAT, sizeof(float)}, {GE_DT_FLOAT16, sizeof(half)}, {GE_DT_BF16, sizeof(bfloat16_t)}};

    for (uint32_t x1QuantMode : X1_QUANT_MODES) {
        for (uint32_t x2QuantMode : X2_QUANT_MODES) {
            MixCaseCfg noBiasCfg{16,          16, k, 1, x1QuantMode, x2QuantMode, false, GE_DT_FLOAT, sizeof(float),
                                 outElemSize, ""};
            {
                SCOPED_TRACE(testing::Message()
                             << "x1QuantMode=" << x1QuantMode << ", x2QuantMode=" << x2QuantMode << ", without bias");
                ASSERT_NO_FATAL_FAILURE((RunTypedMixSmoke<AType, BType, X2ScaleType>(kernelFunc, noBiasCfg, false)));
            }

            for (const auto& biasCase : BIAS_CASES) {
                MixCaseCfg biasCfg{
                    16, 16, k, 1, x1QuantMode, x2QuantMode, true, biasCase.dtype, biasCase.elemSize, outElemSize, ""};
                SCOPED_TRACE(testing::Message() << "x1QuantMode=" << x1QuantMode << ", x2QuantMode=" << x2QuantMode
                                                << ", biasDtype=" << biasCase.dtype);
                ASSERT_NO_FATAL_FAILURE((RunTypedMixSmoke<AType, BType, X2ScaleType>(kernelFunc, biasCfg, false)));
            }
        }
    }
}

template <typename AType, typename BType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE,
          bool WithoutBatch = false>
void RunMxL0CPingpongSmoke(const L0CPingpongCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t batchCount = static_cast<size_t>(WithoutBatch ? 1 : cfg.batch);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K) * batchCount;
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N) * batchCount;
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(float);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(half) * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMML0CPingpongTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMML0CPingpongTilingData*>(tilingGM.Get());
    *tilingData = QBMMUT::QBMML0CPingpongTilingData{
        cfg.M,     cfg.N,          cfg.K,     static_cast<int64_t>(batchCount),
        cfg.baseM, cfg.baseN,      cfg.baseK, cfg.kL1,
        cfg.kL1,   cfg.nBufferNum, 2U,        0U,
        1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_l0c_pingpong_kernel_entry<AType, BType, half, float>;
    if constexpr (WithoutBatch) {
        if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
            kernelFunc = qbmm_mx_l0c_pingpong_without_batch_a_full_load_kernel_entry<AType, BType, half, float>;
        } else {
            kernelFunc = qbmm_mx_l0c_pingpong_without_batch_kernel_entry<AType, BType, half, float>;
        }
    } else if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_l0c_pingpong_a_full_load_kernel_entry<AType, BType, half, float>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX L0C ping-pong kernel execution failed";
}

template <typename AType = fp8_e4m3fn_t, typename BType = fp8_e5m2_t, bool ScaleNz = false>
void RunMxStreamKSmoke(const StreamKCaseCfg& cfg)
{
    using CType = half;
    using BiasType = float;

    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K);
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N);
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleN = ScaleNz ? Blaze::Gemm::CeilAlign(cfg.N, int64_t{Blaze::Gemm::MATMUL_MNK_ALIGN}) : cfg.N;
    const size_t scaleSize = scaleKLen * scaleN * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType);
    const size_t workspaceSize = cfg.blockNum * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);
    FillGmBuffer(workspaceGM.Get(), workspaceSize, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMMStreamKTilingData*>(tilingGM.Get());
    *tilingData = QBMMUT::QBMMStreamKTilingData{cfg.M, cfg.N, cfg.K, 1,  cfg.blockNum, 16, 16,
                                                64,    64,    64,    64, 1U,           0U, 1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_streamk_kernel_entry<AType, BType, CType, BiasType, ScaleNz>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX StreamK kernel execution failed";
}

template <typename AType, typename BType, typename X2ScaleType, typename OutType, typename BiasType>
void RunPertensorStreamKSmoke(const PertensorStreamKCaseCfg& cfg)
{
    const size_t x1Size = static_cast<size_t>(cfg.M * cfg.K) * sizeof(AType);
    const size_t x2Size = static_cast<size_t>(cfg.K * cfg.N) * sizeof(BType);
    const size_t biasSize = static_cast<size_t>(cfg.N) * cfg.biasElemSize;
    const size_t outSize = static_cast<size_t>(cfg.M * cfg.N) * sizeof(OutType);
    const size_t workspaceSize = cfg.blockNum * PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer perTokenScaleGM(sizeof(float));
    GmBuffer scaleGM(sizeof(X2ScaleType));
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(outSize);
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(perTokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmValues<float>(perTokenScaleGM.Get(), 1U, 1.0F);
    FillPertensorStreamKScaleOne<X2ScaleType>(scaleGM.Get());
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), outSize, 0U);
    FillGmBuffer(workspaceGM.Get(), workspaceSize, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM.Get());
    *tilingData = {cfg.M,     cfg.N,     cfg.K,           1,       cfg.blockNum,         cfg.baseM,
                   cfg.baseN, cfg.baseK, cfg.singleCoreK, cfg.kL1, cfg.isBias ? 1U : 0U, cfg.biasDtype};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_pertensor_streamk_kernel_entry<AType, BType, X2ScaleType, OutType, BiasType>;
    GM_ADDR x1ScaleGM = cfg.hasX1Scale ? perTokenScaleGM.Get() : nullptr;
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), x1ScaleGM, scaleGM.Get(),
                                  biasGM.Get(), yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM per-tensor StreamK kernel execution failed";
    // tikicpulib does not model RegTensor dequantization numerics. These cases validate that the production
    // GemmUniversal assembly, scheduler, AIC/AIV synchronization, and valid bias placement execute together.
}

template <typename AType, typename BType, typename X2ScaleType, typename OutType, typename BiasType>
void RunPertensorStreamKPostBiasModes(PertensorStreamKCaseCfg cfg)
{
    constexpr RuntimeBiasCase BIAS_CASES[] = {
        {GE_DT_FLOAT, sizeof(float)}, {GE_DT_FLOAT16, sizeof(half)}, {GE_DT_BF16, sizeof(bfloat16_t)}};
    constexpr bool X1_SCALE_MODES[] = {false, true};
    cfg.isBias = true;

    for (bool hasX1Scale : X1_SCALE_MODES) {
        cfg.hasX1Scale = hasX1Scale;
        for (const auto& biasCase : BIAS_CASES) {
            cfg.biasDtype = biasCase.dtype;
            cfg.biasElemSize = biasCase.elemSize;
            SCOPED_TRACE(testing::Message() << "hasX1Scale=" << hasX1Scale << ", biasDtype=" << biasCase.dtype);
            ASSERT_NO_FATAL_FAILURE((RunPertensorStreamKSmoke<AType, BType, X2ScaleType, OutType, BiasType>(cfg)));
        }
    }
}

void RunBatchInputRejectedSmoke()
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr int64_t K = 128;
    constexpr int64_t BATCH = 2;
    constexpr uint32_t BLOCK_NUM = 2U;
    constexpr uint8_t OUTPUT_SENTINEL = 0x5AU;
    const size_t workspaceSize = BLOCK_NUM * PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(static_cast<size_t>(BATCH * M * K) * sizeof(int8_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(int8_t));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer yGM(static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t));
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(yGM.Get(), static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t), OUTPUT_SENTINEL);
    auto* tilingData = reinterpret_cast<QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM.Get());
    *tilingData = {M, N, K, BATCH, BLOCK_NUM, 16, 16, 64, 64, 64, 0, QBMMUT::GE_DT_FLOAT};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_pertensor_streamk_kernel_entry<int8_t, int8_t, float, bfloat16_t, int32_t>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, BLOCK_NUM, x1GM.Get(), x2GM.Get(), nullptr, scaleGM.Get(), nullptr,
                                  yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM batched input rejection kernel launch failed";
    const auto* output = reinterpret_cast<const uint8_t*>(yGM.Get());
    const size_t outputBytes = static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t);
    EXPECT_TRUE(std::all_of(output, output + outputBytes, [](uint8_t value) { return value == OUTPUT_SENTINEL; }));
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
void RunMxSmoke(const MxCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t batchCount = static_cast<size_t>(cfg.batch);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K) * batchCount;
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N) * batchCount;
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType) * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType) * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMxTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_kernel_entry<AType, BType, CType, BiasType>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_a_full_load_kernel_entry<AType, BType, CType, BiasType>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX kernel execution failed";
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
void RunMxWithoutBatchSmoke(const MxCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K);
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N);
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType);

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMxTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_without_batch_kernel_entry<AType, BType, CType, BiasType>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_without_batch_a_full_load_kernel_entry<AType, BType, CType, BiasType>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX without-batch kernel execution failed";
}

} // namespace

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_AFullLoad)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_X1Scale_MultiBatch)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x1_mode pertensor --x2_mode pertensor --scale_dtype float32"};
    cfg.batch = 2;
    cfg.scaleElemSize = sizeof(float);
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_X1Scale_MultiBatch_AFullLoad)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x1_mode pertensor --x2_mode pertensor --scale_dtype float32"};
    cfg.batch = 2;
    cfg.scaleElemSize = sizeof(float);
    auto kernelFunc = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_MultiBatchWithThreeDimBias)
{
    CubeCaseCfg cfg{16,
                    128,
                    64,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32"};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FIXPIPE_BlockMmad_AL1SplitBL1)
{
    CubeCaseCfg cfg{16,
                    16,
                    96,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32"};
    cfg.kL1 = 64;
    cfg.kBL1 = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FIXPIPE_BlockMmad_BL1SplitAL1)
{
    CubeCaseCfg cfg{16,
                    16,
                    96,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32"};
    cfg.kL1 = 32;
    cfg.kBL1 = 64;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FIXPIPE_BlockMmad_AsymmetricL1_WithoutBatch_U64Scale)
{
    CubeCaseCfg cfg{16,
                    16,
                    96,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel"};
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;

    cfg.kL1 = 64;
    cfg.kBL1 = 32;
    RunCubeSmoke(kernelFunc, cfg);

    cfg.kL1 = 32;
    cfg.kBL1 = 64;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FIXPIPE_BlockMmad_AsymmetricL1_WithoutBatch_FloatScale)
{
    CubeCaseCfg cfg{16,
                    16,
                    96,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel"};
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t, float>;

    cfg.kL1 = 64;
    cfg.kBL1 = 32;
    RunCubeSmoke(kernelFunc, cfg);

    cfg.kL1 = 32;
    cfg.kBL1 = 64;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FIXPIPE_BlockMmad_PerChannel_AFullLoad)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel"};
    auto kernelFunc = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_CubeTiling_DefaultSingleTile)
{
    CubeCaseCfg cfg{16, 32, 64, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    QBMMV3TilingData tiling{};
    FillCubeTiling(&tiling, cfg);

    EXPECT_EQ(tiling.baseM, 16);
    EXPECT_EQ(tiling.baseN, 32);
    EXPECT_EQ(tiling.baseM_qbmm, 16U);
    EXPECT_EQ(tiling.baseN_qbmm, 32U);
    EXPECT_EQ(tiling.kAL1, 64U);
    EXPECT_EQ(tiling.kBL1, 64U);
    EXPECT_EQ(tiling.baseK_qbmm, 64U);
    EXPECT_EQ(tiling.nBufferNum, 2U);
}

TEST_F(QBMMV3Test, Test_CubeTiling_TripleBufferWithTails)
{
    CubeCaseCfg cfg{48,
                    80,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseM = 32;
    cfg.baseN = 32;
    cfg.baseK = 32;
    QBMMV3TilingData tiling{};
    FillCubeTiling(&tiling, cfg);

    EXPECT_EQ(tiling.m, 48);
    EXPECT_EQ(tiling.n, 80);
    EXPECT_EQ(tiling.k, 288);
    EXPECT_EQ(tiling.baseM, 32);
    EXPECT_EQ(tiling.baseN, 32);
    EXPECT_EQ(tiling.baseM_qbmm, 32U);
    EXPECT_EQ(tiling.baseN_qbmm, 32U);
    EXPECT_EQ(tiling.kAL1, 64U);
    EXPECT_EQ(tiling.kBL1, 64U);
    EXPECT_EQ(tiling.baseK_qbmm, 32U);
    EXPECT_EQ(tiling.nBufferNum, 3U);
    // Five L1 iterations exercise 0 -> 1 -> 2 -> 0 -> 1, with a 32-element K tail.
    EXPECT_EQ((tiling.k + tiling.kAL1 - 1) / tiling.kAL1, 5);
    EXPECT_EQ(tiling.k % tiling.kAL1, 32);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_WithoutBatch_TripleBuffer)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_WithoutBatch_TripleBuffer_AFullLoad)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    // Reuse the resident A tile across three N tiles on the same core.
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_AFullLoad)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel"};
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_TripleBuffer)
{
    CubeCaseCfg cfg{48,
                    80,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseM = 32;
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_X1Scale_WithoutBatch)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x1_mode pertensor --x2_mode pertensor --scale_dtype float32"};
    cfg.scaleElemSize = sizeof(float);
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t, float>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_DoubleBuffer_L0CPingpong)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    // Five L1 iterations wrap the double buffer; three N tiles toggle the L0C buffer twice.
    cfg.dbL0C = static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT);
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_QuadBuffer)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::QUADRUPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    // Five L1 iterations exercise buffer IDs 0 -> 1 -> 2 -> 3 -> 0.
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FP8_A8W8_PERTENSOR_WithoutBatch_AFullLoad_DoubleBuffer)
{
    CubeCaseCfg cfg{32,
                    64,
                    160,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(float),
                    sizeof(bfloat16_t),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    cfg.scaleElemSize = sizeof(float);
    // Three L1 iterations wrap the double-buffered B path; two N tiles reuse the fully loaded A tile.
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float,
                                                                       float>;
    RunTypedCubeSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_HIFLOAT8_A8W8_PERTENSOR_WithoutBatch_AFullLoad_QuadBuffer)
{
    CubeCaseCfg cfg{32,
                    64,
                    288,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(float),
                    sizeof(bfloat16_t),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::QUADRUPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    cfg.scaleElemSize = sizeof(float);
    // Five L1 iterations cover every B buffer and wrap to buffer 0 in A-full-load mode.
    auto
        kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<hifloat8_t, hifloat8_t, bfloat16_t, float, float>;
    RunTypedCubeSmoke<hifloat8_t, hifloat8_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FP8_A8W8_PERCHANNEL_WithoutBatch_AFullLoad)
{
    CubeCaseCfg cfg{16, 16, 64, 1, QM_DEFAULT, QM_PERCHANNEL, false, GE_DT_FLOAT, sizeof(float), sizeof(bfloat16_t),
                    ""};
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float,
                                                                       float>;
    RunTypedCubeSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_HIFLOAT8_A8W8_PERCHANNEL_WithoutBatch_AFullLoad)
{
    CubeCaseCfg cfg{16, 16, 64, 1, QM_DEFAULT, QM_PERCHANNEL, false, GE_DT_FLOAT, sizeof(float), sizeof(bfloat16_t),
                    ""};
    auto
        kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<hifloat8_t, hifloat8_t, bfloat16_t, float, float>;
    RunTypedCubeSmoke<hifloat8_t, hifloat8_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadDoubleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadDoubleBuffer_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    cfg.batch = 2;
    RunMxSmoke<MxType, MxType, float, float, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_MultiBatchWithThreeDimBias)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{16, 128, 128, 1, 16, 128, 64, 64, 64, 2, true};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 192, 1, 64, 128, 64, 64, 64, 3, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadQuadBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 256, 1, 64, 128, 64, 64, 64, 4, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchDoubleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchDoubleBuffer_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 192, 1, 64, 128, 64, 64, 64, 3, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchQuadBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 256, 1, 64, 128, 64, 64, 64, 4, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpong)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    cfg.batch = 2;
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpong_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    cfg.batch = 2;
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongWithoutBatch)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::NONE_FULL_LOAD_MODE, true>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongWithoutBatch_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::A_FULL_LOAD_MODE, true>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 192, 64, 128, 64, 64, 3, 1};
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP4_L0CPingpongSplitN)
{
    using MxType = fp4x2_e2m1_t;
    L0CPingpongCaseCfg cfg{128, 256, 128, 128, 256, 64, 64, 2, 1};
    cfg.batch = 2;
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_StreamK)
{
    StreamKCaseCfg cfg{16, 16, 128, 2};
    RunMxStreamKSmoke(cfg);
}

// ===================== MIX 完整 kernel 路径用例 =====================
// 均通过 QBMMMixWrapper 装配并由 KERNEL_RUN_KF 执行，不直接调用 block epilogue。

TEST_F(QBMMV3Test, Test_MIX_BlockScheduler_TailSplit)
{
    MixCaseCfg cfg{64,
                   64,
                   16,
                   24,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    cfg.baseM = 64;
    cfg.baseN = 64;
    cfg.mTailTile = 4;
    cfg.nTailTile = 4;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_BlockScheduler_TailSplit_AFullLoad)
{
    MixCaseCfg cfg{256,
                   64,
                   16,
                   24,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    cfg.baseM = 64;
    cfg.baseN = 64;
    cfg.nTailTile = 4;
    auto kernelFunc = qbmm_mix_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_BlockScheduler_MultiBatchNonWrappedRound)
{
    MixCaseCfg cfg{104,
                   80,
                   16,
                   24,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    cfg.batch = 2;
    cfg.baseM = 16;
    cfg.baseN = 16;
    cfg.mBaseTailSplitCnt = 2;
    cfg.mTailMain = 12;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_BlockScheduler_MultiBatchWrappedRoundAndSwat)
{
    MixCaseCfg cfg{224,
                   48,
                   16,
                   24,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    cfg.batch = 2;
    cfg.baseM = 16;
    cfg.baseN = 16;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_BlockMmad_AL1SplitBL1)
{
    MixCaseCfg cfg{16,
                   16,
                   96,
                   1,
                   QM_DEFAULT,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    cfg.kAL1 = 64;
    cfg.kBL1 = 32;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_BlockMmad_BL1SplitAL1)
{
    MixCaseCfg cfg{16,
                   16,
                   96,
                   1,
                   QM_DEFAULT,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    cfg.kAL1 = 32;
    cfg.kBL1 = 64;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_RuntimeModes_OutputFloat)
{
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, float, int32_t>;
    RunMixRuntimeModes<int8_t, int8_t, float>(kernelFunc, 16, sizeof(float));
}

TEST_F(QBMMV3Test, Test_MIX_BlockMmad_AsymmetricL1_WithoutBatch)
{
    MixCaseCfg cfg{16,
                   16,
                   96,
                   1,
                   QM_DEFAULT,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;

    cfg.kAL1 = 64;
    cfg.kBL1 = 32;
    RunMixSmoke(kernelFunc, cfg);

    cfg.kAL1 = 32;
    cfg.kBL1 = 64;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_RuntimeModes_OutputFP16)
{
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixRuntimeModes<int8_t, int8_t, float>(kernelFunc, 16, sizeof(half));
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_PerToken_AFullLoad)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    cfg.batch = 2;
    auto kernelFunc = qbmm_mix_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_MultiBatchWithThreeDimBias)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   true,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32 --bias --bias_dtype float32"};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_RuntimeModes_BF16Scale_OutputFloat)
{
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, float, int32_t, bfloat16_t>;
    RunMixRuntimeModes<int8_t, int8_t, bfloat16_t>(kernelFunc, 16, sizeof(float));
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_RuntimeModes_OutputBF16)
{
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, bfloat16_t, int32_t>;
    RunMixRuntimeModes<int8_t, int8_t, float>(kernelFunc, 16, sizeof(bfloat16_t));
}

TEST_F(QBMMV3Test, Test_MIX_FP8_RuntimeModes_OutputFP16)
{
    auto kernelFunc = qbmm_mix_kernel_entry<fp8_e4m3fn_t, fp8_e4m3fn_t, half, float>;
    RunMixRuntimeModes<fp8_e4m3fn_t, fp8_e4m3fn_t, float>(kernelFunc, 64, sizeof(half));
}

// 单 batch 特化：走 GemmUniversal without_batch，激活 per-token + 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch_AFullLoad)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// MIX wrapper with selectable transpose, weight layout and batch specialization.
template <class AType_, class BType_, class OutType_, bool TransA_, bool TransB_, bool WithoutBatch_,
          uint64_t FullLoadMode_, bool WeightNz_>
__global__ __aicore__ void QbmmMixMigrationEntry(GM_ADDR a, GM_ADDR b, GM_ADDR s1, GM_ADDR s2, GM_ADDR bias,
                                                 GM_ADDR out, GM_ADDR tiling)
{
    using LayoutA = AscendC::Std::conditional_t<TransA_, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NdLayoutB = AscendC::Std::conditional_t<TransB_, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NzLayoutB = AscendC::Std::conditional_t<TransB_, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<WeightNz_, NzLayoutB, NdLayoutB>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Schedule = AscendC::Std::conditional_t<WithoutBatch_, Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch,
                                                 Blaze::Gemm::KernelMmadWithScaleMix>;
    using Policy = Blaze::Gemm::MatmulWithScaleMix<FullLoadMode_, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType_, LayoutA, AscendC::Std::tuple<BType_, float>, LayoutB,
                                               OutType_, LayoutC, int32_t, LayoutC>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType_, int32_t, float, float,
                                                                  typename Mmad::L0CType>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<Shape, FullLoadMode_, LayoutA, LayoutB,
                                                                           AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;
    const auto& td = *reinterpret_cast<const QBMMV3TilingData*>(tiling);
    typename Kernel::Params params{};
    params.problemShape = {td.m, td.n, td.k, td.b};
    if constexpr (WithoutBatch_) {
        QBMMUT::FillMixMmadParams(params.mmParams, a, b, bias, td);
    } else {
        QBMMUT::FillMixMmadParams(params.mmadParams, a, b, bias, td);
        QBMMUT::FillQbmmBatchParams(params.qbmmParams, td);
        QBMMUT::FillQbmmTileParams(params.qbmmParams, td);
    }
    QBMMUT::FillQbmmSchParams(params.schParams, td);
    QBMMUT::FillEpilogueParams(params.epilogueParams, s1, s2, bias, out, td);
    Kernel kernel;
    kernel(params);
}

namespace {
template <class AType_, class BType_, class OutType_, bool TransA_, bool TransB_, bool WithoutBatch_,
          uint64_t FullLoadMode_ = Blaze::Gemm::NONE_FULL_LOAD_MODE, bool WeightNz_ = false>
void RunMix(uint32_t x1Mode, uint32_t x2Mode, uint32_t biasDtype, uint32_t kAL1 = 64, uint32_t kBL1 = 64,
            uint32_t buffers = 2)
{
    constexpr int64_t M = 16, N = 32, K = 64;
    constexpr int64_t BATCH = WithoutBatch_ ? 1 : 2;
    GmBuffer a(M * K * sizeof(AType_)), b(BATCH * K * N * sizeof(BType_));
    GmBuffer s1(M * sizeof(float)), s2(N * sizeof(float));
    GmBuffer bias(BATCH * N * sizeof(int32_t)), out(BATCH * M * N * sizeof(OutType_));
    GmBuffer tiling(sizeof(QBMMV3TilingData));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(s1.Get(), nullptr);
    ASSERT_NE(s2.Get(), nullptr);
    ASSERT_NE(bias.Get(), nullptr);
    ASSERT_NE(out.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);
    std::fill_n(a.Get(), M * K * sizeof(AType_), 0);
    std::fill_n(b.Get(), BATCH * K * N * sizeof(BType_), 0);
    std::fill_n(reinterpret_cast<float*>(s1.Get()), M, 0.5F);
    std::fill_n(reinterpret_cast<float*>(s2.Get()), N, 2.0F);
    std::fill_n(bias.Get(), BATCH * N * sizeof(int32_t), 0);
    auto& td = *reinterpret_cast<QBMMV3TilingData*>(tiling.Get());
    td = {};
    td.m = M;
    td.n = N;
    td.k = K;
    td.b = BATCH;
    // Match host tiling: transposed A uses a C0-aligned tile even when logical M is smaller.
    td.baseM = td.baseM_qbmm = TransA_ ? 32 : M;
    td.baseN = td.baseN_qbmm = N;
    td.baseK_qbmm = 32;
    td.kAL1 = kAL1;
    td.kBL1 = kBL1;
    td.nBufferNum = buffers;
    td.dbL0C = 1;
    td.mTailTile = td.nTailTile = td.mBaseTailSplitCnt = td.nBaseTailSplitCnt = 1;
    td.batchA1 = td.batchA2 = td.batchA3 = td.batchA4 = 1;
    td.batchB1 = td.batchB2 = td.batchB3 = 1;
    td.batchB4 = BATCH;
    td.batchC1 = td.batchC2 = td.batchC3 = 1;
    td.batchC4 = BATCH;
    td.biasThreeDim = WithoutBatch_ ? 0 : 1;
    td.isBias = 1;
    td.biasDtype = biasDtype;
    td.x1QuantMode = x1Mode;
    td.x2QuantMode = x2Mode;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = QbmmMixMigrationEntry<AType_, BType_, OutType_, TransA_, TransB_, WithoutBatch_, FullLoadMode_,
                                        WeightNz_>;
    EXPECT_TRUE(KERNEL_RUN_KF(kernel, 1, a.Get(), b.Get(), s1.Get(), s2.Get(), bias.Get(), out.Get(), tiling.Get()));
    // CPU-debug checks kernel execution; RegTensor output accuracy requires device tests.
}
} // namespace

TEST(QbmmMixNdMigration, Int8FourTransposesAndIntegerBias)
{
    for (uint32_t x1 : {0U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<int8_t, int8_t, half, false, false, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, false, true, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, half, true, false, false>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, false>(x1, x2, DT_INT32);
        }
    }
}

TEST(QbmmMixNdMigration, Fp8PairsAndHif8ScaleModes)
{
    for (uint32_t x1 : {1U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<fp8_e4m3fn_t, fp8_e4m3fn_t, half, false, false, true>(x1, x2, DT_FLOAT);
            RunMix<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true, false>(x1, x2, DT_FLOAT);
            RunMix<fp8_e5m2_t, fp8_e4m3fn_t, float, true, false, false>(x1, x2, DT_FLOAT);
            RunMix<fp8_e5m2_t, fp8_e5m2_t, half, true, true, true>(x1, x2, DT_FLOAT);
            RunMix<hifloat8_t, hifloat8_t, float, true, true, false>(x1, x2, DT_FLOAT);
        }
    }
}

TEST(QbmmMixNdMigration, SingleBatchAl1)
{
    RunMix<int8_t, int8_t, bfloat16_t, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE>(4, 2, DT_INT32);
    RunMix<fp8_e5m2_t, fp8_e4m3fn_t, float, false, true, true, Blaze::Gemm::A_FULL_LOAD_MODE>(1, 1, DT_FLOAT);
}

TEST(QbmmMixNdMigration, Int8NzTransposedA)
{
    for (uint32_t x1 : {0U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<int8_t, int8_t, half, true, false, true, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, half, true, true, true, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, false, false, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, false, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE, true>(x1, x2,
                                                                                                       DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, true, Blaze::Gemm::A_FULL_LOAD_MODE, true>(x1, x2, DT_INT32);
        }
    }
}

TEST(QbmmMixNdMigration, IntegerBiasSplitKAndFourBuffers)
{
    // All K-loop nesting orders must initialize bias once per output tile.
    for (auto depths : {std::pair{32U, 64U}, std::pair{64U, 32U}, std::pair{32U, 32U}}) {
        RunMix<int8_t, int8_t, half, true, false, false>(4, 2, DT_INT32, depths.first, depths.second);
        RunMix<int8_t, int8_t, half, true, true, false, 0, true>(4, 2, DT_INT32, depths.first, depths.second);
    }
    RunMix<int8_t, int8_t, half, false, false, false>(4, 2, DT_INT32, 32, 32, 4);
    RunMix<int8_t, int8_t, half, true, true, false, 0, true>(4, 2, DT_INT32, 32, 32, 4);
    RunMix<int8_t, int8_t, half, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE>(4, 2, DT_INT32, 64, 32, 4);
}

namespace {

using ND = asc::te::nd_ext_layout_ptn;
using DN = asc::te::dn_ext_layout_ptn;
using NZ = asc::te::nz_layout_ptn;
using ZN = asc::te::zn_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

template <class Kernel>
inline constexpr bool HAS_KERNEL_CALL = std::is_same_v<
    decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>;

template <class Schedule, class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA,
          class LayoutB, uint64_t FullLoadMode>
void CheckCubeTensorApiAssembly()
{
    using Types = QBMMUT::QBMMCubeTypes<AType, BType, ScaleType, OutType, BiasType, LayoutA, LayoutB, ND, FullLoadMode,
                                        Schedule>;
    using Kernel = typename Types::Kernel;
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType, Schedule>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Types::BlockMmad::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutC, ND>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue, Blaze::Epilogue::Block::BlockEpilogueEmpty>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA,
          class LayoutB, uint64_t FullLoadMode>
void CheckMixTensorApiAssembly()
{
    using Types = QBMMUT::QBMMMixTypes<AType, BType, OutType, ScaleType, float, BiasType, FullLoadMode, LayoutA,
                                       LayoutB, ND, Schedule>;
    using Kernel = typename Types::Kernel;
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType, Schedule>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Types::BTypeTuple, AscendC::Std::tuple<BType, ScaleType>>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::OutType, OutType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::BiasType, BiasType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::L0CType, typename Types::BlockMmad::L0CType>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA, LayoutB,
                                                                           AType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Schedule>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Mmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Mmad::LayoutC, ND>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxL0CPingpongTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA, LayoutB,
                                                                           AType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Schedule>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Mmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Mmad::LayoutC, ND>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxStreamKTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpiloguePolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, OutType, EpiloguePolicy>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Blaze::Gemm::KernelQbmmMultiBlockStreamK>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Epilogue::WorkspaceType, float>);
    static_assert(std::is_same_v<typename Epilogue::OutType, OutType>);
    static_assert(std::is_same_v<typename Epilogue::DispatchPolicy, EpiloguePolicy>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA, class LayoutB,
          uint64_t FullLoadMode>
void CheckPertensorStreamKTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode, false,
                                                            Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, AscendC::Std::tuple<BType, ScaleType>, LayoutB,
                                               OutType, ND, BiasType, ND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, OutType,
                                                                               Policy, ScaleType, float>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Epilogue::WorkspaceType, typename Mmad::WorkspaceType>);
    static_assert(std::is_same_v<typename Epilogue::DispatchPolicy, Policy>);
    static_assert(std::is_same_v<typename Epilogue::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Epilogue::X1ScaleType, float>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    EXPECT_GT(sizeof(Kernel), 0U);
}

// Mirrors arch35 dispatch reachability: Cube/MX allow both A layouts, all four B layouts and both load modes;
// MIX is limited to ND A + NZ/ZN B; StreamK is limited to NONE_FULL_LOAD_MODE.
template <class Schedule, class LayoutA, class LayoutB>
void CheckCubeLayoutAndLoadModes()
{
    CheckCubeTensorApiAssembly<Schedule, int8_t, int8_t, uint64_t, half, int32_t, LayoutA, LayoutB,
                               Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Schedule, int8_t, int8_t, uint64_t, half, int32_t, LayoutA, LayoutB,
                               Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckCubeBLayoutMatrix()
{
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckCubeLayoutMatrix()
{
    CheckCubeBLayoutMatrix<Schedule, ND>();
    CheckCubeBLayoutMatrix<Schedule, DN>();
}

template <class Schedule>
void CheckMixLayoutMatrix()
{
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, ZN,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, NZ,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, NZ,
                              Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, ZN,
                              Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA, class LayoutB>
void CheckMxLayoutAndLoadModes()
{
    CheckMxTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                             Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                             Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckMxBLayoutMatrix()
{
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckMxLayoutMatrix()
{
    CheckMxBLayoutMatrix<Schedule, ND>();
    CheckMxBLayoutMatrix<Schedule, DN>();
}

template <class Schedule, class LayoutA, class LayoutB>
void CheckMxL0CPingpongLayoutAndLoadModes()
{
    CheckMxL0CPingpongTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                                        Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxL0CPingpongTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                                        Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckMxL0CPingpongBLayoutMatrix()
{
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckMxL0CPingpongLayoutMatrix()
{
    CheckMxL0CPingpongBLayoutMatrix<Schedule, ND>();
    CheckMxL0CPingpongBLayoutMatrix<Schedule, DN>();
}

template <class LayoutA>
void CheckMxStreamKBLayoutMatrix()
{
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, DN, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, ZN, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

void CheckMxStreamKLayoutMatrix()
{
    CheckMxStreamKBLayoutMatrix<ND>();
    CheckMxStreamKBLayoutMatrix<DN>();
}

template <class LayoutA>
void CheckPertensorStreamKBLayoutMatrix()
{
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, ND,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, DN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, NZ,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, ZN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

void CheckPertensorStreamKLayoutMatrix()
{
    CheckPertensorStreamKBLayoutMatrix<ND>();
    CheckPertensorStreamKBLayoutMatrix<DN>();
}

} // namespace

TEST(QBMMTemplateContractTest, Arch35AssemblyMatrix)
{
    CheckCubeLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleFixpipeQuant>();
    CheckCubeLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch>();
    CheckMixLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMix>();
    CheckMixLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>();
    CheckMxLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMx>();
    CheckMxLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>();
    CheckMxL0CPingpongLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMx>();
    CheckMxL0CPingpongLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>();
    CheckMxStreamKLayoutMatrix();
    CheckPertensorStreamKLayoutMatrix();
}

TEST(QBMMTemplateContractTest, ProductionAssembliesPreserveArch35DtypeParameters)
{
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuant, hifloat8_t, hifloat8_t, bfloat16_t, float,
                               float, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch, fp8_e4m3fn_t, fp8_e5m2_t,
                               float, bfloat16_t, float, DN, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch, int8_t, int8_t, int64_t, half,
                               int32_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMix, hifloat8_t, hifloat8_t, float, float, float, ND, NZ,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float,
                              float, ND, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch, int8_t, int8_t, bfloat16_t, bfloat16_t,
                              int32_t, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMx, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, ND, NZ,
                             Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxL0CPingpongTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch, fp4x2_e2m1_t, fp4x2_e2m1_t,
                                        float, ND, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, uint64_t, float, float, DN, ZN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, int64_t, half, int32_t, ND, ND,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

class QBMMPertensorStreamKTest : public testing::Test {};

TEST_F(QBMMPertensorStreamKTest, TemplateContracts)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DefaultDispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Int8Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                                   Layout, half, Layout, int32_t, Layout>;
    using Int8Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Int8Mmad::WorkspaceType,
                                                                                   half, DispatchPolicy, float, float>;
    using Int8Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Mmad, Int8Epilogue, Scheduler>;
    using Fp8Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                                  AscendC::Std::tuple<fp8_e5m2_t, uint64_t>, Layout, float, Layout,
                                                  float, Layout>;
    using Fp8Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Fp8Mmad::WorkspaceType, float, DispatchPolicy, uint64_t, float>;
    using Fp8Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Fp8Mmad, Fp8Epilogue, Scheduler>;
    using Fp8PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                                          AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float,
                                                          Layout, float, Layout>;
    using Fp8PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Fp8PostBiasMmad::WorkspaceType, float, DispatchPolicy, float, float>;
    using Fp8PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Fp8PostBiasMmad, Fp8PostBiasEpilogue,
                                                                 Scheduler>;
    using Int8Fp32PostBiasMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>, Layout, bfloat16_t, Layout, float, Layout>;
    using Int8Fp32PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Int8Fp32PostBiasMmad::WorkspaceType, bfloat16_t, DispatchPolicy, float, float>;
    using Int8Fp32PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Fp32PostBiasMmad,
                                                                      Int8Fp32PostBiasEpilogue, Scheduler>;
    using Int8Bf16PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout,
                                                               AscendC::Std::tuple<int8_t, bfloat16_t>, Layout,
                                                               bfloat16_t, Layout, bfloat16_t, Layout>;
    using Int8Bf16PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Int8Bf16PostBiasMmad::WorkspaceType, bfloat16_t, DispatchPolicy, bfloat16_t, float>;
    using Int8Bf16PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Bf16PostBiasMmad,
                                                                      Int8Bf16PostBiasEpilogue, Scheduler>;

    static_assert(
        std::is_same_v<typename DispatchPolicy::ScheduleType, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>);
    static_assert(
        std::is_same_v<typename DefaultDispatchPolicy::ScheduleType, Blaze::Gemm::KernelMmadWithScaleFixpipeQuant>);
    static_assert(std::is_same_v<typename Int8Mmad::WorkspaceType, int32_t>);
    static_assert(std::is_same_v<typename Fp8Mmad::WorkspaceType, float>);
    static_assert(Int8Mmad::BIAS_IN_MMAD);
    static_assert(Fp8Mmad::BIAS_IN_MMAD);
    static_assert(!Fp8PostBiasMmad::BIAS_IN_MMAD);
    static_assert(!Int8Fp32PostBiasMmad::BIAS_IN_MMAD);
    static_assert(!Int8Bf16PostBiasMmad::BIAS_IN_MMAD);
    static_assert(std::is_same_v<typename Int8Epilogue::WorkspaceType, typename Int8Mmad::WorkspaceType>);
    static_assert(std::is_same_v<typename Int8Kernel::BlockMmad, Int8Mmad>);
    static_assert(std::is_same_v<typename Fp8Kernel::BlockMmad, Fp8Mmad>);
    static_assert(std::is_same_v<typename Fp8PostBiasKernel::BlockMmad, Fp8PostBiasMmad>);
    static_assert(std::is_same_v<typename Int8Fp32PostBiasKernel::BlockMmad, Int8Fp32PostBiasMmad>);
    static_assert(std::is_same_v<typename Int8Bf16PostBiasKernel::BlockMmad, Int8Bf16PostBiasMmad>);

    SUCCEED();
}

TEST_F(QBMMPertensorStreamKTest, SingleScaleWithoutPostBiasIsMaskedBeforeMultiply)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                               Layout, bfloat16_t, Layout, int32_t, Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, bfloat16_t,
                                                                               DispatchPolicy, float, float>;

    constexpr uint32_t rawScaleBits = 0x3F812345U;
    const float maskedScale = Epilogue::DecodeMaskedDequantScale(rawScaleBits);
    const uint32_t actualBits = *reinterpret_cast<const uint32_t*>(&maskedScale);

    EXPECT_EQ(actualBits, rawScaleBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, DoubleScaleWithoutPostBiasMergesBeforeMask)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                               AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float, Layout, float,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, float,
                                                                               DispatchPolicy, float, float>;

    constexpr float x2Scale = 1.013741F;
    constexpr float x1Scale = 0.987653F;
    const float actual = Epilogue::MergeAndMaskDequantScale(x2Scale, x1Scale);
    const float merged = x2Scale * x1Scale;
    const uint32_t mergedBits = *reinterpret_cast<const uint32_t*>(&merged);
    const uint32_t actualBits = *reinterpret_cast<const uint32_t*>(&actual);

    EXPECT_EQ(actualBits, mergedBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, Int8FloatScaleWithoutBiasPureSk)
{
    PertensorStreamKCaseCfg cfg{16, 16, 128, 2};
    RunPertensorStreamKSmoke<int8_t, int8_t, float, half, float>(cfg);
}

TEST_F(QBMMPertensorStreamKTest, Int8Uint64ScaleInt32MmadBiasPureSk)
{
    PertensorStreamKCaseCfg cfg{16, 16, 128, 2};
    cfg.isBias = true;
    cfg.biasDtype = GE_DT_INT32;
    cfg.biasElemSize = sizeof(int32_t);
    RunPertensorStreamKSmoke<int8_t, int8_t, uint64_t, half, int32_t>(cfg);
}

TEST_F(QBMMPertensorStreamKTest, Fp8DoubleScaleWithoutBiasDpSk)
{
    PertensorStreamKCaseCfg cfg{48, 17, 128, 4};
    cfg.hasX1Scale = true;
    RunPertensorStreamKSmoke<fp8_e4m3fn_t, fp8_e5m2_t, float, float, float>(cfg);
}

TEST_F(QBMMPertensorStreamKTest, Fp8DoubleScalePostBiasModesPureSk)
{
    PertensorStreamKCaseCfg cfg{16, 16, 128, 2};
    RunPertensorStreamKPostBiasModes<fp8_e4m3fn_t, fp8_e5m2_t, float, float, float>(cfg);
}

TEST_F(QBMMPertensorStreamKTest, Int8FloatScalePostBiasModesPureSk)
{
    PertensorStreamKCaseCfg cfg{16, 16, 128, 2};
    RunPertensorStreamKPostBiasModes<int8_t, int8_t, float, half, float>(cfg);
}

TEST_F(QBMMPertensorStreamKTest, BatchedInputReturnsBeforeScheduling) { RunBatchInputRejectedSmoke(); }

// Weight transpose and ScaleNZ storage are independent template parameters.
// An ND [N,K] weight uses DN, while ScaleB remains packed [N1,Kg,16,2].
template <class Policy, class WeightLayout>
void CheckMxIndependentScaleNzAssembly()
{
    using PackedScale = asc::te::nn_layout_ptn;
    using LayoutPair = AscendC::Std::tuple<WeightLayout, PackedScale>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp8_e4m3fn_t, ND, fp8_e4m3fn_t, LayoutPair, bfloat16_t, ND,
                                               float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, Policy::FULL_LOAD_MODE, ND,
                                                                           WeightLayout, fp8_e4m3fn_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                                      Scheduler>;
    static_assert(std::is_same_v<typename Mmad::LayoutB, WeightLayout>);
    static_assert(std::is_same_v<typename Mmad::LayoutScaleB, PackedScale>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    EXPECT_GT(sizeof(Kernel), 0U);
}

TEST(QBMMTemplateContractTest, NdTransposedWeightPreservesIndependentScaleNzLayout)
{
    using Regular = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                   Blaze::Gemm::KernelMmadWithScaleMx>;
    using Pingpong = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                               Blaze::Gemm::KernelMmadWithScaleMx>;
    using FullLoad = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<Blaze::Gemm::A_FULL_LOAD_MODE, false,
                                                               Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    CheckMxIndependentScaleNzAssembly<Regular, DN>();
    CheckMxIndependentScaleNzAssembly<Pingpong, DN>();
    CheckMxIndependentScaleNzAssembly<FullLoad, DN>();
    CheckMxIndependentScaleNzAssembly<Pingpong, NZ>();
}

TEST(QBMMTemplateContractTest, StreamKPreservesIndependentScaleNzLayout)
{
    using Policy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                  Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using LayoutPair = AscendC::Std::tuple<ZN, asc::te::nn_layout_ptn>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp8_e4m3fn_t, ND, fp8_e4m3fn_t, LayoutPair, float, ND, float,
                                               ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, float,
                                                                        Blaze::Gemm::MatmulMultiBlockWithStreamK<>>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Mmad::LayoutB, ZN>);
    static_assert(std::is_same_v<typename Mmad::LayoutScaleB, asc::te::nn_layout_ptn>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    EXPECT_GT(sizeof(Kernel), 0U);
}

TEST_F(QBMMV3Test, ScaleNzStreamKFp8WithKNAndMnTileTails)
{
    RunMxStreamKSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, true>({34, 33, 194, 4});
    RunMxStreamKSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, true>({66, 65, 256, 4});
}

TEST_F(QBMMV3Test, ScaleNzStreamKFp4WithKNAndMnTileTails)
{
    RunMxStreamKSmoke<fp4x2_e2m1_t, fp4x2_e2m1_t, true>({34, 33, 194, 4});
    RunMxStreamKSmoke<fp4x2_e2m1_t, fp4x2_e2m1_t, true>({66, 65, 256, 4});
}

// The NN Ascend950 INT8_FLOW/BF16_FLOW CSV entries both use M/K/N=4/8192/128,
// transposeB=true, INT8 inputs, uint64 scale and no bias. Exercise the same
// problem through Tensor's production A-full-load assembly for both outputs.
TEST_F(QBMMV3Test, NnPipelineInt8FlowTransposedWeight)
{
    CubeCaseCfg cfg{4, 128, 8192, 8, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_INT32, sizeof(int32_t), sizeof(half), ""};
    cfg.kL1 = 512;
    cfg.baseM = 16;
    cfg.baseN = 16;
    cfg.baseK = 128;
    auto entry = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t, uint64_t, DN>;
    RunCubeSmoke(entry, cfg);
}

TEST_F(QBMMV3Test, NnPipelineBf16FlowTransposedWeight)
{
    CubeCaseCfg cfg{4, 128, 8192, 8, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_INT32, sizeof(int32_t), sizeof(bfloat16_t),
                    ""};
    cfg.kL1 = 512;
    cfg.baseM = 16;
    cfg.baseN = 16;
    cfg.baseK = 128;
    auto entry = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, bfloat16_t, int32_t, uint64_t, DN>;
    RunCubeSmoke(entry, cfg);
}
