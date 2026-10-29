/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "acl/acl.h"
#include "platform/platform_ascendc.h"

/*============================================================================
 * ACL Error Check Macro
 *============================================================================*/

#define ACL_CHECK(expr)                                                                                             \
    do {                                                                                                            \
        aclError ret = (expr);                                                                                      \
        if (ret != ACL_SUCCESS) {                                                                                   \
            std::cerr << "ACL_CHECK failed: " #expr << " returned " << ret << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                                 \
            std::exit(1);                                                                                           \
        }                                                                                                           \
    } while (0)

/*============================================================================
 * ACL Device Auto-Cleanup Guard
 *============================================================================*/

/**
 * @brief RAII guard that ensures ResetDevice+Finalize on any exit path,
 *        including std::exit() from ACL_CHECK failures.
 *
 * Usage: put "ACLDeviceGuard guard;" right after SetDevice succeeds.
 */
class ACLDeviceGuard {
public:
    ACLDeviceGuard(aclrtStream& stream) : stream_(&stream)
    {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        ACL_CHECK(aclrtCreateStream(stream_));
    }

    ~ACLDeviceGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(*stream_);
            stream_ = nullptr;
        }
        aclrtResetDevice(0);
        aclFinalize();
    }

private:
    aclrtStream* stream_;
};

/*============================================================================
 * File I/O Utilities
 *============================================================================*/

/**
 * @brief Read binary data from file into buffer.
 * @param path   File path to read from.
 * @param buffer Output buffer (caller-allocated, must be >= size bytes).
 * @param size   Number of bytes to read.
 * @return true on success, false on failure (prints error to stderr)
 */
inline bool ReadFile(const std::string& path, void* buffer, size_t size)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ReadFile failed: cannot open " << path << std::endl;
        return false;
    }
    ifs.read(static_cast<char*>(buffer), size);
    if (static_cast<size_t>(ifs.gcount()) != size) {
        std::cerr << "ReadFile failed: read " << ifs.gcount() << " bytes, expected " << size << " from " << path
                  << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Write binary data from buffer to file.
 * @param path   File path to write to.
 * @param buffer Input buffer.
 * @param size   Number of bytes to write.
 * @return true on success, false on failure (prints error to stderr)
 */
inline bool WriteFile(const std::string& path, const void* buffer, size_t size)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "WriteFile failed: cannot open " << path << std::endl;
        return false;
    }
    ofs.write(static_cast<const char*>(buffer), size);
    if (!ofs) {
        std::cerr << "WriteFile failed: write error on " << path << " (expected " << size << " bytes)" << std::endl;
        return false;
    }
    return true;
}

/*============================================================================
 * Math Utilities
 *============================================================================*/
template <typename T>
inline T CeilDiv(T a, T b)
{
    assert(b != 0);
    return (a + b - 1) / b;
}

template <typename T>
inline T CeilAlign(T v, T align)
{
    return CeilDiv(v, align) * align;
}

/*============================================================================
 * Platform Utilities
 *============================================================================*/

struct PlatformInfo {
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l0aSize{0};
    uint64_t l0bSize{0};
    uint64_t l0cSize{0};
    uint64_t l2Size{0};
    uint64_t btSize{0};
    platform_ascendc::SocVersion socVersion{0};
};

inline bool InitPlatformInfo(PlatformInfo& info)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        std::cerr << "[WARN] PlatformAscendCManager::GetInstance() returned null" << std::endl;
        return false;
    }

    info.aicNum = platform->GetCoreNumAic();
    info.aivNum = platform->GetCoreNumAiv();
    info.socVersion = platform->GetSocVersion();

    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0aSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0bSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0cSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, info.l2Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, info.btSize);

    return true;
}

inline int64_t GetAicCoreNum()
{
    PlatformInfo info;
    if (!InitPlatformInfo(info)) {
        std::cerr << "InitPlatformInfo() failed, cannot determine AIC core num" << std::endl;
        std::exit(1);
    }
    if (info.aicNum == 0) {
        std::cerr << "GetCoreNumAic() returned 0, cannot determine AIC core num" << std::endl;
        std::exit(1);
    }
    return static_cast<int64_t>(info.aicNum);
}

#endif /* DATA_UTILS_H */
