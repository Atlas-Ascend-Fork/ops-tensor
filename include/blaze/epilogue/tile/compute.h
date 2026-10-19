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
 * \file compute.h
 * \brief Architecture dispatcher for Epilogue Tile compute operations.
 */
#pragma once

#if defined(__NPU_ARCH__)
#include "blaze/epilogue/tile/arch35/dequant.h"
#include "blaze/epilogue/tile/arch35/gelu.h"
#include "blaze/epilogue/tile/arch35/initialize_empty_softmax.h"
#include "blaze/epilogue/tile/arch35/mx_quant.h"
#include "blaze/epilogue/tile/arch35/reduce_square.h"
#include "blaze/epilogue/tile/arch35/rms_softmax.h"
#endif
