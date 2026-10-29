#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import numpy as np


_CLAMP_LOW_DEFAULT = np.float32(-2147483648.0)
_CLAMP_HIGH_DEFAULT = np.float32(2147483648.0)


def confidence_head_compute(
    hidden,
    weight,
    markov_emb=None,
    temperature=None,
    bias=None,
    *,
    clamp_low=None,
    clamp_high=None,
):
    """Concatenate features, project in FP32 and apply the vector epilogue."""
    hidden = hidden.astype(np.float32)
    weight = weight.astype(np.float32)
    if markov_emb is not None:
        features = np.concatenate((hidden, markov_emb.astype(np.float32)), axis=-1)
    else:
        features = hidden

    value = np.float32(1.0) if temperature is None else np.float32(temperature[0])
    if value == 0:
        value = np.float32(1.0)

    # A zero K still executes bias, temperature, clamp and sigmoid.
    if features.shape[-1] == 0:
        logits = np.zeros(hidden.shape[:-1], dtype=np.float32)
    else:
        logits = np.matmul(features, weight)
    if bias is not None:
        # bias[N] broadcasts over the batch dimension of hidden[B, N, K].
        logits = logits + bias.astype(np.float32)
    scaled = logits * (np.float32(-1.0) / value)
    clipped = scaled
    # FP32(+/-INT32_MAX) is reserved for an omitted bound.
    low = _CLAMP_LOW_DEFAULT if clamp_low is None else np.float32(clamp_low)
    high = _CLAMP_HIGH_DEFAULT if clamp_high is None else np.float32(clamp_high)
    if low != _CLAMP_LOW_DEFAULT:
        clipped = np.maximum(clipped, low)
    if high != _CLAMP_HIGH_DEFAULT:
        clipped = np.minimum(clipped, high)
    with np.errstate(over="ignore"):
        out = np.float32(1.0) / (np.float32(1.0) + np.exp(clipped))
    return [out.astype(np.float32)]
