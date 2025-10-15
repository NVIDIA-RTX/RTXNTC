/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
 
// Include the constants header unconditionally so that NTC_NETWORK_UNKNOWN is always defined
#include "libntc/shaders/InferenceConstants.h"

#define USE_COOPVEC

#include "libntc/shaders/InferenceCoopVec.hlsli"

#include "NtcForwardShadingPass.hlsl"
