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

#pragma once

#include <memory>
#include <nvrhi/nvrhi.h>
#if NTC_WITH_DX12
#include <dstorage.h>
#endif

namespace donut::app
{
    struct DeviceCreationParameters;
}

struct GDeflateFeatures
{
#if NTC_WITH_DX12
    nvrhi::RefCountPtr<IDStorageQueue2> dstorageQueue;
    HANDLE dstorageEvent = NULL;
#endif
    bool gpuDecompressionSupported = false;

    ~GDeflateFeatures();
};

void SetNtcGraphicsDeviceParameters(
    donut::app::DeviceCreationParameters& deviceParams,
    nvrhi::GraphicsAPI graphicsApi,
    bool enableSharedMemory,
    bool enableDX12ExperimentalFeatures,
    char const* windowTitle);

bool IsDX12DeveloperModeEnabled();

std::unique_ptr<GDeflateFeatures> InitGDeflate(nvrhi::IDevice* device, bool debugMode);