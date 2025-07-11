/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvrhi/nvrhi.h>
#include <libntc/ntc.h>
#include <unordered_map>
#include <donut/engine/BindingCache.h>

class GraphicsBlockCompressionPass
{
public:
    GraphicsBlockCompressionPass(nvrhi::IDevice* device, bool useAccelerationBuffer, int maxConstantBufferVersions = 1)
        : m_device(device)
        , m_useAccelerationBuffer(useAccelerationBuffer)
        , m_maxConstantBufferVersions(maxConstantBufferVersions)
        , m_bindingCache(device)
    { }

    bool Init();

    // Note: ExecuteComputePass expects that the commandList is open, and leaves it open.
    // The output buffer must be large enough and have the canHaveUAVs and canHaveRawViews flags set.
    bool ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass,
        nvrhi::ITexture* inputTexture, nvrhi::Format inputFormat, int inputMipLevel,
        nvrhi::ITexture* outputTexture, int outputMipLevel, nvrhi::IBuffer* accelerationBuffer);

    void ClearBindingSetCache() { m_bindingCache.Clear(); }

private:
    nvrhi::DeviceHandle m_device;
    std::unordered_map<const void*, nvrhi::ComputePipelineHandle> m_pipelines; // shader bytecode -> pipeline
    nvrhi::BindingLayoutHandle m_bindingLayout;
    donut::engine::BindingCache m_bindingCache;
    nvrhi::BufferHandle m_constantBuffer;
    bool m_useAccelerationBuffer;
    int m_maxConstantBufferVersions;
};
