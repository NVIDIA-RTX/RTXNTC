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

class GraphicsDecompressionPass
{
public:
    GraphicsDecompressionPass(nvrhi::IDevice* device, int descriptorTableSize)
        : m_device(device)
        , m_descriptorTableSize(descriptorTableSize)
        , m_bindingCache(device)
    { }

    bool Init();

    void WriteDescriptor(nvrhi::BindingSetItem item);

    bool SetLatentDataFromTextureSet(nvrhi::ICommandList* commandList, ntc::IStream* inputStream,
        ntc::ITextureSetMetadata* textureSetMetadata);

    void SetLatentTexture(nvrhi::ITexture* texture);

    bool SetWeightsFromTextureSet(nvrhi::ICommandList* commandList, ntc::ITextureSetMetadata* textureSetMetadata,
        ntc::InferenceWeightType weightType);

    void SetWeightBuffer(nvrhi::IBuffer* buffer);

    bool ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass);

    void ClearBindingSetCache() { m_bindingCache.Clear(); }

private:
    nvrhi::DeviceHandle m_device;
    int m_descriptorTableSize;
    std::unordered_map<const void*, nvrhi::ComputePipelineHandle> m_pipelines; // shader bytecode -> pipeline
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingLayoutHandle m_bindlessLayout;
    donut::engine::BindingCache m_bindingCache;
    nvrhi::DescriptorTableHandle m_descriptorTable;
    nvrhi::TextureHandle m_latentTexture;
    nvrhi::BufferHandle m_weightUploadBuffer;
    nvrhi::BufferHandle m_weightBuffer;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::SamplerHandle m_latentSampler;
    bool m_latentTextureIsExternal = false;
    bool m_weightBufferIsExternal = false;
};
