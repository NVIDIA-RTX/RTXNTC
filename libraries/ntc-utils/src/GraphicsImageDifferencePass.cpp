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

#include <ntc-utils/GraphicsImageDifferencePass.h>
#include <libntc/shaders/Bindings.h>

bool GraphicsImageDifferencePass::Init()
{
    // Create the binding layout
    nvrhi::VulkanBindingOffsets vulkanBindingOffsets;
    vulkanBindingOffsets
        .setConstantBufferOffset(0)
        .setSamplerOffset(0)
        .setShaderResourceOffset(0)
        .setUnorderedAccessViewOffset(0);

    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Compute)
        .setBindingOffsets(vulkanBindingOffsets)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(NTC_BINDING_IMAGE_DIFF_CONSTANT_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_A))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_B))
        .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(NTC_BINDING_IMAGE_DIFF_OUTPUT_BUFFER));

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);
    if (!m_bindingLayout)
        return false;

    // Create the results buffer
    auto resultBufferDesc = nvrhi::BufferDesc()
        .setByteSize(BytesPerQuery * m_maxQueries)
        .setDebugName("Compare Results")
        .setCanHaveRawViews(true)
        .setCanHaveUAVs(true)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
        .setKeepInitialState(true);

    m_outputBuffer = m_device->createBuffer(resultBufferDesc);
    if (!m_outputBuffer)
        return false;

    // Create the staging/readback buffer
    auto stagingBufferDesc = nvrhi::BufferDesc()
        .setByteSize(resultBufferDesc.byteSize)
        .setDebugName("Compare Staging")
        .setCpuAccess(nvrhi::CpuAccessMode::Read)
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);

    m_stagingBuffer = m_device->createBuffer(stagingBufferDesc);
    if (!m_stagingBuffer)
        return false;

    return true;
}

uint32_t GraphicsImageDifferencePass::GetOffsetForQuery(uint32_t queryIndex) const
{
    return queryIndex * BytesPerQuery;
}

bool GraphicsImageDifferencePass::ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass,
    nvrhi::ITexture* texture1, int mipLevel1, nvrhi::ITexture* texture2, int mipLevel2, uint32_t queryIndex)
{
    // Create the pipeline for this shader if it doesn't exist yet
    auto& pipeline = m_pipelines[computePass.computeShader];
    if (!pipeline)
    {
        nvrhi::ShaderHandle computeShader = m_device->createShader(nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute),
            computePass.computeShader, computePass.computeShaderSize);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc
            .setComputeShader(computeShader)
            .addBindingLayout(m_bindingLayout);

        pipeline = m_device->createComputePipeline(pipelineDesc);

        if (!pipeline)
            return false;
    }

    // Create the constant buffer if it doesn't exist yet or if it is too small (which shouldn't happen currently)
    if (!m_constantBuffer || m_constantBuffer->getDesc().byteSize < computePass.constantBufferSize)
    {
        nvrhi::BufferDesc constantBufferDesc;
        constantBufferDesc
            .setByteSize(computePass.constantBufferSize)
            .setDebugName("CompareImagesConstants")
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(m_maxQueries);

        m_constantBuffer = m_device->createBuffer(constantBufferDesc);
        
        if (!m_constantBuffer)
            return false;
    }

    // Create the binding set
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(NTC_BINDING_IMAGE_DIFF_CONSTANT_BUFFER, m_constantBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_A, texture1)
            .setSubresources(nvrhi::TextureSubresourceSet().setBaseMipLevel(mipLevel1)))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_B, texture2)
            .setSubresources(nvrhi::TextureSubresourceSet().setBaseMipLevel(mipLevel2)))
        .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(NTC_BINDING_IMAGE_DIFF_OUTPUT_BUFFER, m_outputBuffer));

    auto bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);

    if (!bindingSet)
        return false;

    static uint8_t QueryInitValue[BytesPerQuery] = { 0 };
    uint32_t const bufferOffset = GetOffsetForQuery(queryIndex);

    // Record the command list items
    commandList->writeBuffer(m_constantBuffer, computePass.constantBufferData, computePass.constantBufferSize);
    commandList->writeBuffer(m_outputBuffer, QueryInitValue, BytesPerQuery, bufferOffset);
    auto state = nvrhi::ComputeState()
        .setPipeline(pipeline)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(computePass.dispatchWidth, computePass.dispatchHeight);

    commandList->copyBuffer(m_stagingBuffer, bufferOffset, m_outputBuffer, bufferOffset, BytesPerQuery);

    m_resultsRead = false;

    return true;
}

bool GraphicsImageDifferencePass::ReadResults()
{
    uint64_t const* results = static_cast<uint64_t const*>(
        m_device->mapBuffer(m_stagingBuffer, nvrhi::CpuAccessMode::Read));

    if (!results)
        return false;

    for (int ch = 0; ch < m_maxQueries * ChannelsPerQuery; ++ch)
    {
        m_mseValues[ch] = float(ntc::DecodeImageDifferenceResult(results[ch]));
    }

    m_device->unmapBuffer(m_stagingBuffer);

    m_resultsRead = true;

    return true;
}

bool GraphicsImageDifferencePass::GetQueryResult(uint32_t queryIndex, float outPerChannelMSE[4],
    float* outOverallMSE, float* outOverallPSNR, int channels, float maxSignalValue)
{
    if (queryIndex >= m_maxQueries)
        return false;

    // Check that ReadResults() has been called after executing queries, to avoid returning garbage.
    // We can't really check if the comparison command list has been executed by the caller,
    // but this is better than nothing.
    if (!m_resultsRead)
        return false;

    float overallMSE = 0.f;
    for (int ch = 0; ch < channels; ++ch)
    {
        float const mse = m_mseValues[queryIndex * ChannelsPerQuery + ch];
        overallMSE += mse / float(channels);

        if (outPerChannelMSE)
            outPerChannelMSE[ch] = mse;
    }

    if (outOverallMSE)
        *outOverallMSE = overallMSE;

    if (outOverallPSNR)
        *outOverallPSNR = ntc::LossToPSNR(overallMSE / (maxSignalValue * maxSignalValue));

    return true;
}

