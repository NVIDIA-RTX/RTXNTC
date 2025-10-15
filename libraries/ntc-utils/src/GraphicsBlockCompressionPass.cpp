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

#include <ntc-utils/GraphicsBlockCompressionPass.h>
#include <libntc/shaders/Bindings.h>

bool GraphicsBlockCompressionPass::Init()
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
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(NTC_BINDING_BC_CONSTANT_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(NTC_BINDING_BC_INPUT_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::Texture_UAV(NTC_BINDING_BC_OUTPUT_TEXTURE));
    if (m_useAccelerationBuffer)
        bindingLayoutDesc.addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(NTC_BINDING_BC_ACCELERATION_BUFFER));

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);
    if (!m_bindingLayout)
        return false;
    
    return true;
}

bool GraphicsBlockCompressionPass::ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass,
    nvrhi::ITexture* inputTexture, nvrhi::Format inputFormat, int inputMipLevel,
    nvrhi::ITexture* outputTexture, int outputMipLevel, nvrhi::IBuffer* accelerationBuffer)
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
            .setDebugName("BlockCompressionConstants")
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(m_maxConstantBufferVersions);

        m_constantBuffer = m_device->createBuffer(constantBufferDesc);
        
        if (!m_constantBuffer)
            return false;
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(NTC_BINDING_BC_CONSTANT_BUFFER, m_constantBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(NTC_BINDING_BC_INPUT_TEXTURE, inputTexture, inputFormat)
            .setSubresources(nvrhi::TextureSubresourceSet().setBaseMipLevel(inputMipLevel)))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(NTC_BINDING_BC_OUTPUT_TEXTURE, outputTexture)
            .setSubresources(nvrhi::TextureSubresourceSet().setBaseMipLevel(outputMipLevel)));
    assert((accelerationBuffer != nullptr) == m_useAccelerationBuffer);
    if (accelerationBuffer)
        bindingSetDesc.addItem(nvrhi::BindingSetItem::RawBuffer_UAV(NTC_BINDING_BC_ACCELERATION_BUFFER, accelerationBuffer));

    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_bindingLayout);
    if (!bindingSet)
        return false;

    // Record the command list items
    commandList->writeBuffer(m_constantBuffer, computePass.constantBufferData, computePass.constantBufferSize);
    auto state = nvrhi::ComputeState()
        .setPipeline(pipeline)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(computePass.dispatchWidth, computePass.dispatchHeight);

    return true;
}