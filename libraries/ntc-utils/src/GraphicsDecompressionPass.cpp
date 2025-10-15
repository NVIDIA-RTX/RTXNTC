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

#include <ntc-utils/GraphicsDecompressionPass.h>
#include <libntc/shaders/Bindings.h>

bool GraphicsDecompressionPass::Init()
{
    // Make sure the binding layout exists
    if (!m_bindingLayout)
    {
        nvrhi::VulkanBindingOffsets vulkanBindingOffsets;
        vulkanBindingOffsets
            .setConstantBufferOffset(0)
            .setSamplerOffset(0)
            .setShaderResourceOffset(0)
            .setUnorderedAccessViewOffset(0);

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc
            .setVisibility(nvrhi::ShaderType::Compute)
            .setBindingOffsets(vulkanBindingOffsets)
            .setRegisterSpaceAndDescriptorSet(NTC_BINDING_DECOMPRESSION_INPUT_SPACE)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(NTC_BINDING_DECOMPRESSION_CONSTANT_BUFFER))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(NTC_BINDING_DECOMPRESSION_LATENT_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(NTC_BINDING_DECOMPRESSION_WEIGHT_BUFFER))
            .addItem(nvrhi::BindingLayoutItem::Sampler(NTC_BINDING_DECOMPRESSION_LATENT_SAMPLER));

        m_bindingLayout = m_device->createBindingLayout(layoutDesc);

        if (!m_bindingLayout)
            return false;
    }

    // Make sure the bindless layout exists
    if (!m_bindlessLayout)
    {
        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc
            .setVisibility(nvrhi::ShaderType::Compute)
            .setMaxCapacity(m_descriptorTableSize)
            .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_UAV(NTC_BINDING_DECOMPRESSION_OUTPUT_SPACE));

        m_bindlessLayout = m_device->createBindlessLayout(bindlessLayoutDesc);

        if (!m_bindlessLayout)
            return false;
    }

    // Make sure the descriptor table exists
    if (!m_descriptorTable)
    {
        m_descriptorTable = m_device->createDescriptorTable(m_bindlessLayout);
        if (!m_descriptorTable)
            return false;

        m_device->resizeDescriptorTable(m_descriptorTable, m_descriptorTableSize, false);
    }

    if (!m_latentSampler)
    {
        nvrhi::SamplerDesc samplerDesc = nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setMagFilter(true)
            .setMinFilter(true)
            .setMipFilter(false);

        m_latentSampler = m_device->createSampler(samplerDesc);
    }

    return true;
}

void GraphicsDecompressionPass::WriteDescriptor(nvrhi::BindingSetItem item)
{
    m_device->writeDescriptorTable(m_descriptorTable, item);
}

static bool IsLatentTextureCompatible(nvrhi::TextureDesc const& a, nvrhi::TextureDesc const& b)
{
    return a.dimension == b.dimension
        && a.format == b.format
        && a.width == b.width
        && a.height == b.height
        && a.arraySize == b.arraySize
        && a.mipLevels == b.mipLevels;
}

bool GraphicsDecompressionPass::SetLatentDataFromTextureSet(nvrhi::ICommandList* commandList, ntc::IStream* inputStream,
    ntc::ITextureSetMetadata* textureSetMetadata)
{
    ntc::LatentTextureDesc const latentTextureDescSrc = textureSetMetadata->GetLatentTextureDesc();

    nvrhi::TextureDesc latentTextureDesc = nvrhi::TextureDesc()
        .setDebugName("Latent Texture")
        .setDimension(nvrhi::TextureDimension::Texture2DArray)
        .setFormat(nvrhi::Format::BGRA4_UNORM)
        .setWidth(latentTextureDescSrc.width)
        .setHeight(latentTextureDescSrc.height)
        .setArraySize(latentTextureDescSrc.arraySize)
        .setMipLevels(latentTextureDescSrc.mipLevels)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true);

    if (!m_latentTexture || m_latentTextureIsExternal ||
        !IsLatentTextureCompatible(m_latentTexture->getDesc(), latentTextureDesc))
    {
        m_latentTexture = m_device->createTexture(latentTextureDesc);
        m_latentTextureIsExternal = false;

        if (!m_latentTexture)
            return false;
    }

    std::vector<uint8_t> latentsBuffer;
    for (int mipLevel = 0; mipLevel < latentTextureDescSrc.mipLevels; ++mipLevel)
    {
        ntc::LatentTextureFootprint footprint;
        ntc::Status ntcStatus = textureSetMetadata->GetLatentTextureFootprint(mipLevel, footprint);
        if (ntcStatus != ntc::Status::Ok)
            return false;

        if (!inputStream->Seek(footprint.range.offset))
            return false;

        latentsBuffer.resize(footprint.range.size);
            
        if (!inputStream->Read(latentsBuffer.data(), latentsBuffer.size()))
            return false;

        uint8_t const* srcData = latentsBuffer.data();

        for (int layerIndex = 0; layerIndex < latentTextureDescSrc.arraySize; ++layerIndex)
        {
            commandList->writeTexture(m_latentTexture, layerIndex, mipLevel,
                srcData, footprint.rowPitch);

            srcData += footprint.slicePitch;
        }
    }


    return true;
}

void GraphicsDecompressionPass::SetLatentTexture(nvrhi::ITexture* texture)
{
    if (texture == m_latentTexture)
        return;

    m_latentTexture = texture;
    m_latentTextureIsExternal = true; // Prevent the texture from being overwritten by a subsequent call to SetInputData
}

bool GraphicsDecompressionPass::SetWeightsFromTextureSet(nvrhi::ICommandList* commandList,
    ntc::ITextureSetMetadata* textureSetMetadata, ntc::InferenceWeightType weightType)
{
    void const* uploadData = nullptr;
    size_t uploadSize = 0;
    size_t convertedSize = 0;
    textureSetMetadata->GetInferenceWeights(weightType, &uploadData, &uploadSize, &convertedSize);

    bool const uploadBufferNeeded = convertedSize != 0;

    // Create the weight upload buffer if it doesn't exist yet or if it is too small
    if (!m_weightUploadBuffer && uploadBufferNeeded ||
        m_weightUploadBuffer && m_weightUploadBuffer->getDesc().byteSize < uploadSize)
    {
        nvrhi::BufferDesc uploadBufferDesc;
        uploadBufferDesc
            .setByteSize(uploadSize)
            .setDebugName("DecompressionWeightsUpload")
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true);

        m_weightUploadBuffer = m_device->createBuffer(uploadBufferDesc);

        if (!m_weightUploadBuffer)
            return false;
    }

    size_t finalWeightBufferSize = convertedSize ? convertedSize : uploadSize;

    // Create the weight buffer if it doesn't exist yet or if it is too small
    if (!m_weightBuffer || m_weightBufferIsExternal || m_weightBuffer->getDesc().byteSize < finalWeightBufferSize)
    {
        nvrhi::BufferDesc weightBufferDesc;
        weightBufferDesc
            .setByteSize(finalWeightBufferSize)
            .setDebugName("DecompressionWeights")
            .setCanHaveRawViews(true)
            .setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        m_weightBuffer = m_device->createBuffer(weightBufferDesc);
        m_weightBufferIsExternal = false;

        if (!m_weightBuffer)
            return false;
    }

    if (uploadBufferNeeded)
    {
        // Write the weight upload buffer
        commandList->writeBuffer(m_weightUploadBuffer, uploadData, uploadSize);

        // Place the barriers before layout conversion - which happens in LibNTC and bypasses NVRHI
        commandList->setBufferState(m_weightUploadBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->setBufferState(m_weightBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();

        // Unwrap the command list and buffer objects from NVRHI
        bool const isVulkan = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN;
        nvrhi::ObjectType const commandListType = isVulkan
            ? nvrhi::ObjectTypes::VK_CommandBuffer
            : nvrhi::ObjectTypes::D3D12_GraphicsCommandList;
        nvrhi::ObjectType const bufferType = isVulkan
            ? nvrhi::ObjectTypes::VK_Buffer
            : nvrhi::ObjectTypes::D3D12_Resource;

        void* nativeCommandList = commandList->getNativeObject(commandListType);
        void* nativeSrcBuffer = m_weightUploadBuffer->getNativeObject(bufferType);
        void* nativeDstBuffer = m_weightBuffer->getNativeObject(bufferType);

        // Convert the weight layout to CoopVec
        textureSetMetadata->ConvertInferenceWeights(weightType, nativeCommandList,
            nativeSrcBuffer, 0, nativeDstBuffer, 0);
    }
    else
    {
        // Write the weight buffer directly
        commandList->writeBuffer(m_weightBuffer, uploadData, uploadSize);
    }

    return true;
}

void GraphicsDecompressionPass::SetWeightBuffer(nvrhi::IBuffer* buffer)
{
    if (buffer == m_weightBuffer)
        return;
        
    m_weightBuffer = buffer;
    m_weightBufferIsExternal = true; // Prevent the buffer from being overwritten by a subsequent call to SetWeightsFromTextureSet
}

bool GraphicsDecompressionPass::ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass)
{
    // Create the pipeline for this shader if it doesn't exist yet
    auto& pipeline = m_pipelines[computePass.computeShader];
    if (!pipeline)
    {
        nvrhi::ShaderHandle computeShader = m_device->createShader(nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute),
                                                                   computePass.computeShader,
                                                                   computePass.computeShaderSize);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc
            .setComputeShader(computeShader)
            .addBindingLayout(m_bindingLayout)
            .addBindingLayout(m_bindlessLayout);

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
            .setDebugName("DecompressionConstants")
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(NTC_MAX_MIPS * NTC_MAX_CHANNELS);

        m_constantBuffer = m_device->createBuffer(constantBufferDesc);

        if (!m_constantBuffer)
            return false;
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(NTC_BINDING_DECOMPRESSION_CONSTANT_BUFFER, m_constantBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(NTC_BINDING_DECOMPRESSION_LATENT_TEXTURE, m_latentTexture))
        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(NTC_BINDING_DECOMPRESSION_WEIGHT_BUFFER, m_weightBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(NTC_BINDING_DECOMPRESSION_LATENT_SAMPLER, m_latentSampler));
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_bindingLayout);
    if (!bindingSet)
        return false;

    // Write the constant buffer
    commandList->writeBuffer(m_constantBuffer, computePass.constantBufferData, computePass.constantBufferSize);

    // Execute the compute shader for decompression
    nvrhi::ComputeState state;
    state.setPipeline(pipeline)
         .addBindingSet(bindingSet)
         .addBindingSet(m_descriptorTable);
    commandList->setComputeState(state);
    commandList->dispatch(computePass.dispatchWidth, computePass.dispatchHeight);

    return true;
}