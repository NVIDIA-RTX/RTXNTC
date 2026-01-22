/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "GraphicsPasses.h"
#include "Utils.h"
#include <ntc-utils/BufferLoading.h>
#include <ntc-utils/DeviceUtils.h>
#include <ntc-utils/GraphicsDecompressionPass.h>
#include <ntc-utils/GraphicsImageDifferencePass.h>
#include <ntc-utils/GraphicsBlockCompressionPass.h>
#include <ntc-utils/Manifest.h>
#include <tinyexr.h>
#include <filesystem>
#include <donut/core/log.h>
#include <numeric>
#include <memory>
#include <mutex>
namespace fs = std::filesystem;

bool CreateGraphicsResourcesFromMetadata(
    ntc::IContext* context,
    nvrhi::IDevice* device,
    ntc::ITextureSetMetadata* metadata,
    int mipLevels,
    bool enableCudaSharing,
    GraphicsResourcesForTextureSet& resources)
{
    int const maxImageDimension = 16384;
    ntc::TextureSetDesc const& textureSetDesc = metadata->GetDesc();
    if (textureSetDesc.width > maxImageDimension || textureSetDesc.height > maxImageDimension)
    {
        donut::log::error("Cannot perform any graphics API based processing on the texture set because it is too large. "
            "The texture set is %dx%d pixels, and maximum supported size is %dx%d.",
            textureSetDesc.width, textureSetDesc.height, maxImageDimension, maxImageDimension);
        return false;
    }

    int const numTextures = metadata->GetTextureCount();
    
    for (int i = 0; i < numTextures; ++i)
    {
        ntc::ITextureMetadata* textureMetadata = metadata->GetTexture(i);
        assert(textureMetadata);

        char const* name = textureMetadata->GetName();
        int firstChannel, numChannels;
        textureMetadata->GetChannels(firstChannel, numChannels);
        ntc::ChannelFormat const channelFormat = textureMetadata->GetChannelFormat();
        ntc::BlockCompressedFormat const bcFormat = textureMetadata->GetBlockCompressedFormat();

        nvrhi::Format colorFormat = nvrhi::Format::UNKNOWN;
        ntc::ChannelFormat sharedFormat = ntc::ChannelFormat::UNKNOWN;
        switch (channelFormat)
        {
            case ntc::ChannelFormat::UNORM8:
                colorFormat = nvrhi::Format::RGBA8_UNORM;
                sharedFormat = ntc::ChannelFormat::UNORM8;
                break;
            case ntc::ChannelFormat::UNORM16:
                // Note: graphics passes don't support saving 16-bit PNGs at this time, so cast to u8
                colorFormat = nvrhi::Format::RGBA8_UNORM;
                sharedFormat = ntc::ChannelFormat::UNORM8;
                break;
            case ntc::ChannelFormat::FLOAT16:
            case ntc::ChannelFormat::FLOAT32:
                colorFormat = nvrhi::Format::RGBA32_FLOAT;
                sharedFormat = ntc::ChannelFormat::FLOAT32;
                break;
            case ntc::ChannelFormat::UINT32:
                colorFormat = nvrhi::Format::R32_UINT;
                sharedFormat = ntc::ChannelFormat::UINT32;
                break;
        }

        GraphicsResourcesForTexture textureResources(context);
        textureResources.name = name;

        auto colorTextureDesc = nvrhi::TextureDesc()
            .setDebugName(name)
            .setFormat(colorFormat)
            .setWidth(textureSetDesc.width)
            .setHeight(textureSetDesc.height)
            .setMipLevels(mipLevels)
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setIsUAV(true)
            .setSharedResourceFlags(enableCudaSharing ? nvrhi::SharedResourceFlags::Shared : nvrhi::SharedResourceFlags::None)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true);

        textureResources.color = device->createTexture(colorTextureDesc);
        if (!textureResources.color)
            return false;

        colorTextureDesc
            .setInitialState(nvrhi::ResourceStates::CopyDest);

        textureResources.stagingColor = device->createStagingTexture(colorTextureDesc, nvrhi::CpuAccessMode::Read);
        if (!textureResources.stagingColor)
            return false;

        if (enableCudaSharing)
        {
            ntc::SharedTextureDesc sharedTextureDesc;
            sharedTextureDesc.width = colorTextureDesc.width;
            sharedTextureDesc.height = colorTextureDesc.height;
            sharedTextureDesc.channels = 4;
            sharedTextureDesc.mips = colorTextureDesc.mipLevels;
            sharedTextureDesc.format = sharedFormat;
            sharedTextureDesc.dedicatedResource = true;
        #ifdef _WIN32
            sharedTextureDesc.handleType = device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN
                ? ntc::SharedHandleType::OpaqueWin32
                : ntc::SharedHandleType::D3D12Resource;
        #else
            sharedTextureDesc.handleType = ntc::SharedHandleType::OpaqueFd;
        #endif
            sharedTextureDesc.sizeInBytes = device->getTextureMemoryRequirements(textureResources.color).size;
            sharedTextureDesc.sharedHandle = textureResources.color->getNativeObject(nvrhi::ObjectTypes::SharedHandle).integer;
            
            ntc::Status ntcStatus = context->RegisterSharedTexture(sharedTextureDesc, textureResources.sharedTexture.ptr());
            if (ntcStatus != ntc::Status::Ok)
            {
                fprintf(stderr, "Failed to register a shared texture with NTC, code = %s: %s\n", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                return false;
            }
        }
        
        if (bcFormat != ntc::BlockCompressedFormat::None)
        {
            BcFormatDefinition const* bcFormatDef = GetBcFormatDefinition(bcFormat);

            int const widthBlocks = (metadata->GetDesc().width + 3) / 4;
            int const heightBlocks = (metadata->GetDesc().height + 3) / 4;
            auto blockTextureDesc = nvrhi::TextureDesc()
                .setDebugName(name)
                .setFormat(bcFormatDef->bytesPerBlock == 8 ? nvrhi::Format::RG32_UINT : nvrhi::Format::RGBA32_UINT)
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth(widthBlocks)
                .setHeight(heightBlocks)
                .setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true);

            textureResources.blocks = device->createTexture(blockTextureDesc);
            if (!textureResources.blocks)
                return false;

            blockTextureDesc
                .setInitialState(nvrhi::ResourceStates::CopyDest);

            textureResources.stagingBlocks = device->createStagingTexture(blockTextureDesc, nvrhi::CpuAccessMode::Read);
            if (!textureResources.stagingBlocks)
                return false;

            auto bcTextureDesc = nvrhi::TextureDesc()
                .setDebugName(name)
                .setFormat(bcFormatDef->nvrhiFormat)
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth(metadata->GetDesc().width)
                .setHeight(metadata->GetDesc().height)
                .setMipLevels(mipLevels)
                .setInitialState(nvrhi::ResourceStates::CopyDest)
                .setKeepInitialState(true);

            textureResources.bc = device->createTexture(bcTextureDesc);
            if (!textureResources.bc)
                return false;
        }

        resources.perTexture.push_back(std::move(textureResources));
    }

    return true;
}

bool DecompressTextureSetWithGraphicsAPI(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    nvrhi::ITimerQuery* timerQuery,
    GraphicsDecompressionPass& gdp,
    GDeflateFeatures* gdeflateFeatures,
    ntc::IContext* context,
    ntc::ITextureSetMetadata* metadata,
    ntc::IStream* inputFile,
    int mipLevels,
    bool enableDithering,
    GraphicsResourcesForTextureSet const& graphicsResources)
{
    // In some cases, this function is called without a file - which means we reuse the previously uploaded data.
    if (inputFile)
    {
        if (!gdp.SetLatentDataFromTextureSet(commandList, context, gdeflateFeatures, inputFile, metadata))
        {
            fprintf(stderr, "GraphicsDecompressionPass::SetInputData failed.\n");
            return false;
        }
    }

    int const numTextures = int(graphicsResources.perTexture.size());
    
    // Write UAV descriptors for all necessary mip levels into the descriptor table
    for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        for (int index = 0; index < numTextures; ++index)
        {
            const auto bindingSetItem = nvrhi::BindingSetItem::Texture_UAV(
                mipLevel * numTextures + index,
                graphicsResources.perTexture[index].color,
                nvrhi::Format::UNKNOWN,
                nvrhi::TextureSubresourceSet(mipLevel, 1, 0, 1));

            gdp.WriteDescriptor(bindingSetItem);
        }
    }

    ntc::InferenceWeightType weightType = metadata->GetBestSupportedWeightType();
    if (weightType == ntc::InferenceWeightType::Unknown)
    {
        fprintf(stderr, "The texture set does not provide any weights compatible with the current device.\n");
        return false;
    }

    commandList->open();

    if (!gdp.SetWeightsFromTextureSet(commandList, metadata, weightType))
    {
        fprintf(stderr, "GraphicsDecompressionPass::SetWeightsFromTextureSet failed.\n");
        commandList->close();
        return false;
    }

    commandList->beginTimerQuery(timerQuery);

    // Decompress each mip level in a loop
    for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        std::vector<ntc::OutputTextureDesc> outputs;
        for (int textureIndex = 0; textureIndex < numTextures; ++textureIndex)
        {
            ntc::ITextureMetadata const* textureMetadata = metadata->GetTexture(textureIndex);
            ntc::OutputTextureDesc& outputDesc = outputs.emplace_back();

            outputDesc.descriptorIndex = textureIndex;

            textureMetadata->GetChannels(outputDesc.firstChannel, outputDesc.numChannels);
            outputDesc.rgbColorSpace = textureMetadata->GetRgbColorSpace();
            outputDesc.alphaColorSpace = textureMetadata->GetAlphaColorSpace();

            if (textureMetadata->GetChannelFormat() == ntc::ChannelFormat::UNORM8)
            {
                if (enableDithering)
                {
                    outputDesc.ditherScale = 1.f / 255.f;
                }
                else
                {
                    // Assume that if the user has requested no dithering, they also want the output
                    // most closely matching the original image, so enable explicit quantization.
                    outputDesc.quantizationScale = 1.f / 255.f;
                }
            }
        }

        // Obtain the compute pass description and constant buffer data from NTC
        ntc::MakeDecompressionComputePassParameters params;
        params.textureSetMetadata = metadata;
        params.mipLevel = mipLevel;
        params.firstOutputDescriptorIndex = mipLevel * numTextures;
        params.weightType = weightType;
        params.numOutputTextures = numTextures;
        params.pOutputTextures = outputs.data();
        ntc::ComputePassDesc computePass{};
        ntc::Status ntcStatus = context->MakeDecompressionComputePass(params, &computePass);
        CHECK_NTC_RESULT("MakeDecompressionComputePass");

        if (!gdp.ExecuteComputePass(commandList, computePass))
        {
            fprintf(stderr, "GraphicsDecompressionPass::ExecuteComputePass failed.\n");
            commandList->close();
            return false;
        }
    }

    commandList->endTimerQuery(timerQuery);

    // Copy the decompressed textures into staging resources
    for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        for (int index = 0; index < numTextures; ++index)
        {
            auto const slice = nvrhi::TextureSlice().setMipLevel(mipLevel);
            commandList->copyTexture(graphicsResources.perTexture[index].stagingColor, slice,
                graphicsResources.perTexture[index].color, slice);
        }
    }

    commandList->close();
    device->executeCommandList(commandList);
    
    return true;
}

bool CopyTextureSetDataIntoGraphicsTextures(
    ntc::IContext* context,
    ntc::ITextureSet* textureSet,
    ntc::TextureDataPage page,
    bool allMipLevels,
    bool onlyBlockCompressedFormats,
    GraphicsResourcesForTextureSet const& graphicsResources)
{
    for (int textureIndex = 0; textureIndex < textureSet->GetTextureCount(); ++textureIndex)
    {
        ntc::ITextureMetadata* textureMetadata = textureSet->GetTexture(textureIndex);
        ntc::BlockCompressedFormat const bcFormat = textureMetadata->GetBlockCompressedFormat();
        if (onlyBlockCompressedFormats && bcFormat == ntc::BlockCompressedFormat::None)
            continue;

        GraphicsResourcesForTexture const& textureResources = graphicsResources.perTexture[textureIndex];

        int mipLevels = allMipLevels ? textureResources.color->getDesc().mipLevels : 1;

        for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
        {
            ntc::ReadChannelsIntoTextureParameters params;
            params.page = page;
            params.mipLevel = mipLevel;
            params.firstChannel = textureMetadata->GetFirstChannel();
            params.numChannels = textureMetadata->GetNumChannels();
            params.texture = textureResources.sharedTexture;
            params.textureMipLevel = mipLevel;
            params.dstRgbColorSpace = textureMetadata->GetRgbColorSpace();
            params.dstAlphaColorSpace = textureMetadata->GetAlphaColorSpace();
            params.useDithering = true;

            ntc::Status ntcStatus = textureSet->ReadChannelsIntoTexture(params);

            CHECK_NTC_RESULT("ReadChannelsIntoTexture")
        }
    }

    return true;
}

bool SaveGraphicsStagingTextures(
    ntc::ITextureSetMetadata* metadata,
    nvrhi::IDevice* device,
    char const* savePath,
    ImageContainer const userProvidedContainer,
    bool saveMips,
    GraphicsResourcesForTextureSet const& graphicsResources)
{
    fs::path const outputPath = fs::path(savePath);
    bool mipsDirCreated = false;

    std::mutex mutex;
    bool anyErrors = false;

    for (int index = 0; index < int(graphicsResources.perTexture.size()); ++index)
    {
        ntc::ITextureMetadata* textureMetadata = metadata->GetTexture(index);
        ntc::BlockCompressedFormat bcFormat = textureMetadata->GetBlockCompressedFormat();

        if (bcFormat != ntc::BlockCompressedFormat::None)
            continue;

        if (!mipsDirCreated && saveMips && metadata->GetDesc().mips > 1)
        {
            fs::path mipsPath = outputPath / "mips";
            if (!fs::is_directory(mipsPath) && !fs::create_directories(mipsPath))
            {
                fprintf(stderr, "Failed to create directory '%s'.\n", mipsPath.generic_string().c_str());
                return false;
            }
            mipsDirCreated = true;
        }
        
        GraphicsResourcesForTexture const& textureResources = graphicsResources.perTexture[index];
        nvrhi::TextureDesc const& textureDesc = textureResources.stagingColor->getDesc();

        ImageContainer container = userProvidedContainer;

        // The textures have been created long before, we can only read them as they are at this point...
        // Float32 data means we'll save as EXR.
        // TODO: implement full conversion support.
        if (textureDesc.format == nvrhi::Format::RGBA32_FLOAT)
        {
            if (container != ImageContainer::EXR && container != ImageContainer::Auto)
            {
                printf("Warning: Cannot save texture '%s' as %s in this mode, using EXR instead.\n",
                    textureResources.name.c_str(), GetContainerExtension(container));
            }

            container = ImageContainer::EXR;
        }
        else if (container == ImageContainer::EXR)
        {
            printf("Warning: Cannot save texture '%s' as EXR in this mode, using BMP instead.\n",
                textureResources.name.c_str());

            container = ImageContainer::BMP;
        }

        // Use PNG as the default container for non-float data
        if (container == ImageContainer::Auto)
            container = ImageContainer::PNG;

        // Fallback from PNG16 to regular PNG, 16-bit support not implemented here
        if (container == ImageContainer::PNG16)
        {
            printf("Warning: Cannot save texture '%s' as PNG16 in this mode, using regular PNG instead.\n",
                textureResources.name.c_str());
            
            container = ImageContainer::PNG;
        }
        
        for (int mipLevel = 0; mipLevel < int(textureDesc.mipLevels); ++mipLevel)
        {
            auto const slice = nvrhi::TextureSlice().setMipLevel(mipLevel);
            size_t rowPitch = 0;
            uint8_t* mappedTexture = static_cast<uint8_t*>(device->mapStagingTexture(textureResources.stagingColor,
                slice, nvrhi::CpuAccessMode::Read, &rowPitch));
            if (!mappedTexture)
            {
                fprintf(stderr, "Failed to map texture '%s' mip level %d.\n", textureResources.name.c_str(), mipLevel);
                return false;
            }

            uint32_t const mipWidth = std::max(textureDesc.width >> mipLevel, 1u);
            uint32_t const mipHeight = std::max(textureDesc.height >> mipLevel, 1u);

            // Copy the pixel data into a CPU buffer without row padding (rowPitch = bpp * width),
            // because that's what SaveImageToContainer expects.
            size_t const bytesPerPixel = nvrhi::getFormatInfo(textureDesc.format).bytesPerBlock;
            size_t dstRowPitch = bytesPerPixel * mipWidth;

            std::shared_ptr<uint8_t> textureData = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(malloc(dstRowPitch * mipHeight)));

            for (uint32_t row = 0; row < mipHeight; ++row)
            {
                memcpy(textureData.get() + dstRowPitch * row, mappedTexture + rowPitch * row, dstRowPitch);
            }

            device->unmapStagingTexture(textureResources.stagingColor);

            std::string outputFileName;
            if (saveMips && mipLevel > 0)
            {
                outputFileName = (outputPath / "mips" / textureResources.name).generic_string();

                char mipStr[8];
                snprintf(mipStr, sizeof(mipStr), ".%02d", mipLevel);
                outputFileName += mipStr;
            }
            else
            {
                outputFileName = (outputPath / textureResources.name).generic_string();
            }
                        
            outputFileName += GetContainerExtension(container);

            StartAsyncTask([&anyErrors, &mutex, container, outputFileName, textureData, textureDesc, mipWidth, mipHeight]()
            {
                int const numChannels = 4; // Lower channel counts not currently supported

                bool success = SaveImageToContainer(container, textureData.get(), mipWidth, mipHeight,
                    numChannels, outputFileName.c_str());

                auto lockGuard = std::lock_guard(mutex);

                if (!success)
                {
                    fprintf(stderr, "Failed to write a texture into '%s'.\n", outputFileName.c_str());
                    anyErrors = true;
                }
                else
                {
                    printf("Saved image '%s': %dx%d pixels, %d channels.\n", outputFileName.c_str(),
                        mipWidth, mipHeight, numChannels);
                }
            });
        }
    }

    WaitForAllTasks();
    if (anyErrors)
        return false;

    return true;
}

void CopyBlocksIntoBCTexture(
    nvrhi::ICommandList* commandList,
    GraphicsResourcesForTexture const& textureResources,
    uint32_t width,
    uint32_t height)
{
    int const widthBlocks = (width + 3) / 4;
    int const heightBlocks = (height + 3) / 4;

    auto srcSlice = nvrhi::TextureSlice().setWidth(widthBlocks).setHeight(heightBlocks);
    auto dstSlice = nvrhi::TextureSlice().setWidth(width).setHeight(height);
    commandList->copyTexture(textureResources.bc, dstSlice, textureResources.blocks, srcSlice);
}

bool ComputeBlockCompressedImageError(
    ntc::IContext* context,
    GraphicsImageDifferencePass& compareImagesPass,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    GraphicsResourcesForTexture const& textureResources,
    uint32_t width,
    uint32_t height,
    int mipLevel,
    bool reuseCompressedData,
    bool useAlphaThreshold,
    float alphaThreshold,
    bool useMSLE,
    float* outOverallMSE,
    float* outOverallPSNR,
    int channels = 4)
{
    // Obtain the pass descriptor from NTC
    ntc::MakeImageDifferenceComputePassParameters params;
    params.extent.width = width;
    params.extent.height = height;
    params.useAlphaThreshold = useAlphaThreshold;
    params.alphaThreshold = alphaThreshold;
    params.useMSLE = useMSLE;
    ntc::ComputePassDesc computePass{};
    ntc::Status ntcStatus = context->MakeImageDifferenceComputePass(params, &computePass);
    CHECK_NTC_RESULT("MakeImageDifferenceComputePass");

    // Record the command list
    commandList->open();

    if (!reuseCompressedData)
    {
        CopyBlocksIntoBCTexture(commandList, textureResources, width, height);
    }

    if (!compareImagesPass.ExecuteComputePass(commandList,  computePass,
        textureResources.bc, 0, textureResources.color, mipLevel, 0))
    {
        commandList->close();
        return false;
    }

    commandList->close();

    // Execute the command list and read the outputs
    device->executeCommandList(commandList);
    device->waitForIdle();

    if (!compareImagesPass.ReadResults())
        return false;

    if (!compareImagesPass.GetQueryResult(0, nullptr, outOverallMSE, outOverallPSNR, channels))
        return false;

    return true;
}

bool BlockCompressAndSaveGraphicsTextures(
    ntc::IContext* context,
    ntc::ITextureSetMetadata* metadata,
    ntc::IStream* inputFile,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    nvrhi::ITimerQuery* timerQuery,
    GDeflateFeatures* gdeflateFeatures,
    char const* savePath,
    int benchmarkIterations,
    GraphicsResourcesForTextureSet const& graphicsResources)
{
    GraphicsBlockCompressionPass blockCompressionPass(device, 2);
    if (!blockCompressionPass.Init())
        return false;

    GraphicsImageDifferencePass compareImagesPass(device);
    if (!compareImagesPass.Init())
        return false;

    float const alphaThreshold = 1.f / 255.f;

    ntc::TextureSetDesc const& textureSetDesc = metadata->GetDesc();
    
    for (int index = 0; index < int(graphicsResources.perTexture.size()); ++index)
    {
        GraphicsResourcesForTexture const& textureResources = graphicsResources.perTexture[index];
        ntc::ITextureMetadata* textureMetadata = metadata->GetTexture(index);
        ntc::BlockCompressedFormat bcFormat = textureMetadata->GetBlockCompressedFormat();

        if (bcFormat == ntc::BlockCompressedFormat::None)
            continue;

        bool const useAlphaThreshold = bcFormat == ntc::BlockCompressedFormat::BC1;
        bool const useMSLE = bcFormat == ntc::BlockCompressedFormat::BC6;

        nvrhi::BufferHandle modeBuffer;
        std::vector<BufferLoadingTask> modeBufferTasks;

        if (bcFormat == ntc::BlockCompressedFormat::BC7)
        {
            size_t stagingBufferSize = 0;
            size_t tempBufferSize = 0;
            size_t finalBufferSize = 0;

            FillBufferLoadingTasksForBC(textureSetDesc, textureMetadata, modeBufferTasks,
                gdeflateFeatures && gdeflateFeatures->gpuDecompressionSupported, device->getGraphicsAPI(),
                stagingBufferSize, tempBufferSize, finalBufferSize);
            
            if (!ExecuteBufferLoadingTasks(device, commandList, context, inputFile, gdeflateFeatures,
                modeBufferTasks, modeBuffer, stagingBufferSize, tempBufferSize, finalBufferSize))
                return false;
        }

        nvrhi::TextureDesc const& textureDesc = textureResources.color->getDesc();
        BcFormatDefinition const* bcFormatDef = GetBcFormatDefinition(bcFormat);

        std::string outputFileName = (fs::path(savePath) / fs::path(textureResources.name)).generic_string() + ".dds";
        ntc::FileStreamWrapper outputFile(context);
        ntc::Status ntcStatus = context->OpenFile(outputFileName.c_str(), true, outputFile.ptr());
        if (ntcStatus != ntc::Status::Ok)
        {
            fprintf(stderr, "Failed to open output file '%s', code = %s: %s\n", outputFileName.c_str(),
                StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        ntc::ColorSpace const rgbColorSpace = textureMetadata->GetRgbColorSpace();
        if (!WriteDdsHeader(outputFile, textureDesc.width, textureDesc.height, textureDesc.mipLevels, bcFormatDef, rgbColorSpace))
        {
            fprintf(stderr, "Failed to write into output file '%s': %s.\n", outputFileName.c_str(), strerror(errno));
            return false;
        }

        float perMipCompressionTimeMs[NTC_MAX_MIPS]{};
        float perMipMSE[NTC_MAX_MIPS]{};
        float perMipPSNR[NTC_MAX_MIPS]{};

        for (int mipLevel = 0; mipLevel < int(textureDesc.mipLevels); ++mipLevel)
        {
            uint32_t const mipWidth = std::max(textureDesc.width >> mipLevel, 1u);
            uint32_t const mipHeight = std::max(textureDesc.height >> mipLevel, 1u);

            uint32_t const mipWidthBlocks = (mipWidth + 3) / 4;
            uint32_t const mipHeightBlocks = (mipHeight + 3) / 4;

            ntc::MakeBlockCompressionComputePassParameters params;
            params.srcRect.width = int(mipWidth);
            params.srcRect.height = int(mipHeight);
            params.dstFormat = bcFormat;
            params.alphaThreshold = alphaThreshold;
            if (!modeBufferTasks.empty())
            {
                params.modeBufferSource = modeBufferTasks[mipLevel].pipeline != BufferLoadingPipeline::None
                    ? ntc::BlockCompressionModeBufferSource::TextureSet
                    : ntc::BlockCompressionModeBufferSource::None;
                params.modeBufferByteOffset = modeBufferTasks[mipLevel].finalBufferRange.byteOffset;
            }
            params.modeBufferInfo.textureSet.texture = textureMetadata;
            params.modeBufferInfo.textureSet.mipLevel = mipLevel;

            ntc::ComputePassDesc computePass{};
            ntcStatus = context->MakeBlockCompressionComputePass(params, &computePass);
            CHECK_NTC_RESULT("MakeBlockCompressionComputePass");
            
            std::vector<float> iterationTimes;
            iterationTimes.resize(benchmarkIterations);
            
            nvrhi::TextureSlice const slice = nvrhi::TextureSlice()
                .setWidth(mipWidthBlocks)
                .setHeight(mipHeightBlocks);

            for (int iteration = 0; iteration < benchmarkIterations; ++iteration)
            {
                commandList->open();
                commandList->beginTimerQuery(timerQuery);

                if (!blockCompressionPass.ExecuteComputePass(commandList, computePass,
                    textureResources.color, nvrhi::Format::UNKNOWN, mipLevel,
                    params.modeBufferSource != ntc::BlockCompressionModeBufferSource::None ? modeBuffer : nullptr,
                    textureResources.blocks, 0))
                {
                    commandList->close();
                    return false;
                }

                commandList->endTimerQuery(timerQuery);

                commandList->copyTexture(textureResources.stagingBlocks, slice, textureResources.blocks, slice);
                commandList->close();

                device->executeCommandList(commandList);
                device->waitForIdle(); 
                device->runGarbageCollection();

                float const iterationTimeSeconds = device->getTimerQueryTime(timerQuery);
                iterationTimes[iteration] = iterationTimeSeconds;
            }
            
            float const compressTimeSeconds = Median(iterationTimes);
            perMipCompressionTimeMs[mipLevel] = compressTimeSeconds * 1e3f;

            // Compute the compression PSNR
            ComputeBlockCompressedImageError(context, compareImagesPass, device, commandList, textureResources,
                mipWidth, mipHeight, mipLevel, false, useAlphaThreshold, alphaThreshold, useMSLE,
                &perMipMSE[mipLevel], &perMipPSNR[mipLevel], bcFormatDef->channels);

            size_t rowPitch = 0;
            uint8_t const* mappedData = static_cast<uint8_t const*>(device->mapStagingTexture(
                textureResources.stagingBlocks, slice, nvrhi::CpuAccessMode::Read, &rowPitch));
            if (!mappedData)
                return false;

            bool success = true;
            for (uint32_t row = 0; row < mipHeightBlocks; ++row)
            {
                if (!outputFile->Write(mappedData + rowPitch * row, mipWidthBlocks * bcFormatDef->bytesPerBlock))
                {
                    success = false;
                    break;
                }
            }

            device->unmapStagingTexture(textureResources.stagingBlocks);

            if (!success)
            {
                fprintf(stderr, "Failed to write into output file '%s': %s.\n", outputFileName.c_str(), strerror(errno));
                return false;
            }
        }

        outputFile.Close();

        printf("Saved image '%s': %dx%d pixels, %d mips, %s:\n",
            outputFileName.c_str(), textureDesc.width, textureDesc.height, textureDesc.mipLevels,
            ntc::BlockCompressedFormatToString(bcFormatDef->ntcFormat));

        for (int mipLevel = 0; mipLevel < int(textureDesc.mipLevels); ++mipLevel)
        {
            printf("  MIP %2d ", mipLevel);
            if (useMSLE)
            {
                printf("MSLE: %.6f", sqrtf(perMipMSE[mipLevel]));
            }
            else
            {
                printf("PSNR: %.2f dB", perMipPSNR[mipLevel]);
            }
            if (benchmarkIterations > 1)
            {
                printf(", Encoding time: %.2f ms", perMipCompressionTimeMs[mipLevel]);
            }
            if (bcFormat == ntc::BlockCompressedFormat::BC7)
            {
                bool const useModeBuffer = !modeBufferTasks.empty() &&
                    modeBufferTasks[mipLevel].pipeline != BufferLoadingPipeline::None;
                printf(", Accelerated: %s", useModeBuffer ? "YES" : "NO");
            }
            printf("\n");
        }
    }

    return true;
}

bool OptimizeBlockCompression(
    ntc::IContext* context,
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    float psnrThreshold,
    GraphicsResourcesForTextureSet const& graphicsResources)
{
    assert(device);
    
    bool anyBC7Textures = false;
    for (int textureIndex = 0; textureIndex < textureSetMetadata->GetTextureCount(); ++textureIndex)
    {
        ntc::ITextureMetadata* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        if (textureMetadata->GetBlockCompressedFormat() == ntc::BlockCompressedFormat::BC7)
        {
            anyBC7Textures = true;
            break;
        }
    }

    if (!anyBC7Textures)
        return true;

    GraphicsBlockCompressionPass blockCompressionPass(device, 1);
    if (!blockCompressionPass.Init())
        return false;

    GraphicsImageDifferencePass compareImagesPass(device);
    if (!compareImagesPass.Init())
        return false;
    
    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();

    nvrhi::BufferHandle modeBuffer;
    std::vector<uint8_t> blockCompressedData;

    for (int textureIndex = 0; textureIndex < textureSetMetadata->GetTextureCount(); ++textureIndex)
    {
        ntc::ITextureMetadata* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        if (textureMetadata->GetBlockCompressedFormat() != ntc::BlockCompressedFormat::BC7)
            continue;

        GraphicsResourcesForTexture const& textureResources = graphicsResources.perTexture[textureIndex];

        for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
        {
            int mipWidth = std::max(textureSetDesc.width >> mipLevel, 1);
            int mipHeight = std::max(textureSetDesc.height >> mipLevel, 1);
            int mipWidthInBlocks = (mipWidth + 3) / 4;
            int mipHeightInBlocks = (mipHeight + 3) / 4;

            // First pass - compress without the mode buffer, doing an exhaustive mode search for each block

            ntc::MakeBlockCompressionComputePassParameters compressionParams;
            compressionParams.srcRect.width = mipWidth;
            compressionParams.srcRect.height = mipHeight;
            compressionParams.dstFormat = textureMetadata->GetBlockCompressedFormat();
            ntc::ComputePassDesc blockCompressionComputePass;
            ntc::Status ntcStatus = context->MakeBlockCompressionComputePass(compressionParams, &blockCompressionComputePass);
            CHECK_NTC_RESULT("MakeBlockCompressionComputePass");
            
            commandList->open();

            if (!blockCompressionPass.ExecuteComputePass(commandList, blockCompressionComputePass,
                textureResources.color, nvrhi::Format::UNKNOWN, mipLevel,
                /* modeBuffer = */ nullptr,
                textureResources.blocks, /* outputMipLevel = */ 0))
            {
                commandList->close();
                return false;
            }
            
            commandList->copyTexture(textureResources.stagingBlocks, nvrhi::TextureSlice(),
                textureResources.blocks, nvrhi::TextureSlice());

            commandList->close();

            device->executeCommandList(commandList);
            device->waitForIdle();
            device->runGarbageCollection();
            
            // Read the block-compressed data back into CPU memory

            size_t rowPitch = 0;
            size_t blockDataSize = 0;
            {
                void const* mappedBlockData = device->mapStagingTexture(
                    textureResources.stagingBlocks, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);
                if (!mappedBlockData)
                    return false;

                // Copy the block-compressed data into a CPU buffer for more efficient access
                // and for later validation of the optimized compression pass.
                blockDataSize = rowPitch * mipHeightInBlocks;
                blockCompressedData.resize(blockDataSize);
                memcpy(blockCompressedData.data(), mappedBlockData, blockDataSize);

                device->unmapStagingTexture(textureResources.stagingBlocks);
            }

            // Use the block-compressed data to fill the BC7 mode buffer

            ntcStatus = textureMetadata->MakeAndStoreBC7ModeBuffer(mipLevel, mipWidthInBlocks, mipHeightInBlocks,
                blockCompressedData.data(), blockDataSize, rowPitch);

            CHECK_NTC_RESULT("MakeAndStoreBC7ModeBuffer");

            // Retrieve the mode buffer data we just created above

            void const* modeBufferData = nullptr;
            size_t modeBufferSize = 0;
            textureMetadata->GetBC7ModeBuffer(mipLevel, &modeBufferData, &modeBufferSize);
            
            if (modeBufferData == nullptr || modeBufferSize == 0)
            {
                fprintf(stderr, "Failed to retrieve BC7 mode buffer for texture '%s' mip %d after optimization.\n",
                    textureResources.name.c_str(), mipLevel);
                return false;
            }
            
            if (!modeBuffer)
            {
                nvrhi::BufferDesc modeBufferDesc = nvrhi::BufferDesc()
                    .setDebugName("BC7 Mode Buffer")
                    .setByteSize(modeBufferSize)
                    .setCanHaveRawViews(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                
                modeBuffer = device->createBuffer(modeBufferDesc);
                if (!modeBuffer)
                    return false;
            }

            // We create the mode buffer for mip 0, and reuse it for all mips in the pass.
            // The mip 0 buffer is always larger than the others, so this is safe.
            assert(modeBuffer->getDesc().byteSize >= modeBufferSize);

            // Second pass - compress using the mode buffer for validation

            compressionParams.modeBufferSource = ntc::BlockCompressionModeBufferSource::TextureSet;
            compressionParams.modeBufferInfo.textureSet.texture = textureMetadata;
            compressionParams.modeBufferInfo.textureSet.mipLevel = mipLevel;
            ntcStatus = context->MakeBlockCompressionComputePass(compressionParams, &blockCompressionComputePass);
            CHECK_NTC_RESULT("MakeBlockCompressionComputePass #2");

            commandList->open();
            commandList->writeBuffer(modeBuffer, modeBufferData, modeBufferSize);

            if (!blockCompressionPass.ExecuteComputePass(commandList, blockCompressionComputePass,
                textureResources.color, nvrhi::Format::UNKNOWN, mipLevel,
                modeBuffer,
                textureResources.blocks, /* outputMipLevel = */ 0))
            {
                commandList->close();
                return false;
            }

            commandList->copyTexture(textureResources.stagingBlocks, nvrhi::TextureSlice(),
                textureResources.blocks, nvrhi::TextureSlice());

            commandList->close();
            device->executeCommandList(commandList);
            device->waitForIdle();
            device->runGarbageCollection();

            // Map the compressed data and compare against the original compression output.
            // They should match exactly.

            {
                void const* mappedBlockData = device->mapStagingTexture(
                    textureResources.stagingBlocks, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);
                if (!mappedBlockData)
                    return false;

                if (memcmp(mappedBlockData, blockCompressedData.data(), blockDataSize) != 0)
                {
                    fprintf(stderr, "Warning: Optimized BC7 compression produced different data for texture '%s' mip %d.\n",
                        textureResources.name.c_str(), mipLevel);
                }

                device->unmapStagingTexture(textureResources.stagingBlocks);
            }
        }
    }
    
    return true;
}

bool ComputePsnrForBlockCompressedTextureSet(
    ntc::IContext* context,
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    GraphicsResourcesForTextureSet const& graphicsResources,
    float& outTargetPsnr)
{
    assert(device);
    
    GraphicsBlockCompressionPass blockCompressionPass(device, 1);
    if (!blockCompressionPass.Init())
        return false;

    GraphicsImageDifferencePass compareImagesPass(device);
    if (!compareImagesPass.Init())
        return false;

    std::vector<float> perChannelMSE;

    float const alphaThreshold = 1.f / 255.f;
    float combinedBcBitsPerPixel = 0;

    for (int textureIndex = 0; textureIndex < textureSetMetadata->GetTextureCount(); ++textureIndex)
    {
        ntc::ITextureMetadata* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        ntc::BlockCompressedFormat const bcFormat = textureMetadata->GetBlockCompressedFormat();
        int const numChannels = textureMetadata->GetNumChannels();
        if (textureMetadata->GetBlockCompressedFormat() == ntc::BlockCompressedFormat::None)
            continue;
        
        int const bytesPerBlock = GetBcFormatDefinition(textureMetadata->GetBlockCompressedFormat())->bytesPerBlock;
        combinedBcBitsPerPixel += float(bytesPerBlock) * 0.5f; // (* 8 bits / 16 pixels)
           
        GraphicsResourcesForTexture const& textureResources = graphicsResources.perTexture[textureIndex];

        nvrhi::TextureDesc const& textureDesc = textureResources.color->getDesc();
        int const width = int(textureDesc.width);
        int const height = int(textureDesc.height);
        
        // Make the compression pass
        ntc::MakeBlockCompressionComputePassParameters compressParams;
        compressParams.srcRect.width = width;
        compressParams.srcRect.height = height;
        compressParams.dstFormat = bcFormat;
        compressParams.alphaThreshold = alphaThreshold;
        ntc::ComputePassDesc blockCompressionComputePass;
        ntc::Status ntcStatus = context->MakeBlockCompressionComputePass(compressParams, &blockCompressionComputePass);
        CHECK_NTC_RESULT("MakeBlockCompressionComputePass");
        
        // Make the image comparison pass
        ntc::MakeImageDifferenceComputePassParameters differenceParams;
        differenceParams.extent.width = width;
        differenceParams.extent.height = height;
        differenceParams.useAlphaThreshold = (bcFormat == ntc::BlockCompressedFormat::BC1) && (numChannels == 4);
        differenceParams.alphaThreshold = alphaThreshold;
        ntc::ComputePassDesc imageDifferenceComputePass;
        ntcStatus = context->MakeImageDifferenceComputePass(differenceParams, &imageDifferenceComputePass);
        CHECK_NTC_RESULT("MakeImageDifferenceComputePass");

        commandList->open();
        
        // Compress the color texture into the block texture
        if (!blockCompressionPass.ExecuteComputePass(commandList, blockCompressionComputePass,
            textureResources.color, nvrhi::Format::UNKNOWN, /* inputMipLevel = */ 0,
            /* modeBuffer = */ nullptr,
            textureResources.blocks, /* outputMipLevel = */ 0))
        {
            commandList->close();
            return false;
        }

        // Copy compressed data from the block texture into the BCn texture
        CopyBlocksIntoBCTexture(commandList, textureResources, width, height);
        
        // Compare the BCn texture with the original color texture
        if (!compareImagesPass.ExecuteComputePass(commandList, imageDifferenceComputePass,
            textureResources.bc, 0, textureResources.color, 0, 0))
        {
            commandList->close();
            return false;
        }

        commandList->close();

        device->executeCommandList(commandList);
        device->waitForIdle();
        device->runGarbageCollection();
        
        // Read the per-channel MSE values and overall PSNR
        
        if (!compareImagesPass.ReadResults())
            return false;
            
        float mse[4];
        float psnr;
        if (!compareImagesPass.GetQueryResult(0, mse, nullptr, &psnr, numChannels))
            return false;

        // Append the MSE values for the valid channels in this texture into the overall MSE vector
        for (int ch = 0; ch < numChannels; ++ch)
            perChannelMSE.push_back(mse[ch]);

        printf("Compressed texture '%s' as %s, PSNR = %.2f dB.\n", textureResources.name.c_str(),
            ntc::BlockCompressedFormatToString(bcFormat), psnr);
    }

    int const totalChannels = int(perChannelMSE.size());
    assert(totalChannels > 0); // We shouldn't enter this function if there are no BCn textures
    float const overallMSE = std::accumulate(perChannelMSE.begin(), perChannelMSE.end(), 0.f) / totalChannels;
    float const overallPSNR = ntc::LossToPSNR(overallMSE);

    printf("Combined BCn PSNR: %.2f dB, bit rate: %.1f bpp.\n", overallPSNR, combinedBcBitsPerPixel);
    outTargetPsnr = overallPSNR;
    
    return true;
}
