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

#include "NtcMaterialLoader.h"
#include "NtcMaterial.h"
#include "NtcChannelMapping.h"
#include <ntc-utils/BufferLoading.h>
#include <ntc-utils/GraphicsDecompressionPass.h>
#include <ntc-utils/GraphicsBlockCompressionPass.h>
#include <ntc-utils/DeviceUtils.h>

#include <donut/core/log.h>
#include <donut/core/string_utils.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/Scene.h>

#include <sstream>
#include <fstream>

using namespace donut;
namespace fs = std::filesystem;

// Matches the number of textures in donut::engine::Material
static const uint32_t g_maxTileStagingTextures = 6;

// Skip this many largest latent mips when loading texture sets.
// Set to nonzero for testing purposes.
static const int g_firstLatentMipInTexture = 0;

bool NtcMaterialLoader::Init(bool enableCoopVec, bool enableGpuDeflate, bool debug, nvrhi::ITexture* dummyTexture)
{
    ntc::ContextParameters contextParams;
    contextParams.cudaDevice = ntc::DisableCudaDevice;
    contextParams.graphicsApi = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12
        ? ntc::GraphicsAPI::D3D12
        : ntc::GraphicsAPI::Vulkan;

    bool const osSupportsCoopVec = (contextParams.graphicsApi == ntc::GraphicsAPI::D3D12)
        ? IsDX12DeveloperModeEnabled()
        : true;
    
    contextParams.d3d12Device = m_device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    contextParams.vkInstance = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
    contextParams.vkPhysicalDevice = m_device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
    contextParams.vkDevice = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
    contextParams.enableCooperativeVector = osSupportsCoopVec && enableCoopVec;

    ntc::Status ntcStatus = ntc::CreateContext(m_ntcContext.ptr(), contextParams);
    if (ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable)
    {
        log::error("Failed to create an NTC context, code = %s: ",
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    m_coopVec = m_ntcContext->IsCooperativeVectorSupported();

    if (enableGpuDeflate)
    {
        m_gdeflateFeatures = InitGDeflate(m_device, debug);
    }

    m_dummyTexture = std::make_shared<engine::LoadedTexture>();
    m_dummyTexture->texture = dummyTexture;

    m_graphicsDecompressionPass = std::make_shared<GraphicsDecompressionPass>(m_device,
        /* descriptorTableSize = */ g_maxTileStagingTextures * 2 * TRANSCODE_BATCH_SIZE);
    if (!m_graphicsDecompressionPass->Init())
        return false;

    m_graphicsBlockCompressionPass = std::make_shared<GraphicsBlockCompressionPass>(m_device,
        /* maxConstantBufferVersions = */ 128);
    if (!m_graphicsBlockCompressionPass->Init())
        return false;

    m_commandList = m_device->createCommandList(nvrhi::CommandListParameters().setEnableImmediateExecution(false));

    // Create a buffer for uploading inference weights before their conversion to CoopVec format

    nvrhi::BufferDesc uploadBufferDesc = nvrhi::BufferDesc()
        .setByteSize(65536) // Should be enough for all weight types, validated in the upload code
        .setDebugName("Weight upload buffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);
    m_weightUploadBuffer = m_device->createBuffer(uploadBufferDesc);

    // Create tile staging color textures
    // All these textures are transient and could be aliased with other temp texture resources

    const uint32_t maxTileWidth = 512;
    const uint32_t maxTileHeight = 512;

    nvrhi::TextureDesc colorTextureDesc = nvrhi::TextureDesc()
        .setDimension(nvrhi::TextureDimension::Texture2D)
        .setWidth(maxTileWidth)
        .setHeight(maxTileHeight)
        .setMipLevels(1)
        .setDebugName("Tile color")
        .setIsUAV(true)
        .setIsTypeless(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true);

    for (uint32_t i = 0; i < g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE; i++)
    {
        colorTextureDesc.setFormat(nvrhi::Format::R8_UNORM);
        colorTextureDesc.setDebugName("Tile Color R8 " + std::to_string(i));
        nvrhi::TextureHandle tex = m_device->createTexture(colorTextureDesc);
        if (!tex)
            return false;
        m_texTranscodeTiles.push_back(tex);
    }

    for (uint32_t i = 0; i < g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE; i++)
    {
        colorTextureDesc.setFormat(nvrhi::Format::RGBA8_UNORM);
        colorTextureDesc.setDebugName("Tile Color RGBA " + std::to_string(i));
        nvrhi::TextureHandle tex = m_device->createTexture(colorTextureDesc);
        if (!tex)
            return false;
        m_texTranscodeTiles.push_back(tex);
    }

    // Create tile staging block textures

    nvrhi::TextureDesc blockTextureDesc = nvrhi::TextureDesc()
        .setDimension(nvrhi::TextureDimension::Texture2D)
        .setWidth(maxTileWidth / 4)
        .setHeight(maxTileHeight / 4)
        .setDebugName("Tile blocks")
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
        .setKeepInitialState(true);
        
    for (uint32_t i = 0; i < g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE; i++)
    {
        blockTextureDesc.setFormat(nvrhi::Format::RG32_UINT);
        blockTextureDesc.setDebugName("Tile Blocks RG " + std::to_string(i));
        nvrhi::TextureHandle tex = m_device->createTexture(blockTextureDesc);
        if (!tex)
            return false;
        m_texTranscodeTiles.push_back(tex);
    }

    for (uint32_t i = 0; i < g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE; i++)
    {
        blockTextureDesc.setFormat(nvrhi::Format::RGBA32_UINT);
        blockTextureDesc.setDebugName("Tile Blocks RGBA " + std::to_string(i));
        nvrhi::TextureHandle tex = m_device->createTexture(blockTextureDesc);
        if (!tex)
            return false;
        m_texTranscodeTiles.push_back(tex);
    }

    m_texTileColorR8Offset = 0;
    m_texTileColorRGBAOffset = 1 * g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE;
    m_texTileBlocksRGOffset = 2 * g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE;
    m_texTileBlocksRGBAOffset = 3 * g_maxTileStagingTextures * TRANSCODE_BATCH_SIZE;

    return true;
}

static bool TextureSetHasChannels(uint32_t mask, int first, int count)
{
    uint32_t test = ((1u << count) - 1) << first;
    return (mask & test) == test;
}

static void FillMaterialTranscodeMapping(NtcMaterial& material, ntc::ITextureSetMetadata* textureSetMetadata,
    MaterialChannelMap const& channelMap, bool onlyAlphaMask)
{
    // Derive the valid channel mask from the shuffle map, because textureSetMetadata->GetValidChannelMask()
    // returns "1" bits for constant channels, and we want to know where actual textures exist.
    uint32_t channelMask = 0;
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        if (channelMap.swizzle[ch].type == ntc::ShuffleSourceType::Channel)
            channelMask |= (1 << ch);
    }

    // If we only need to create the alpha mask texture, see if the material actually needs an alpha mask
    // and if the NTC texture set has an alpha mask channel.
    if (onlyAlphaMask)
    {
        // Test the channel mask for the presence of opacity channel.
        // Note: not testing the material domain because in some cases, one texture set can be reused
        // for multiple materials with different domains, and we cache and reuse the transcode mapping.
        // One model with such reuse is AlphaBlendModeTest from the Khronos glTF sample asset collection.
        bool opacityChannelPresent = TextureSetHasChannels(channelMask, CHANNEL_OPACITY, 1);

        if (!opacityChannelPresent)
            return;

        TextureTranscodeTask& opacityTexture = material.transcodeMapping.emplace_back();
        opacityTexture.bcFormat = ntc::BlockCompressedFormat::BC4;
        opacityTexture.nvrhiBcFormat = nvrhi::Format::BC4_UNORM;
        opacityTexture.firstChannel = CHANNEL_OPACITY;
        opacityTexture.numChannels = 1;
        opacityTexture.pMaterialTexture = &NtcMaterial::opacityTexture;
        opacityTexture.pFeedbackTexture = &NtcMaterial::opacityTextureFeedback;
        opacityTexture.name = "Opacity";
    }
    else
    {
        if (TextureSetHasChannels(channelMask, CHANNEL_BASE_COLOR, 3))
        {
            TextureTranscodeTask& baseColorTexture = material.transcodeMapping.emplace_back();
            baseColorTexture.bcFormat = ntc::BlockCompressedFormat::BC7;
            baseColorTexture.nvrhiBcFormat = nvrhi::Format::BC7_UNORM_SRGB;
            baseColorTexture.firstChannel = CHANNEL_BASE_COLOR;
            baseColorTexture.numChannels = TextureSetHasChannels(channelMask, CHANNEL_OPACITY, 1) ? 4 : 3;
            static_assert(CHANNEL_OPACITY == CHANNEL_BASE_COLOR + 3);
            baseColorTexture.sRGB = true;
            baseColorTexture.pMaterialTexture = &NtcMaterial::baseOrDiffuseTexture;
            baseColorTexture.pFeedbackTexture = &NtcMaterial::baseOrDiffuseTextureFeedback;
            baseColorTexture.name = "BaseColor";
        }
        
        if (material.useSpecularGlossModel)
        {
            if (TextureSetHasChannels(channelMask, CHANNEL_SPECULAR_COLOR, 3))
            {
                TextureTranscodeTask& specularTexture = material.transcodeMapping.emplace_back();
                specularTexture.bcFormat = ntc::BlockCompressedFormat::BC7;
                specularTexture.nvrhiBcFormat = nvrhi::Format::BC7_UNORM_SRGB;
                specularTexture.firstChannel = CHANNEL_SPECULAR_COLOR;
                specularTexture.numChannels = TextureSetHasChannels(channelMask, CHANNEL_GLOSSINESS, 1) ? 4 : 3;
                static_assert(CHANNEL_GLOSSINESS == CHANNEL_SPECULAR_COLOR + 3);
                specularTexture.sRGB = true;
                specularTexture.pMaterialTexture = &NtcMaterial::metalRoughOrSpecularTexture;
                specularTexture.pFeedbackTexture = &NtcMaterial::metalRoughOrSpecularTextureFeedback;
                specularTexture.name = "SpecularColor";
            }
        }
        else
        {
            if (TextureSetHasChannels(channelMask, CHANNEL_METALNESS, 2))
            {
                TextureTranscodeTask& metalRoughTexture = material.transcodeMapping.emplace_back();
                metalRoughTexture.bcFormat = ntc::BlockCompressedFormat::BC5;
                metalRoughTexture.nvrhiBcFormat = nvrhi::Format::BC5_UNORM;
                metalRoughTexture.firstChannel = CHANNEL_METALNESS;
                metalRoughTexture.numChannels = 2;
                static_assert(CHANNEL_ROUGHNESS == CHANNEL_METALNESS + 1);
                metalRoughTexture.pMaterialTexture = &NtcMaterial::metalRoughOrSpecularTexture;
                metalRoughTexture.pFeedbackTexture = &NtcMaterial::metalRoughOrSpecularTextureFeedback;
                metalRoughTexture.name = "MetallicRoughness";
            }
            
            if (TextureSetHasChannels(channelMask, CHANNEL_OCCLUSION, 1))
            {
                TextureTranscodeTask& occlusionTexture = material.transcodeMapping.emplace_back();
                occlusionTexture.bcFormat = ntc::BlockCompressedFormat::BC4;
                occlusionTexture.nvrhiBcFormat = nvrhi::Format::BC4_UNORM;
                occlusionTexture.firstChannel = CHANNEL_OCCLUSION;
                occlusionTexture.numChannels = 1;
                occlusionTexture.pMaterialTexture = &NtcMaterial::occlusionTexture;
                occlusionTexture.pFeedbackTexture = &NtcMaterial::occlusionTextureFeedback;
                occlusionTexture.name = "Occlusion";
            }
        }
        
        if (TextureSetHasChannels(channelMask, CHANNEL_NORMAL, 3))
        {
            TextureTranscodeTask& normalTexture = material.transcodeMapping.emplace_back();
            normalTexture.bcFormat = ntc::BlockCompressedFormat::BC7;
            normalTexture.nvrhiBcFormat = nvrhi::Format::BC7_UNORM;
            normalTexture.firstChannel = CHANNEL_NORMAL;
            normalTexture.numChannels = 3;
            normalTexture.pMaterialTexture = &NtcMaterial::normalTexture;
            normalTexture.pFeedbackTexture = &NtcMaterial::normalTextureFeedback;
            normalTexture.name = "Normal";
        }

        if (TextureSetHasChannels(channelMask, CHANNEL_EMISSIVE, 3))
        {
            TextureTranscodeTask& emissiveTexture = material.transcodeMapping.emplace_back();
            emissiveTexture.bcFormat = ntc::BlockCompressedFormat::BC7;
            emissiveTexture.nvrhiBcFormat = nvrhi::Format::BC7_UNORM_SRGB;
            emissiveTexture.firstChannel = CHANNEL_EMISSIVE;
            emissiveTexture.numChannels = 3;
            emissiveTexture.sRGB = true;
            emissiveTexture.pMaterialTexture = &NtcMaterial::emissiveTexture;
            emissiveTexture.pFeedbackTexture = &NtcMaterial::emissiveTextureFeedback;
            emissiveTexture.name = "Emissive";
        }

        if (TextureSetHasChannels(channelMask, CHANNEL_TRANSMISSION, 1))
        {
            TextureTranscodeTask& transmissionTexture = material.transcodeMapping.emplace_back();
            transmissionTexture.bcFormat = ntc::BlockCompressedFormat::BC4;
            transmissionTexture.nvrhiBcFormat = nvrhi::Format::BC4_UNORM;
            transmissionTexture.firstChannel = CHANNEL_TRANSMISSION;
            transmissionTexture.numChannels = 1;
            transmissionTexture.pMaterialTexture = &NtcMaterial::transmissionTexture;
            transmissionTexture.pFeedbackTexture = &NtcMaterial::transmissionTextureFeedback;
            transmissionTexture.name = "Transmission";
        }
    }

    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    for (int textureIndex = 0; textureIndex < material.transcodeMapping.size(); ++textureIndex)
    {
        TextureTranscodeTask& textureVersions = material.transcodeMapping[textureIndex];

        // Figure out which channels in the original NTC texture set correspond to the channels for this texture
        std::array<int, 4> originalChannels;
        originalChannels.fill(-1);
        bool originalChannelsSequential = true;
        for (int ch = 0; ch < textureVersions.numChannels; ++ch)
        {
            ntc::ShuffleSource const& src = channelMap.swizzle[textureVersions.firstChannel + ch];
            originalChannels[ch] = src.GetChannelIndex();
            if (ch > 0 && originalChannels[ch] != originalChannels[ch-1] + 1)
                originalChannelsSequential = false;
        }
        
        // If this texture covers a contiguous span of channels in the NTC texture set, try to find a matching
        // texture metadata object. It's only needed for BC7 encoding acceleration, so no big deal if it's not found.
        if (originalChannelsSequential)
        {
            for (int ntcTextureIndex = 0; ntcTextureIndex < textureSetMetadata->GetTextureCount(); ++ntcTextureIndex)
            {
                ntc::ITextureMetadata* textureMetadata = textureSetMetadata->GetTexture(ntcTextureIndex);
                if (textureMetadata->GetFirstChannel() == originalChannels[0] &&
                    textureMetadata->GetNumChannels() >= textureVersions.numChannels)
                {
                    textureVersions.metadata = textureMetadata;
                    break;
                }
            }
        }
    }
}

static bool LoadMaterialFile(donut::engine::FilePathOrInlineData const& source, NtcMaterial& material,
    ntc::IContext* ntcContext, ntc::FileStreamWrapper& ntcFile, ntc::MemoryStreamWrapper& ntcMemory,
    ntc::TextureSetMetadataWrapper& textureSetMetadata)
{
    if (material.name.empty())
        material.name = "Material";
    
    ntc::Status ntcStatus;
    ntc::IStream* stream = nullptr;

    if (source.data)
    {
        ntcStatus = ntcContext->OpenReadOnlyMemory(source.data->buffer->data(), source.data->buffer->size(),
            ntcMemory.ptr());

        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Cannot open '%s', error code = %s: %s", source.ToString().c_str(),
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        stream = ntcMemory.Get();
    }
    else
    {
        ntcStatus = ntcContext->OpenFile(source.path.c_str(), false, ntcFile.ptr());
        if (ntcStatus == ntc::Status::FileUnavailable)
        {
            log::warning("Material file '%s' does not exist.", source.path.c_str());
            return false;
        }
        else if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Cannot open '%s', error code = %s: %s", source.path.c_str(),
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        stream = ntcFile.Get();
    }

    ntcStatus = ntcContext->CreateTextureSetMetadataFromStream(stream, textureSetMetadata.ptr());
    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Cannot load metadata for '%s', error code = %s: %s", source.ToString().c_str(),
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    return true;
}

bool NtcMaterialLoader::TranscodeTiles(const std::vector<TranscodeTileInfo>& tiles, nvrhi::ICommandList* commandList,
    bool enableBlockCompression)
{
    if (tiles.empty())
        return true;

    assert(tiles.size() <= TRANSCODE_BATCH_SIZE);

    // Indices for grabbing the next available texture from the staging textures
    uint32_t texTileColorR8Index = m_texTileColorR8Offset;
    uint32_t texTileColorRGBAIndex = m_texTileColorRGBAOffset;
    uint32_t texTileBlocksRGIndex = m_texTileBlocksRGOffset;
    uint32_t texTileBlocksRGBAIndex = m_texTileBlocksRGBAOffset;

    // Write all descriptors for the color textures into the decompression pass descriptor table
    for (int descriptorIndex = 0; descriptorIndex < m_texTileBlocksRGOffset; ++descriptorIndex)
    {
        nvrhi::BindingSetItem descriptor = nvrhi::BindingSetItem::Texture_UAV(
            descriptorIndex,
            m_texTranscodeTiles[descriptorIndex]);
        m_graphicsDecompressionPass->WriteDescriptor(descriptor);
    }

    // Indices which map every tile/textureIndex into the list of temporary textures
    std::vector<uint32_t> colorTextureIndices;
    std::vector<uint32_t> blockTextureIndices;

    std::array<bool, TRANSCODE_BATCH_SIZE> compressThisTexture;

    commandList->beginMarker("Transcode Tiles: NTC Decompression");

    // Phase 1 - Select the temporary tile textures from the pool and make state transitions
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const NtcMaterial& material = *transcodeTile.material;

        int textureCount = int(material.transcodeMapping.size());
        assert(textureCount <= g_maxTileStagingTextures); // Maximum number of textures supported

        // TODO: Does this this need to be handled without faulting?
        assert(material.transcodeMapping.empty() == false);

        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            const TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];

            bool const isSingleChannel = transcodeTask.numChannels == 1;

            // Select the color texture
            uint32_t colorTextureIndex = isSingleChannel ? texTileColorR8Index++ : texTileColorRGBAIndex++;
            colorTextureIndices.push_back(colorTextureIndex);

            compressThisTexture[tileIndex] = transcodeTask.bcFormat != ntc::BlockCompressedFormat::None
                && enableBlockCompression;
            if (compressThisTexture[tileIndex])
            {
                // Select the block texture
                bool const isSmallBlock =
                    (transcodeTask.bcFormat == ntc::BlockCompressedFormat::BC1) ||
                    (transcodeTask.bcFormat == ntc::BlockCompressedFormat::BC4);

                uint32_t blockTextureIndex = isSmallBlock ? texTileBlocksRGIndex++ : texTileBlocksRGBAIndex++;
                blockTextureIndices.push_back(blockTextureIndex);

                // Disable automatic UAV barriers for the block texture
                commandList->setEnableUavBarriersForTexture(m_texTranscodeTiles[blockTextureIndex], false);
            }

            // Transition to UAV
            commandList->setTextureState(m_texTranscodeTiles[colorTextureIndex], nvrhi::AllSubresources,
                nvrhi::ResourceStates::UnorderedAccess);
        }
    }

    commandList->commitBarriers();

    // Phase 2 - Run NTC decompression

    static_assert(TRANSCODE_BATCH_SIZE <= 32, "successfulTilesMask assumes no more than 32 tiles per batch");
    uint32_t successfulTilesMask = 0;

    uint32_t colorTextureIndex = 0;
    uint32_t blockTextureIndex = 0;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const nvfeedback::FeedbackTextureTileInfo tileInfo = transcodeTile.tileInfo;
        const NtcMaterial& material = *transcodeTile.material;

        ntc::ITextureSetMetadata* textureSetMetadata = material.textureSetMetadata->Get();
        ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
        const uint32_t mipWidth = std::max(1, textureSetDesc.width >> tileInfo.mip);
        const uint32_t mipHeight = std::max(1, textureSetDesc.height >> tileInfo.mip);
        int textureCount = int(material.transcodeMapping.size());
        assert(textureCount <= g_maxTileStagingTextures); // Maximum number of textures supported

        // Make sure that the latent and weight buffers have already been created
        assert(material.ntcLatentsTexture);
        assert(material.ntcWeightsBuffer);

        m_graphicsDecompressionPass->SetLatentTexture(material.ntcLatentsTexture);
        m_graphicsDecompressionPass->SetWeightBuffer(material.ntcWeightsBuffer);

        ntc::Rect rectDecompress;
        rectDecompress.left = tileInfo.xInTexels;
        rectDecompress.top = tileInfo.yInTexels;
        rectDecompress.width = std::min(tileInfo.widthInTexels, mipWidth); // Tiles can be block sizes of 4x4 while the mip could be smaller
        rectDecompress.height = std::min(tileInfo.heightInTexels, mipHeight);

        ntc::Point offsetDecompress;
        offsetDecompress.x = 0;
        offsetDecompress.y = 0;
        
        std::array<ntc::OutputTextureDesc, g_maxTileStagingTextures> outputTextureDescs;
        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            const TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];
            ntc::OutputTextureDesc& outputDesc = outputTextureDescs[textureIndex];
            outputDesc.firstChannel = transcodeTask.firstChannel;
            outputDesc.numChannels = transcodeTask.numChannels;
            outputDesc.descriptorIndex = colorTextureIndices[colorTextureIndex++];
            outputDesc.rgbColorSpace = transcodeTask.sRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
            outputDesc.ditherScale = 1.f / 255.f;
        }

        ntc::MakeDecompressionComputePassParameters decompressionParams;
        decompressionParams.textureSetMetadata = textureSetMetadata;
        decompressionParams.mipLevel = tileInfo.mip;
        decompressionParams.firstOutputDescriptorIndex = 0;
        decompressionParams.firstLatentMipInTexture = g_firstLatentMipInTexture;
        decompressionParams.pOutputTextures = outputTextureDescs.data();
        decompressionParams.numOutputTextures = textureCount;
        decompressionParams.weightType = ntc::InferenceWeightType(material.weightType);
        decompressionParams.pSrcRect = &rectDecompress;
        decompressionParams.pDstOffset = &offsetDecompress;
        ntc::ComputePassDesc decompressionPass;
        ntc::Status ntcStatus = m_ntcContext->MakeDecompressionComputePass(decompressionParams, &decompressionPass);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Failed to make a decompression pass for material '%s' mip %d, error code = %s: %s",
                material.name.c_str(), tileInfo.mip, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            continue;
        }

        m_graphicsDecompressionPass->ExecuteComputePass(commandList, decompressionPass);

        successfulTilesMask |= (1 << tileIndex);
    }

    commandList->endMarker();

    // Phase 3 - Compress all mips of the color textures into BCn, where necessary

    commandList->beginMarker("Transcode Tiles: BCn Compression");

    // Transition textures for BCn compression
    colorTextureIndex = 0;
    blockTextureIndex = 0;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        if ((successfulTilesMask & (1 << tileIndex)) == 0)
            continue;
            
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const nvfeedback::FeedbackTextureTileInfo tileInfo = transcodeTile.tileInfo;
        const NtcMaterial& material = *transcodeTile.material;
        int textureCount = int(material.transcodeMapping.size());
        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            nvrhi::TextureHandle colorTexture = m_texTranscodeTiles[colorTextureIndices[colorTextureIndex++]];
            if (compressThisTexture[tileIndex])
            {
                nvrhi::TextureHandle blockTexture = m_texTranscodeTiles[blockTextureIndices[blockTextureIndex++]];
                commandList->setTextureState(colorTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
                commandList->setTextureState(blockTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            }
        }
    }
    commandList->commitBarriers();

    colorTextureIndex = 0;
    blockTextureIndex = 0;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        if ((successfulTilesMask & (1 << tileIndex)) == 0)
            continue;
            
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const nvfeedback::FeedbackTextureTileInfo tileInfo = transcodeTile.tileInfo;
        const NtcMaterial& material = *transcodeTile.material;

        ntc::ITextureSetMetadata* textureSetMetadata = material.textureSetMetadata->Get();
        ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
        const uint32_t mipWidth = std::max(1, textureSetDesc.width >> tileInfo.mip);
        const uint32_t mipHeight = std::max(1, textureSetDesc.height >> tileInfo.mip);
        int textureCount = int(material.transcodeMapping.size());
        assert(textureCount <= g_maxTileStagingTextures); // Maximum number of textures supported

        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            const TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];

            nvrhi::TextureHandle colorTexture = m_texTranscodeTiles[colorTextureIndices[colorTextureIndex++]];

            if (compressThisTexture[tileIndex])
            {
                nvrhi::TextureHandle blockTexture = m_texTranscodeTiles[blockTextureIndices[blockTextureIndex++]];
                float const alphaThreshold = 1.f / 255.f;

                ntc::MakeBlockCompressionComputePassParameters compressionParams;
                // Tiles can be block sizes of 4x4 while the mip is smaller
                compressionParams.srcRect.width = std::min(tileInfo.widthInTexels, mipWidth);
                compressionParams.srcRect.height = std::min(tileInfo.heightInTexels, mipHeight);
                compressionParams.dstFormat = transcodeTask.bcFormat;
                compressionParams.alphaThreshold = alphaThreshold;
                nvrhi::IBuffer* modeBuffer = nullptr;
                if (transcodeTask.bc7ModeBuffer && transcodeTask.bc7ModeBufferMipRanges[tileInfo.mip].byteSize != 0)
                {
                    compressionParams.modeBufferSource = ntc::BlockCompressionModeBufferSource::TextureSet;
                    compressionParams.modeBufferByteOffset = transcodeTask.bc7ModeBufferMipRanges[tileInfo.mip].byteOffset;
                    compressionParams.modeBufferInfo.textureSet.texture = transcodeTask.metadata;
                    compressionParams.modeBufferInfo.textureSet.mipLevel = tileInfo.mip;
                    compressionParams.modeMapOffsetInBlocks.x = tileInfo.xInTexels >> 2;
                    compressionParams.modeMapOffsetInBlocks.y = tileInfo.yInTexels >> 2;
                    modeBuffer = transcodeTask.bc7ModeBuffer;
                }
                else
                {
                    compressionParams.modeBufferSource = ntc::BlockCompressionModeBufferSource::None;
                }
                ntc::ComputePassDesc compressionPass;
                ntc::Status ntcStatus = m_ntcContext->MakeBlockCompressionComputePass(compressionParams, &compressionPass);

                if (ntcStatus != ntc::Status::Ok)
                {
                    log::warning("Failed to make a block compression pass for material '%s', error code = %s: %s",
                        material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                    return false;
                }

                nvrhi::Format const inputFormat = (transcodeTask.numChannels == 1)
                    ? nvrhi::Format::R8_UNORM
                    : nvrhi::Format::RGBA8_UNORM;

                if (!m_graphicsBlockCompressionPass->ExecuteComputePass(commandList, compressionPass,
                    colorTexture, inputFormat, 0, modeBuffer, blockTexture, 0))
                    return false;
            }
        }
    }
    commandList->endMarker();

    // Phase 4 - Copy tiles to the destination tiled resources

    commandList->beginMarker("Transcode Tiles: Copy to Tiled Resources");

    // Transition textures for copying
    colorTextureIndex = 0;
    blockTextureIndex = 0;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        if ((successfulTilesMask & (1 << tileIndex)) == 0)
            continue;
            
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const nvfeedback::FeedbackTextureTileInfo tileInfo = transcodeTile.tileInfo;
        const NtcMaterial& material = *transcodeTile.material;
        int textureCount = int(material.transcodeMapping.size());
        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            const TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];
            nvrhi::ITexture* pDestTexture = (material.*transcodeTask.pFeedbackTexture)->GetReservedTexture();
            commandList->setTextureState(pDestTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
            
            nvrhi::TextureHandle colorTexture = m_texTranscodeTiles[colorTextureIndices[colorTextureIndex++]];
            if (compressThisTexture[tileIndex])
            {
                nvrhi::TextureHandle blockTexture = m_texTranscodeTiles[blockTextureIndices[blockTextureIndex++]];
                commandList->setTextureState(blockTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
            }
            else
            {
                commandList->setTextureState(colorTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
            }
        }
    }
    commandList->commitBarriers();

    colorTextureIndex = 0;
    blockTextureIndex = 0;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex)
    {
        if ((successfulTilesMask & (1 << tileIndex)) == 0)
            continue;
            
        const TranscodeTileInfo& transcodeTile = tiles[tileIndex];
        const nvfeedback::FeedbackTextureTileInfo tileInfo = transcodeTile.tileInfo;
        const NtcMaterial& material = *transcodeTile.material;

        ntc::ITextureSetMetadata* textureSetMetadata = material.textureSetMetadata->Get();
        ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
        const uint32_t mipWidth = std::max(1, textureSetDesc.width >> tileInfo.mip);
        const uint32_t mipHeight = std::max(1, textureSetDesc.height >> tileInfo.mip);
        int textureCount = int(material.transcodeMapping.size());
        assert(textureCount <= g_maxTileStagingTextures); // Maximum number of textures supported

        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            const TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];
            nvrhi::ITexture* pDestTexture = (material.*transcodeTask.pFeedbackTexture)->GetReservedTexture();

            nvrhi::TextureSlice textureSliceDst = {};
            textureSliceDst.x = tileInfo.xInTexels;
            textureSliceDst.y = tileInfo.yInTexels;
            textureSliceDst.z = 0;
            textureSliceDst.mipLevel = tileInfo.mip;
            textureSliceDst.width = tileInfo.widthInTexels;
            textureSliceDst.height = tileInfo.heightInTexels;
            textureSliceDst.depth = 1;

            nvrhi::TextureHandle colorTexture = m_texTranscodeTiles[colorTextureIndices[colorTextureIndex++]];

            if (compressThisTexture[tileIndex])
            {
                nvrhi::TextureHandle blockTexture = m_texTranscodeTiles[blockTextureIndices[blockTextureIndex++]];

                nvrhi::TextureSlice textureSliceSrc = {};
                textureSliceSrc.x = 0;
                textureSliceSrc.y = 0;
                textureSliceSrc.z = 0;
                textureSliceSrc.mipLevel = 0;
                textureSliceSrc.width = (tileInfo.widthInTexels + 3) / 4;
                textureSliceSrc.height = (tileInfo.heightInTexels + 3) / 4;
                textureSliceSrc.depth = 1;

                commandList->copyTexture(pDestTexture, textureSliceDst, blockTexture, textureSliceSrc);
            }
            else
            {
                nvrhi::TextureSlice textureSliceSrc = {};
                textureSliceSrc.x = 0;
                textureSliceSrc.y = 0;
                textureSliceSrc.z = 0;
                textureSliceSrc.mipLevel = 0;
                textureSliceSrc.width = tileInfo.widthInTexels;
                textureSliceSrc.height = tileInfo.heightInTexels;
                textureSliceSrc.depth = 1;

                commandList->copyTexture(pDestTexture, textureSliceDst, colorTexture, textureSliceSrc);
            }
        }
    }
    commandList->endMarker();


    return true;
}

bool NtcMaterialLoader::CreateAndLoadModeBufferForTexture(ntc::IContext* ntcContext, ntc::IStream* ntcFile,
    ntc::ITextureSetMetadata* textureSetMetadata, TextureTranscodeTask& transcodeTask,
    nvrhi::ICommandList* commandList, std::string const& materialTextureName)
{
    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();

    std::vector<BufferLoadingTask> modeBufferTasks;

    size_t stagingBufferSize = 0;
    size_t tempBufferSize = 0;
    size_t finalBufferSize = 0;

    FillBufferLoadingTasksForBC(textureSetDesc, transcodeTask.metadata, modeBufferTasks,
        m_gdeflateFeatures && m_gdeflateFeatures->gpuDecompressionSupported, m_device->getGraphicsAPI(),
        stagingBufferSize, tempBufferSize, finalBufferSize);
    
    if (!ExecuteBufferLoadingTasks(m_device, commandList, ntcContext, ntcFile, m_gdeflateFeatures.get(),
        modeBufferTasks, transcodeTask.bc7ModeBuffer, stagingBufferSize, tempBufferSize, finalBufferSize))
        return false;

    transcodeTask.bc7ModeBufferMipRanges.resize(modeBufferTasks.size());
    for (size_t mip = 0; mip < modeBufferTasks.size(); ++mip)
    {
        BufferLoadingTask const& mipTask = modeBufferTasks[mip];
        if (mipTask.pipeline != BufferLoadingPipeline::None)
            transcodeTask.bc7ModeBufferMipRanges[mip] = mipTask.finalBufferRange;
    }

    return true;
}

bool NtcMaterialLoader::TranscodeMaterial(ntc::IContext* context, ntc::IStream* ntcFile,
    ntc::ITextureSetMetadata* textureSetMetadata,
    NtcMaterial& material, nvrhi::ICommandList* commandList, bool enableBlockCompression)
{
    if (material.transcodeMapping.empty())
        return true;

    int const textureCount = int(material.transcodeMapping.size());

    // Phase 1 - Create textures (color, block, BCn) and write descriptors for NTC decompression

    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];
        std::string const materialTextureName = material.name + ":" + std::string(transcodeTask.name);

        // Create the color texture

        nvrhi::TextureDesc colorTextureDesc = nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(textureSetDesc.width)
            .setHeight(textureSetDesc.height)
            .setMipLevels(textureSetDesc.mips)
            .setFormat((transcodeTask.numChannels == 1)
                ? nvrhi::Format::R8_UNORM
                : transcodeTask.sRGB
                    ? nvrhi::Format::SRGBA8_UNORM
                    : nvrhi::Format::RGBA8_UNORM)
            .setDebugName(materialTextureName)
            .setIsUAV(true)
            .setIsTypeless(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        transcodeTask.color = m_device->createTexture(colorTextureDesc);
        if (!transcodeTask.color)
            return false;

        bool compressThisTexture = enableBlockCompression && transcodeTask.bcFormat != ntc::BlockCompressedFormat::None;

        if (compressThisTexture)
        {
            // Create the BCn texture

            nvrhi::TextureDesc compressedTextureDesc = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setFormat(transcodeTask.nvrhiBcFormat)
                .setWidth(textureSetDesc.width)
                .setHeight(textureSetDesc.height)
                .setMipLevels(textureSetDesc.mips)
                .setDebugName(materialTextureName)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true);

            transcodeTask.compressed = m_device->createTexture(compressedTextureDesc);
            if (!transcodeTask.compressed)
                return false;

            // Create the block texture

            bool const isSmallBlock =
                (transcodeTask.bcFormat == ntc::BlockCompressedFormat::BC1) ||
                (transcodeTask.bcFormat == ntc::BlockCompressedFormat::BC4);

            nvrhi::TextureDesc blockTextureDesc = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth((textureSetDesc.width + 3) / 4)
                .setHeight((textureSetDesc.height + 3) / 4)
                .setFormat(isSmallBlock ? nvrhi::Format::RG32_UINT : nvrhi::Format::RGBA32_UINT)
                .setDebugName(materialTextureName)
                .setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true);

            transcodeTask.blocks = m_device->createTexture(blockTextureDesc);
            if (!transcodeTask.blocks)
                return false;

            if (transcodeTask.bcFormat == ntc::BlockCompressedFormat::BC7)
            {
                if (!CreateAndLoadModeBufferForTexture(context, ntcFile, textureSetMetadata, transcodeTask,
                    commandList, materialTextureName))
                    return false;
            }
        }
        
        // Descriptors for all mip levels of one texture are packed together
        transcodeTask.mipZeroDescriptor = textureSetDesc.mips * textureIndex;

        // Write descriptors for all mips of the color texture
        for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
        {
            int descriptorIndex = transcodeTask.mipZeroDescriptor + mipLevel;

            nvrhi::BindingSetItem descriptor = nvrhi::BindingSetItem::Texture_UAV(
                descriptorIndex,
                transcodeTask.color,
                (transcodeTask.numChannels == 1)
                    ? nvrhi::Format::R8_UNORM
                    : nvrhi::Format::RGBA8_UNORM, // Always use non-sRGB formats so that we can create a UAV
                nvrhi::TextureSubresourceSet().setBaseMipLevel(mipLevel));

            m_graphicsDecompressionPass->WriteDescriptor(descriptor);
        }

        // Create a LoadedTexture object to attach the texture to the material
        std::shared_ptr<engine::LoadedTexture> loadedTexture = std::make_shared<engine::LoadedTexture>();
        loadedTexture->texture = compressThisTexture ? transcodeTask.compressed : transcodeTask.color;

        // Count the final texture size in the material's memory consumption metric
        size_t const textureMemorySize = m_device->getTextureMemoryRequirements(loadedTexture->texture).size;
        material.transcodedMemorySize += textureMemorySize;
        
        // Bind the created texture object to the material texture slot
        material.*transcodeTask.pMaterialTexture = loadedTexture;
    }

    commandList->open();

    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];

        // Transition the texture to the UAV state because NVRHI won't do that when resources are accessed
        // through a descriptor table. Note that there is no need to transition it back to SRV after decompression
        // because the next operations are using regular binding sets. There is also no need for commitBarriers()
        // because that's called by the decompression dispatch call.
        commandList->setTextureState(transcodeTask.color, nvrhi::AllSubresources,
            nvrhi::ResourceStates::UnorderedAccess);
    }

    // Submit the texture transitions performed above via setTextureState(...) to the command list.
    // This is not really necessary because the next call to commandList->setComputeState(...) will do it,
    // but let's be explicit.
    commandList->commitBarriers();

    // Phase 2 - Run NTC decompression

    // Make sure that the latent and weight buffers have already been created
    assert(material.ntcLatentsTexture);
    assert(material.ntcWeightsBuffer);

    m_graphicsDecompressionPass->SetLatentTexture(material.ntcLatentsTexture);
    m_graphicsDecompressionPass->SetWeightBuffer(material.ntcWeightsBuffer);

    // Pre-fill the OutputTextureDesc array for decoding of all mip levels
    std::vector<ntc::OutputTextureDesc> outputTextureDescs;
    outputTextureDescs.resize(textureCount);
    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        TextureTranscodeTask& transcodeTask = material.transcodeMapping[textureIndex];
        ntc::OutputTextureDesc& outputDesc = outputTextureDescs[textureIndex];
        outputDesc.firstChannel = transcodeTask.firstChannel;
        outputDesc.numChannels = transcodeTask.numChannels;
        outputDesc.descriptorIndex = transcodeTask.mipZeroDescriptor;
        outputDesc.rgbColorSpace = transcodeTask.sRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
        outputDesc.ditherScale = 1.f / 255.f;
    }

    uint32_t successfulMipLevelMask = 0;
    for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
    {
        // Obtain the description of the decompression pass from LibNTC.
        // The description includes the shader code, weights, and constants.
        ntc::MakeDecompressionComputePassParameters decompressionParams;
        decompressionParams.textureSetMetadata = textureSetMetadata;
        decompressionParams.mipLevel = mipLevel;
        // This index is added to descriptorIndex from the outputTextureDescs array, so the indexing math works out
        decompressionParams.firstOutputDescriptorIndex = mipLevel;
        decompressionParams.firstLatentMipInTexture = g_firstLatentMipInTexture;
        decompressionParams.pOutputTextures = outputTextureDescs.data();
        decompressionParams.numOutputTextures = int(outputTextureDescs.size());
        decompressionParams.weightType = ntc::InferenceWeightType(material.weightType);
        ntc::ComputePassDesc decompressionPass;
        ntc::Status ntcStatus = m_ntcContext->MakeDecompressionComputePass(decompressionParams, &decompressionPass);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Failed to make a decompression pass for material '%s' mip %d, error code = %s: %s",
                material.name.c_str(), mipLevel, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            continue;
        }

        // Execute the compute pass to decompress the texture.
        // Note: ExecuteComputePass is application code (not LibNTC) and it caches PSOs based on shader code pointers.
        m_graphicsDecompressionPass->ExecuteComputePass(commandList, decompressionPass);

        successfulMipLevelMask |= (1 << mipLevel);
    }

    // Phase 3 - Compress all mips of the color textures into BCn, where necessary

    for (TextureTranscodeTask const& transcodeTask : material.transcodeMapping)
    {
        if (!transcodeTask.compressed)
            continue;

        float const alphaThreshold = 1.f / 255.f;
        
        for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
        {
            if ((successfulMipLevelMask & (1 << mipLevel)) == 0)
                continue; // This mip level was not decompressed successfully, skip BCn compression

            int const mipWidth = std::max(textureSetDesc.width >> mipLevel, 1);
            int const mipHeight = std::max(textureSetDesc.height >> mipLevel, 1);

            // Obtain the description of the BC compression pass from LibNTC.
            ntc::MakeBlockCompressionComputePassParameters compressionParams;
            compressionParams.srcRect.width = mipWidth;
            compressionParams.srcRect.height = mipHeight;
            compressionParams.dstFormat = transcodeTask.bcFormat;
            compressionParams.alphaThreshold = alphaThreshold;
            nvrhi::IBuffer* modeBuffer = nullptr;
            if (transcodeTask.bc7ModeBuffer && transcodeTask.bc7ModeBufferMipRanges[mipLevel].byteSize != 0)
            {
                compressionParams.modeBufferSource = ntc::BlockCompressionModeBufferSource::TextureSet;
                compressionParams.modeBufferByteOffset = transcodeTask.bc7ModeBufferMipRanges[mipLevel].byteOffset;
                compressionParams.modeBufferInfo.textureSet.texture = transcodeTask.metadata;
                compressionParams.modeBufferInfo.textureSet.mipLevel = mipLevel;
                modeBuffer = transcodeTask.bc7ModeBuffer;
            }
            else
            {
                compressionParams.modeBufferSource = ntc::BlockCompressionModeBufferSource::None;
            }
            ntc::ComputePassDesc compressionPass;
            ntc::Status ntcStatus = m_ntcContext->MakeBlockCompressionComputePass(compressionParams, &compressionPass);

            if (ntcStatus != ntc::Status::Ok)
            {
                log::warning("Failed to make a block compression pass for material '%s', error code = %s: %s",
                    material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                return false;
            }

            nvrhi::Format const inputFormat = (transcodeTask.numChannels == 1)
                ? nvrhi::Format::R8_UNORM
                : nvrhi::Format::RGBA8_UNORM;

            // Execute the compute pass to compress the texture.
            // Note: ExecuteComputePass is application code (not LibNTC) and it caches PSOs based on shader code pointers.
            if (!m_graphicsBlockCompressionPass->ExecuteComputePass(commandList, compressionPass,
                transcodeTask.color, inputFormat, mipLevel, modeBuffer, transcodeTask.blocks, 0))
                return false;
            
            int const mipWidthBlocks = (mipWidth + 3) / 4;
            int const mipHeightBlocks = (mipHeight + 3) / 4;

            commandList->copyTexture(transcodeTask.compressed, nvrhi::TextureSlice().setMipLevel(mipLevel),
                transcodeTask.blocks, nvrhi::TextureSlice().setWidth(mipWidthBlocks).setHeight(mipHeightBlocks));
        }
    }
    
    commandList->close();
    m_device->executeCommandList(commandList);

    // Cleanup - release the intermediate textures

    for (TextureTranscodeTask& transcodeTask : material.transcodeMapping)
    {
        transcodeTask.ReleaseTextures();
    }

    // Clear the binding set caches to avoid storing binding sets for every texture after on-load transcoding
    m_graphicsBlockCompressionPass->ClearBindingSetCache();
    m_graphicsDecompressionPass->ClearBindingSetCache();

    // We use custom texture packing that puts metalness and roughness into one NTC "texture"
    // with Metalness in R channel and Roughness in G channel.
    // Note: Only set this flag when Inference on Load is active, otherwise we get rendering corruption
    // because reference materials store ORM in that order.
    material.metalnessInRedChannel = true;

    return true;
}

static void GetNtcDataSourceAndChannelMap(NtcMaterial const& material,
    donut::engine::FilePathOrInlineData& ntcData, MaterialChannelMap& channelMap)
{
    // Initialize the channel shuffle map with default texture values.
    // Most textures have "1.0" as the no-effect value, except for the normal map, for which it is (0.5, 0.5, 1.0)
    // See also DefaultMaterialTextures() in <donut/shaders/scene_material.hlsli>
    channelMap.swizzle.fill(ntc::ShuffleSource::Constant(1.f));
    channelMap.swizzle[CHANNEL_NORMAL + 0] = ntc::ShuffleSource::Constant(0.5f);
    channelMap.swizzle[CHANNEL_NORMAL + 1] = ntc::ShuffleSource::Constant(0.5f);

    auto processMaterialTexture = [&ntcData, &channelMap, &material](std::shared_ptr<donut::engine::LoadedTexture> const& texture,
        int ch0, int ch1 = -1, int ch2 = -1, int ch3 = -1)
    {
        int destChannels[4] = { ch0, ch1, ch2, ch3 };
        if (!texture)
            return;
        
        // Find a swizzle option that refers to an NTC file or buffer.
        // Currently, there are no GLTF models that have multiple options, but there might be ones
        // that have e.g. NTC and EXR.
        for (auto const& swizzleOption : texture->swizzleOptions)
        {
            bool const isNtcFile = string_utils::ends_with(swizzleOption.source.path, ".ntc");
            bool const isNtcBuffer = swizzleOption.source.data &&
                swizzleOption.source.data->mimeType == "image/vnd-nvidia.ntc";

            if (!isNtcFile && !isNtcBuffer)
                continue;

            if (!ntcData)
            {
                ntcData = swizzleOption.source;
            }
            else if (ntcData != swizzleOption.source)
            {
                log::warning("Material %s uses different NTC sources for different textures, which is not supported.",
                    material.name.c_str());
            }

            for (int ch = 0; ch < 4; ++ch)
            {
                if (ch < swizzleOption.numChannels)
                {
                    // Copy the source channel index, but preserve -1 so that mappings containing that could be merged
                    int sourceChannel = swizzleOption.channels[ch];
                    int destChannel = destChannels[ch];
                    if (sourceChannel >= 0 && destChannel >= 0)
                    {
                        channelMap.swizzle[destChannel] = ntc::ShuffleSource::Channel(sourceChannel);
                    }
                }
            }

        }
    };

    processMaterialTexture(material.baseOrDiffuseTexture,
        CHANNEL_BASE_COLOR, CHANNEL_BASE_COLOR+1, CHANNEL_BASE_COLOR+2, CHANNEL_OPACITY);

    processMaterialTexture(material.metalRoughOrSpecularTexture,
        CHANNEL_OCCLUSION, CHANNEL_ROUGHNESS, CHANNEL_METALNESS);

    processMaterialTexture(material.occlusionTexture,
        CHANNEL_OCCLUSION);

    processMaterialTexture(material.normalTexture,
        CHANNEL_NORMAL, CHANNEL_NORMAL+1, CHANNEL_NORMAL+2);

    processMaterialTexture(material.emissiveTexture,
        CHANNEL_EMISSIVE, CHANNEL_EMISSIVE+1, CHANNEL_EMISSIVE+2);

    processMaterialTexture(material.transmissionTexture,
        CHANNEL_TRANSMISSION);
}

bool NtcMaterialLoader::PrepareMaterialForInferenceOnSample(ntc::IStream* ntcFile, 
    ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList)
{
    ntc::InferenceWeightType weightType;
    if (m_coopVec && textureSetMetadata->IsInferenceWeightTypeSupported(ntc::InferenceWeightType::CoopVecFP8))
        weightType = ntc::InferenceWeightType::CoopVecFP8;
    else
        weightType = ntc::InferenceWeightType::GenericInt8;

    ntc::InferenceData inferenceData;
    ntc::Status ntcStatus = m_ntcContext->MakeInferenceData(textureSetMetadata, weightType,
        g_firstLatentMipInTexture, &inferenceData);

    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Failed to make inference data for material '%s', error code = %s: %s",
            material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    void const* weightData = nullptr;
    size_t weightSize = 0;
    size_t convertedWeightSize = 0;
    ntcStatus = textureSetMetadata->GetInferenceWeights(weightType, &weightData, &weightSize, &convertedWeightSize);

    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Failed to get inference weights for material '%s', error code = %s: %s",
            material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    nvrhi::BufferDesc constantBufferDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(inferenceData.constants))
        .setIsConstantBuffer(true)
        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
        .setKeepInitialState(true)
        .setDebugName(material.name + " constants");
    material.ntcConstantBuffer = m_device->createBuffer(constantBufferDesc);
    if (!material.ntcConstantBuffer)
        return false;

    nvrhi::BufferDesc weightBufferDesc = nvrhi::BufferDesc()
        .setByteSize(convertedWeightSize ? convertedWeightSize : weightSize)
        .setCanHaveRawViews(true)
        .setCanHaveUAVs(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName(material.name + " weights");
    material.ntcWeightsBuffer = m_device->createBuffer(weightBufferDesc);
    if (!material.ntcWeightsBuffer)
        return false;

    ntc::LatentTextureDesc const latentTextureDescSrc = textureSetMetadata->GetLatentTextureDesc();

    nvrhi::TextureDesc latentTextureDesc = nvrhi::TextureDesc()
        .setDebugName(material.name + " latents")
        .setDimension(nvrhi::TextureDimension::Texture2DArray)
        .setFormat(nvrhi::Format::BGRA4_UNORM)
        .setWidth(std::max(latentTextureDescSrc.width >> g_firstLatentMipInTexture, 1))
        .setHeight(std::max(latentTextureDescSrc.height >> g_firstLatentMipInTexture, 1))
        .setArraySize(latentTextureDescSrc.arraySize)
        .setMipLevels(std::max(latentTextureDescSrc.mipLevels - g_firstLatentMipInTexture, 1))
        .setInitialState(nvrhi::ResourceStates::Common);
        // Note: no keepInitialState! See comments in FillTextureLoadingTasksForLatents(...) for more info.
        
    material.ntcLatentsTexture = m_device->createTexture(latentTextureDesc);
    if (!material.ntcLatentsTexture)
        return false;

    std::vector<TextureSubresourceLoadingTask> tasks;
    size_t compressedBufferSize = 0;
    size_t decompressedBufferSize = 0;
    FillTextureLoadingTasksForLatents(textureSetMetadata, material.ntcLatentsTexture, g_firstLatentMipInTexture, tasks,
        m_gdeflateFeatures && m_gdeflateFeatures->gpuDecompressionSupported, m_device->getGraphicsAPI(),
        compressedBufferSize, decompressedBufferSize);
    
    if (!ExecuteTextureLoadingTasks(m_device, commandList, m_ntcContext, ntcFile, m_gdeflateFeatures.get(), tasks,
        compressedBufferSize, decompressedBufferSize))
        return false;

    commandList->open();
    commandList->writeBuffer(material.ntcConstantBuffer, &inferenceData.constants,
        sizeof(inferenceData.constants));

    if (convertedWeightSize != 0)
    {
        assert(m_weightUploadBuffer->getDesc().byteSize >= weightSize);
        commandList->writeBuffer(m_weightUploadBuffer, weightData, weightSize);

        commandList->setBufferState(m_weightUploadBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->setBufferState(material.ntcWeightsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();

        bool const isVulkan = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN;
        nvrhi::ObjectType const commandListType = isVulkan
            ? nvrhi::ObjectTypes::VK_CommandBuffer
            : nvrhi::ObjectTypes::D3D12_GraphicsCommandList;
        nvrhi::ObjectType const bufferType = isVulkan
            ? nvrhi::ObjectTypes::VK_Buffer
            : nvrhi::ObjectTypes::D3D12_Resource;

        void* nativeCommandList = commandList->getNativeObject(commandListType);
        void* nativeSrcBuffer = m_weightUploadBuffer->getNativeObject(bufferType);
        void* nativeDstBuffer = material.ntcWeightsBuffer->getNativeObject(bufferType);

        textureSetMetadata->ConvertInferenceWeights(weightType, nativeCommandList,
            nativeSrcBuffer, 0, nativeDstBuffer, 0);
    }
    else
    {
        commandList->writeBuffer(material.ntcWeightsBuffer, weightData, weightSize);
    }
    commandList->close();
    m_device->executeCommandList(commandList);

    if (material.baseOrDiffuseTexture)
        material.baseOrDiffuseTexture->texture = m_dummyTexture->texture;
    if (material.metalRoughOrSpecularTexture)
        material.metalRoughOrSpecularTexture->texture = m_dummyTexture->texture;
    if (material.normalTexture)
        material.normalTexture->texture = m_dummyTexture->texture;
    if (material.occlusionTexture)
        material.occlusionTexture->texture = m_dummyTexture->texture;
    if (material.emissiveTexture)
        material.emissiveTexture->texture = m_dummyTexture->texture;
    if (material.transmissionTexture)
        material.transmissionTexture->texture = m_dummyTexture->texture;

    size_t const constantBufferSize = m_device->getBufferMemoryRequirements(material.ntcConstantBuffer).size;
    size_t const weightsBufferSize = m_device->getBufferMemoryRequirements(material.ntcWeightsBuffer).size;
    size_t const latentsTextureSize = m_device->getTextureMemoryRequirements(material.ntcLatentsTexture).size;

    material.ntcMemorySize = constantBufferSize + weightsBufferSize + latentsTextureSize;
    
    material.weightType = int(weightType);
    ++m_weightTypeHistogram[int(weightType)];

    return true;
}

bool NtcMaterialLoader::PrepareFeedbackMaterial(std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager,
    ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, bool enableBlockCompression)
{
    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    for (TextureTranscodeTask& transcodeTask : material.transcodeMapping)
    {
         nvrhi::TextureDesc compressedTextureDesc = nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(textureSetDesc.width)
            .setHeight(textureSetDesc.height)
            .setMipLevels(textureSetDesc.mips)
            .setDebugName(transcodeTask.name)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        if (enableBlockCompression)
        {
            compressedTextureDesc.setFormat(transcodeTask.nvrhiBcFormat);
        }
        else
        {
            compressedTextureDesc.setFormat(transcodeTask.sRGB ? nvrhi::Format::SRGBA8_UNORM : nvrhi::Format::RGBA8_UNORM);
        }

        nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> feedbackTexture;
        feedbackManager->CreateTexture(compressedTextureDesc, &feedbackTexture);
        material.*transcodeTask.pFeedbackTexture = feedbackTexture;
    }

    return true;
}

bool NtcMaterialLoader::LoadMaterialsForScene(donut::engine::Scene& scene, std::filesystem::path const& materialDir, 
    bool enableInferenceOnLoad, bool enableBlockCompression, bool enableInferenceOnSample,
    bool enableInferenceOnFeedback, std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager)
{
    using namespace std::chrono;
    time_point start = steady_clock::now();

    uint64_t totalFileSize = 0;
    uint64_t totalPixels = 0;
    int materialCount = 0;
    m_weightTypeHistogram.fill(0);

    std::vector<std::shared_ptr<engine::Material>> materials;
    for (std::shared_ptr<engine::Material> const& material : scene.GetSceneGraph()->GetMaterials())
        materials.push_back(material);

    std::unordered_map<std::string, std::shared_ptr<NtcMaterial>> ntcMaterialCache; // ntcData.ToString() -> NtcMaterial

    for (std::shared_ptr<engine::Material> const& material : materials)
    {
        std::shared_ptr<NtcMaterial> ntcMaterial = std::static_pointer_cast<NtcMaterial>(material);

        fs::path modelFileName = material->modelFileName;
        fs::path currentMaterialDir = materialDir.empty() ? modelFileName.parent_path() : materialDir;

        // Find a single NTC file that contains channels for all textures for this material.
        // In a general case, there may be multiple different files used in a single material,
        // but our GLTF preparation script only uses one, and we only support one in this renderer.
        // Also derive the channel mapping: where to source the channels for albedo, roughness, etc.
        donut::engine::FilePathOrInlineData ntcData;
        MaterialChannelMap channelMap;
        GetNtcDataSourceAndChannelMap(*ntcMaterial, ntcData, channelMap);

        // No NTC data found, skip the material
        if (!ntcData)
            continue;

        auto cacheIterator = ntcMaterialCache.find(ntcData.ToString());
        if (cacheIterator != ntcMaterialCache.end())
        {
            auto previouslyLoadedMaterial = cacheIterator->second;

            // Copy over all the properties that we touch when decoding NTC materials,
            // but not the entire material: some flags or parameters might be different.
            ntcMaterial->ntcConstantBuffer = previouslyLoadedMaterial->ntcConstantBuffer;
            ntcMaterial->ntcWeightsBuffer = previouslyLoadedMaterial->ntcWeightsBuffer;
            ntcMaterial->ntcLatentsTexture = previouslyLoadedMaterial->ntcLatentsTexture;
            ntcMaterial->weightType = previouslyLoadedMaterial->weightType;
            ntcMaterial->baseOrDiffuseTexture = previouslyLoadedMaterial->baseOrDiffuseTexture;
            ntcMaterial->metalRoughOrSpecularTexture = previouslyLoadedMaterial->metalRoughOrSpecularTexture;
            ntcMaterial->normalTexture = previouslyLoadedMaterial->normalTexture;
            ntcMaterial->emissiveTexture = previouslyLoadedMaterial->emissiveTexture;
            ntcMaterial->occlusionTexture = previouslyLoadedMaterial->occlusionTexture;
            ntcMaterial->transmissionTexture = previouslyLoadedMaterial->transmissionTexture;
            ntcMaterial->opacityTexture = previouslyLoadedMaterial->opacityTexture;
            ntcMaterial->metalnessInRedChannel = previouslyLoadedMaterial->metalnessInRedChannel;
            ntcMaterial->baseOrDiffuseTextureFeedback = previouslyLoadedMaterial->baseOrDiffuseTextureFeedback;
            ntcMaterial->metalRoughOrSpecularTextureFeedback = previouslyLoadedMaterial->metalRoughOrSpecularTextureFeedback;
            ntcMaterial->normalTextureFeedback = previouslyLoadedMaterial->normalTextureFeedback;
            ntcMaterial->emissiveTextureFeedback = previouslyLoadedMaterial->emissiveTextureFeedback;
            ntcMaterial->occlusionTextureFeedback = previouslyLoadedMaterial->occlusionTextureFeedback;
            ntcMaterial->transmissionTextureFeedback = previouslyLoadedMaterial->transmissionTextureFeedback;
            ntcMaterial->opacityTextureFeedback = previouslyLoadedMaterial->opacityTextureFeedback;
            ntcMaterial->textureSetMetadata = previouslyLoadedMaterial->textureSetMetadata;
            ntcMaterial->transcodeMapping = previouslyLoadedMaterial->transcodeMapping;

            continue;
        }

        ntcMaterialCache[ntcData.ToString()] = ntcMaterial;

        // Use differnt wrappers for file and memory streams because they have different deleter functions...
        ntc::FileStreamWrapper ntcFileStream(m_ntcContext);
        ntc::MemoryStreamWrapper ntcMemoryStream(m_ntcContext);

        ntcMaterial->textureSetMetadata = std::make_shared<ntc::TextureSetMetadataWrapper>(m_ntcContext);

        if (!LoadMaterialFile(ntcData, *ntcMaterial, m_ntcContext, ntcFileStream, ntcMemoryStream,
            *ntcMaterial->textureSetMetadata))
            continue;

        // Upcast the file or memory stream to a basic stream type
        ntc::IStream* const dataStream = ntcFileStream.Get() ? ntcFileStream.Get() : ntcMemoryStream.Get();

        ntc::ITextureSetMetadata* textureSetMetadata = *ntcMaterial->textureSetMetadata;

        ntc::Status ntcStatus = textureSetMetadata->ShuffleInferenceOutputs(channelMap.swizzle.data());
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Cannot process material '%s', call to ShuffleInferenceOutputs failed, error code = %s: %s",
                ntcMaterial->name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        }

        // Derive the transcode mapping using the channel map and texture metadata
        bool const onlyAlphaMask = !enableInferenceOnLoad && !enableInferenceOnFeedback;
        FillMaterialTranscodeMapping(*ntcMaterial, textureSetMetadata, channelMap, onlyAlphaMask);

        // Load the material data for Inference On Sample/Feedback first, so that the latent and weight buffers
        // can be reused for On Load.
        bool loadedSuccessfully = PrepareMaterialForInferenceOnSample(dataStream, textureSetMetadata,
            *ntcMaterial, m_commandList);

        // Transcode the material into raw color data or BCn (Inference On Load).
        // When Inference on Load is disabled, we still go through the materials and extract alpha mask channels,
        // encoding them into BC4 when allowed. They are used for the depth pre-pass (or any-hit shaders
        // in a path tracing renderer).
        if (loadedSuccessfully)
        {
            loadedSuccessfully = TranscodeMaterial(m_ntcContext, dataStream,
                textureSetMetadata, *ntcMaterial, m_commandList, enableBlockCompression);
        }

        if (enableInferenceOnFeedback && loadedSuccessfully)
        {
            loadedSuccessfully = PrepareFeedbackMaterial(feedbackManager, textureSetMetadata, *ntcMaterial,
                enableBlockCompression);
        }
        
        if (loadedSuccessfully)
        {
            m_device->waitForIdle();
            m_device->runGarbageCollection();
        }

        auto const& textureSetDesc = textureSetMetadata->GetDesc();
        totalFileSize += dataStream->Size();
        totalPixels += (textureSetDesc.width * textureSetDesc.height * 4) / 3;
        ++materialCount;
    }

    time_point end = steady_clock::now();
    int64_t durationMs = duration_cast<milliseconds>(end - start).count();
    
    log::info("%d materials loaded in %lli ms - that's %.2f Mpix from %.2f MB", materialCount, durationMs,
        double(totalPixels) * 1e-6, double(totalFileSize) * 0x1p-20);

    return true;
}
