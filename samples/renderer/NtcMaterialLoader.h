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

#pragma once

#include <libntc/ntc.h>
#include <nvrhi/nvrhi.h>
#include <filesystem>
#include <unordered_map>

#include "feedbackmanager/include/FeedbackManager.h"

struct NtcMaterial;
class GraphicsDecompressionPass;
class GraphicsBlockCompressionPass;

#define MAX_TILES_PER_FRAME 32
#define TRANSCODE_BATCH_SIZE 8

namespace donut::engine
{
    struct LoadedTexture;
    class Scene;
}

struct TranscodeTileInfo
{
    NtcMaterial* material;
    nvfeedback::FeedbackTextureTileInfo tileInfo;
};

namespace ntc
{
    class IContext;
    class IStream;
    class ITextureSetMetadata;
}

struct MaterialChannelMap
{
    std::array<ntc::ShuffleSource, NTC_MAX_CHANNELS> swizzle;
};

typedef std::array<int, size_t(ntc::InferenceWeightType::Count)> WeightTypeHistogram;

class NtcMaterialLoader
{
public:
    NtcMaterialLoader(nvrhi::IDevice* device)
        : m_device(device)
    { }
    
    bool Init(bool enableCoopVec, nvrhi::ITexture* dummyTexture);

    bool IsCooperativeVectorSupported() const { return m_coopVec; }

    bool LoadMaterialsForScene(donut::engine::Scene& scene, std::filesystem::path const& materialDir, 
        bool enableInferenceOnLoad, bool enableBlockCompression, bool enableInferenceOnSample,
        bool enableInferenceOnFeedback, std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager);

    bool TranscodeTiles(const std::vector<TranscodeTileInfo>& tiles, nvrhi::ICommandList* commandList,
        bool enableBlockCompression);

    WeightTypeHistogram const& GetWeightTypeHistogram() const { return m_weightTypeHistogram; }

private:
    nvrhi::DeviceHandle m_device;
    nvrhi::CommandListHandle m_commandList;

    ntc::ContextWrapper m_ntcContext;

    bool m_coopVec = false;
    WeightTypeHistogram m_weightTypeHistogram;

    std::shared_ptr<donut::engine::LoadedTexture> m_dummyTexture;

    std::shared_ptr<GraphicsDecompressionPass> m_graphicsDecompressionPass;
    std::shared_ptr<GraphicsBlockCompressionPass> m_graphicsBlockCompressionPass;

    nvrhi::BufferHandle m_weightUploadBuffer;

    // Textures for tile-based decompression and recompression
    uint32_t m_texTileColorR8Offset = 0;
    uint32_t m_texTileColorRGBAOffset = 0;
    uint32_t m_texTileBlocksRGOffset = 0;
    uint32_t m_texTileBlocksRGBAOffset = 0;
    std::vector<nvrhi::TextureHandle> m_texTranscodeTiles;

    bool TranscodeMaterial(ntc::IStream* ntcFile,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList,
        bool enableBlockCompression);

    bool PrepareMaterialForInferenceOnSample(ntc::IStream* ntcFile,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList);

    bool PrepareFeedbackMaterial(std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, bool enableBlockCompression);
};