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

#include <donut/engine/SceneGraph.h>
#include <libntc/ntc.h>

#include "feedbackmanager/include/FeedbackManager.h"

struct NtcMaterial;

struct TextureTranscodeTask
{
    ntc::ITextureMetadata const* metadata = nullptr;
    ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
    nvrhi::TextureHandle color;
    nvrhi::TextureHandle blocks;
    nvrhi::TextureHandle compressed;
    nvrhi::Format nvrhiBcFormat = nvrhi::Format::UNKNOWN;
    int firstChannel = 0;
    int numChannels = 0;
    int mipZeroDescriptor = 0;
    bool sRGB = false;
    char const* name = nullptr;
    std::shared_ptr<donut::engine::LoadedTexture> NtcMaterial::* pMaterialTexture = nullptr;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> NtcMaterial::* pFeedbackTexture = nullptr;

    void ReleaseTextures()
    {
        color = nullptr;
        blocks = nullptr;
        compressed = nullptr;
    }
};

struct NtcMaterial : public donut::engine::Material
{
    nvrhi::BufferHandle ntcConstantBuffer;
    nvrhi::BufferHandle ntcWeightsBuffer;
    nvrhi::TextureHandle ntcLatentsTexture;
    int weightType = 0;
    size_t transcodedMemorySize = 0;
    size_t ntcMemorySize = 0;

    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> baseOrDiffuseTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> metalRoughOrSpecularTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> normalTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> emissiveTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> occlusionTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> transmissionTextureFeedback;
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> opacityTextureFeedback;

    std::shared_ptr<ntc::TextureSetMetadataWrapper> textureSetMetadata;

    std::vector<TextureTranscodeTask> transcodeMapping;
};

class NtcSceneTypeFactory : public donut::engine::SceneTypeFactory
{
public:
    std::shared_ptr<donut::engine::Material> CreateMaterial() override;
};
