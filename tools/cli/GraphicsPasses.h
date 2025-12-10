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

#include <nvrhi/nvrhi.h>
#include <libntc/ntc.h>
#include "Utils.h"

struct GraphicsResourcesForTexture
{
    std::string name;
    nvrhi::TextureHandle color;
    nvrhi::StagingTextureHandle stagingColor;
    nvrhi::TextureHandle blocks;
    nvrhi::StagingTextureHandle stagingBlocks;
    nvrhi::TextureHandle bc;
    ntc::SharedTextureWrapper sharedTexture;

    GraphicsResourcesForTexture(ntc::IContext* context)
        : sharedTexture(context)
    { }
};

struct GraphicsResourcesForTextureSet
{
    std::vector<GraphicsResourcesForTexture> perTexture;
};

class GraphicsDecompressionPass;
struct GDeflateFeatures;

bool CreateGraphicsResourcesFromMetadata(
    ntc::IContext* context,
    nvrhi::IDevice* device,
    ntc::ITextureSetMetadata* metadata,
    int mipLevels,
    bool enableCudaSharing,
    GraphicsResourcesForTextureSet& resources);

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
    GraphicsResourcesForTextureSet const& graphicsResources);

bool CopyTextureSetDataIntoGraphicsTextures(
    ntc::IContext* context,
    ntc::ITextureSet* textureSet,
    ntc::TextureDataPage page,
    bool allMipLevels,
    bool onlyBlockCompressedFormats,
    GraphicsResourcesForTextureSet const& graphicsResources);

bool SaveGraphicsStagingTextures(
    ntc::ITextureSetMetadata* metadata,
    nvrhi::IDevice* device,
    char const* savePath,
    ImageContainer const userProvidedContainer,
    bool saveMips,
    GraphicsResourcesForTextureSet const& graphicsResources);

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
    GraphicsResourcesForTextureSet const& graphicsResources);

bool OptimizeBlockCompression(
    ntc::IContext* context,
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    float psnrThreshold,
    GraphicsResourcesForTextureSet const& graphicsResources);

bool ComputePsnrForBlockCompressedTextureSet(
    ntc::IContext* context,
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    GraphicsResourcesForTextureSet const& graphicsResources,
    float& outTargetPsnr);