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

#include <string>
#include <vector>
#include <optional>
#include <libntc/ntc.h>

enum class SemanticLabel
{
    None = 0,

    Albedo,
    AlphaMask,
    Displacement,
    Emissive,
    Glossiness,
    Metalness,
    Normal,
    Occlusion,
    Roughness,
    SpecularColor,
    Transmission,
    // Note: Update ParseSemanticLabel, SemanticLabelToString, GetSemanticChannelCount
    // when adding semantic labels here. Keep the enum labels sorted (no technical reason, just style).

    Count
};

struct ImageSemanticBinding
{
    SemanticLabel label = SemanticLabel::None;
    int firstChannel = 0;
};

struct ManifestEntry
{
    std::string fileName;
    std::string entryName;
    std::string channelSwizzle;
    std::vector<ImageSemanticBinding> semantics;
    int mipLevel = 0;
    int firstChannel = -1;
    bool isSRGB = false;
    bool verticalFlip = false;
    ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
    std::vector<float> lossFunctionScales;
};

struct Manifest
{
    std::vector<ManifestEntry> textures;
    std::optional<int> width;
    std::optional<int> height;
};

enum class ToolInputType
{
    None,
    Directory,
    CompressedTextureSet,
    Manifest,
    Images,
    Mixed
};

constexpr ntc::BlockCompressedFormat BlockCompressedFormat_Auto = ntc::BlockCompressedFormat(999);

void LowercaseString(std::string& s);
void UppercaseString(std::string& s);

std::optional<ntc::BlockCompressedFormat> ParseBlockCompressedFormat(char const* format, bool enableAuto = false);

SemanticLabel ParseSemanticLabel(char const* label);

char const* SemanticLabelToString(SemanticLabel label);

int GetSemanticChannelCount(SemanticLabel label);

void GenerateManifestFromDirectory(const char* path, bool loadMips, Manifest& outManifest);

void GenerateManifestFromFileList(std::vector<const char*> const& files, Manifest& outManifest);
    
bool ReadManifestFromFile(const char* fileName, Manifest& outManifest,
    std::string& outError);

bool IsSupportedImageFileExtension(std::string const& extension);

void UpdateToolInputType(ToolInputType& current, ToolInputType newInput);
