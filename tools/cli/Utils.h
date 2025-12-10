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
#include <functional>
#include <optional>
#if NTC_WITH_DX12
// Make sure DXGI is included before DDSHeader.h to avoid macro redefinition errors later
#include <dxgi.h>
#endif
#include <ntc-utils/DDSHeader.h>

#define CHECK_NTC_RESULT(fname) \
    if (ntcStatus != ntc::Status::Ok) { \
        fprintf(stderr, "Call to " #fname " failed, code = %s\n%s\n", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage()); \
        return false; \
    }

struct BcFormatDefinition
{
    ntc::BlockCompressedFormat ntcFormat;
    DXGI_FORMAT dxgiFormat;
    DXGI_FORMAT dxgiFormatSrgb;
    nvrhi::Format nvrhiFormat;
    int bytesPerBlock;
    int channels;
};

BcFormatDefinition const* GetBcFormatDefinition(ntc::BlockCompressedFormat format);

float Median(std::vector<float>& items);

bool WriteDdsHeader(ntc::IStream* ddsFile, int width, int height, int mipLevels,
    BcFormatDefinition const* outputFormatDefinition, ntc::ColorSpace colorSpace);

bool SavePNG(uint8_t* data, int mipWidth, int mipHeight, int numChannels, bool is16Bit, char const* fileName);

void StartAsyncTask(std::function<void()> function);

void WaitForAllTasks();

enum class ImageContainer
{
    Auto,
    BMP,
    EXR,
    JPG,
    PNG,
    PNG16,
    TGA,
};

std::optional<ImageContainer> ParseImageContainer(char const* s);
ntc::ChannelFormat GetContainerChannelFormat(ImageContainer container);
char const* GetContainerExtension(ImageContainer container);
bool SaveImageToContainer(ImageContainer container, void const* data, int width, int height, int channels, char const* fileName);
