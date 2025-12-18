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

#include "Utils.h"
#include <lodepng.h>
#include <donut/engine/ThreadPool.h>
#include <algorithm>
#include <stb_image_write.h>
#include <tinyexr.h>
#include <ntc-utils/Manifest.h>

static donut::engine::ThreadPool g_ThreadPool;

static const BcFormatDefinition c_BlockCompressedFormats[] = {
    { ntc::BlockCompressedFormat::BC1, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, nvrhi::Format::BC1_UNORM,    8, 4 },
    { ntc::BlockCompressedFormat::BC2, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM_SRGB, nvrhi::Format::BC2_UNORM,   16, 4 },
    { ntc::BlockCompressedFormat::BC3, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB, nvrhi::Format::BC3_UNORM,   16, 4 },
    { ntc::BlockCompressedFormat::BC4, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_UNORM,      nvrhi::Format::BC4_UNORM,    8, 1 },
    { ntc::BlockCompressedFormat::BC5, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_UNORM,      nvrhi::Format::BC5_UNORM,   16, 2 },
    { ntc::BlockCompressedFormat::BC6, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_UF16,      nvrhi::Format::BC6H_UFLOAT, 16, 3 },
    { ntc::BlockCompressedFormat::BC7, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB, nvrhi::Format::BC7_UNORM,   16, 4 },
};

BcFormatDefinition const* GetBcFormatDefinition(ntc::BlockCompressedFormat format)
{
    for (const auto& formatCandidate : c_BlockCompressedFormats)
    {
        if (formatCandidate.ntcFormat == format)
        {
            return &formatCandidate;
        }
    }
    assert(false);
    return nullptr;
}


float Median(std::vector<float>& items)
{
    size_t middleIndex = items.size() / 2;
    std::nth_element(items.begin(), items.begin() + middleIndex, items.end());
    return items[middleIndex];
}

bool WriteDdsHeader(ntc::IStream* ddsFile, int width, int height, int mipLevels, BcFormatDefinition const* outputFormatDefinition, ntc::ColorSpace colorSpace)
{
    using namespace donut::engine::dds;
    DDS_HEADER ddsHeader{};
    DDS_HEADER_DXT10 dx10header = {};
    ddsHeader.size = sizeof(DDS_HEADER);
    ddsHeader.flags = DDS_HEADER_FLAGS_TEXTURE;
    ddsHeader.width = width;
    ddsHeader.height = height;
    ddsHeader.depth = 1;
    ddsHeader.mipMapCount = mipLevels;
    ddsHeader.ddspf.size = sizeof(DDS_PIXELFORMAT);
    ddsHeader.ddspf.flags = DDS_FOURCC;
    ddsHeader.ddspf.fourCC = MAKEFOURCC('D', 'X', '1', '0');
    dx10header.resourceDimension = DDS_DIMENSION_TEXTURE2D;
    dx10header.arraySize = 1;
    dx10header.dxgiFormat = colorSpace == ntc::ColorSpace::sRGB ? outputFormatDefinition->dxgiFormatSrgb : outputFormatDefinition->dxgiFormat;

    uint32_t ddsMagic = DDS_MAGIC;
    bool success;
    success = ddsFile->Write(&ddsMagic, sizeof(ddsMagic));
    success &= ddsFile->Write(&ddsHeader, sizeof(ddsHeader));
    success &= ddsFile->Write(&dx10header, sizeof(dx10header));
    return success;
}

bool SavePNG(uint8_t* data, int mipWidth, int mipHeight, int numChannels, bool is16Bit, char const* fileName)
{
    // Use LodePNG to save PNG's instead of STB.
    // It can write 16-bit-per-channel images and extended metadata.
    
    // LodePNG expects 16-bit data in big endian format, so byte-swap it.
    if (is16Bit)
    {
        for (int offset = 0; offset < mipWidth * mipHeight * numChannels * 2; offset += 2)
        {
            std::swap(data[offset], data[offset + 1]);
        }
    }

    // Prepare input parameters
    LodePNGColorType const colorType =
        numChannels == 4 ? LCT_RGBA :
        numChannels == 3 ? LCT_RGB :
        numChannels == 2 ? LCT_GREY_ALPHA :
        LCT_GREY;
    unsigned const bitDepth = is16Bit ? 16 : 8;

    // Fill out LodePNGState
    // Note: extra info like color profile can also go here.
    LodePNGState state;
    lodepng_state_init(&state);
    state.info_raw.colortype = colorType;
    state.info_raw.bitdepth = bitDepth;
    state.info_png.color.colortype = colorType;
    state.info_png.color.bitdepth = bitDepth;
    state.encoder.zlibsettings.windowsize = 512; // slightly worse compression but much faster, default = 2048

    // Encode the PNG
    unsigned char* pngData = nullptr;
    size_t pngSize = 0;
    lodepng_encode(&pngData, &pngSize, data, mipWidth, mipHeight, &state);
    bool success = state.error == 0;

    lodepng_state_cleanup(&state);

    if (success)
    {
        // Save the PNG data into the output file
        FILE* outputFile = fopen(fileName, "wb");
        if (outputFile)
        {
            if (fwrite(pngData, pngSize, 1, outputFile) != 1)
                success = false;
            fclose(outputFile);
        }
        else
            success = false;
    }

    if (pngData)
        free(pngData);

    return success;
}

void StartAsyncTask(std::function<void()> function)
{
    g_ThreadPool.AddTask(function);
}

void WaitForAllTasks()
{
    g_ThreadPool.WaitForTasks();
}

std::optional<ImageContainer> ParseImageContainer(char const* container)
{
    if (!container || !container[0])
        return ImageContainer::Auto;

    std::string uppercaseContainer = container;
    UppercaseString(uppercaseContainer);
    
    if (uppercaseContainer == "AUTO")
        return ImageContainer::Auto;
    if (uppercaseContainer == "BMP")
        return ImageContainer::BMP;
    if (uppercaseContainer == "EXR")
        return ImageContainer::EXR;
    if (uppercaseContainer == "JPG" || uppercaseContainer == "JPEG" )
        return ImageContainer::JPG;
    if (uppercaseContainer == "PNG")
        return ImageContainer::PNG;
    if (uppercaseContainer == "PNG16")
        return ImageContainer::PNG16;
    if (uppercaseContainer == "TGA")
        return ImageContainer::TGA;

    return std::optional<ImageContainer>();
}

ntc::ChannelFormat GetContainerChannelFormat(ImageContainer container)
{
    switch(container)
    {
    case ImageContainer::Auto:
    default:
        return ntc::ChannelFormat::UNKNOWN;
    case ImageContainer::BMP:
    case ImageContainer::JPG:
    case ImageContainer::PNG:
    case ImageContainer::TGA:
        return ntc::ChannelFormat::UNORM8;
    case ImageContainer::EXR:
        return ntc::ChannelFormat::FLOAT32;
    case ImageContainer::PNG16:
        return ntc::ChannelFormat::UNORM16;
    }
}

char const* GetContainerExtension(ImageContainer container)
{
    switch(container)
    {
    case ImageContainer::Auto:
    default:
        return nullptr; // Invalid call
    case ImageContainer::BMP:
        return ".bmp";
    case ImageContainer::JPG:
        return ".jpg";
    case ImageContainer::PNG:
    case ImageContainer::PNG16:
        return ".png";
    case ImageContainer::TGA:
        return ".tga";
    case ImageContainer::EXR:
        return ".exr";
    }
}

bool SaveImageToContainer(ImageContainer container, void const* data, int width, int height, int channels, char const* fileName)
{
    switch(container)
    {
    case ImageContainer::Auto:
    default:
        return false; // Invalid call
    case ImageContainer::BMP:
        return !!stbi_write_bmp(fileName, width, height, channels, data);
    case ImageContainer::JPG:
        return !!stbi_write_jpg(fileName, width, height, channels, data, /* quality = */ 95);
    case ImageContainer::PNG:
        return SavePNG((uint8_t*)data, width, height, channels, false, fileName);
    case ImageContainer::PNG16:
        return SavePNG((uint8_t*)data, width, height, channels, true, fileName);
    case ImageContainer::TGA:
        return !!stbi_write_tga(fileName, width, height, channels, data);
    case ImageContainer::EXR:
        return SaveEXR((float const*)data, width, height, channels, /* save_as_fp16 = */ true,
            fileName, /* err = */ nullptr) == TINYEXR_SUCCESS;
    }
}
