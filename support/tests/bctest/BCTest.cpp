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

#include <libntc/ntc.h>
#include <argparse.h>
#include <stb_image.h>
#include <tinyexr.h>
#include <donut/app/DeviceManager.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/utils.h>
#include <cmath>
#include <csignal>
#include <queue>
#include <thread>
#include <fstream>
#include <numeric>
#include <filesystem>
#include <ntc-utils/GraphicsImageDifferencePass.h>
#include <ntc-utils/GraphicsBlockCompressionPass.h>
#include <ntc-utils/Manifest.h>
#include <ntc-utils/Misc.h>
#include <ntc-utils/DDSHeader.h>

#if NTC_WITH_NVTT
#include <nvtt/nvtt.h>
#endif

namespace fs = std::filesystem;

struct
{
    const char* sourcePath = nullptr;
    const char* format = nullptr;
    const char* outputPath = nullptr;
    const char* csvOutputPath = nullptr;
    const char* loadBaselinePath = nullptr;
    bool useVulkan = false;
    bool useDX12 = false;
    bool debug = false;
    bool ntc = true;
#if NTC_WITH_NVTT
    bool nvtt = true;
#endif
    bool acceleratedMode = false;
    int adapterIndex = -1;
    int threads = 0;
} g_options;

bool ProcessCommandLine(int argc, const char** argv)
{
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING(0, "source", &g_options.sourcePath, "Load source images from this path recursively"),
        OPT_STRING(0, "output", &g_options.outputPath, "Save compressed DDS images into this path"),
        OPT_STRING(0, "csv", &g_options.csvOutputPath, "Save a summary table in CSV to this file"),
        OPT_STRING(0, "loadBaseline", &g_options.loadBaselinePath, "Load previous results from a CSV file for comparison"),
        OPT_STRING(0, "format", &g_options.format, "Compression format, BC1-BC7"),
#if DONUT_WITH_VULKAN
        OPT_BOOLEAN(0, "vk", &g_options.useVulkan, "Use Vulkan API"),
#endif
#if DONUT_WITH_DX12
        OPT_BOOLEAN(0, "dx12", &g_options.useDX12, "Use D3D12 API"),
#endif
        OPT_BOOLEAN(0, "ntc", &g_options.ntc, "Enable BCn compression through NTC (default on, use --no-ntc)"),
#if NTC_WITH_NVTT
        OPT_BOOLEAN(0, "nvtt", &g_options.nvtt, "Enable BCn compression through NVTT (default on, use --no-nvtt)"),
#endif
        OPT_BOOLEAN(0, "accelerated", &g_options.acceleratedMode, "Test NTC accelerated mode for BC7 compression"),
        OPT_BOOLEAN(0, "debug", &g_options.debug, "Enable debug features such as Vulkan validation layer or D3D12 debug runtime"),
        OPT_INTEGER(0, "adapter", &g_options.adapterIndex, "Index of the graphics adapter to use"),
        OPT_INTEGER(0, "threads", &g_options.threads, "Number of threads to use for preloading images"),
        OPT_END()
    };

    static const char* usages[] = {
        "bctest.exe --source <path> --format <BCn> [options...]",
        nullptr
    };

    struct argparse argparse {};
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "\nBCn compression test tool.", nullptr);
    argparse_parse(&argparse, argc, argv);

    if (!g_options.useVulkan && !g_options.useDX12)
    {
        #if DONUT_WITH_VULKAN
        g_options.useVulkan = true;
        #else
        g_options.useDX12 = true;
        #endif
        assert(g_options.useDX12 || g_options.useVulkan);
    }

    if (!g_options.sourcePath)
    {
        fprintf(stderr, "--source is required.\n");
        return false;
    }
    
    if (!fs::is_directory(g_options.sourcePath))
    {
        fprintf(stderr, "Input path '%s' does not exist or is not a directory.\n", g_options.sourcePath);
        return false;
    }

    if (!g_options.format)
    {
        fprintf(stderr, "--format is required.\n");
        return false;
    }

    auto parsedFormat = ParseBlockCompressedFormat(g_options.format);
    if (!parsedFormat.has_value() || parsedFormat.value() == ntc::BlockCompressedFormat::None)
    {
        fprintf(stderr, "Invalid --format value '%s'.\n", g_options.format);
        return false;
    }

    if (g_options.acceleratedMode && (parsedFormat != ntc::BlockCompressedFormat::BC7 || !g_options.ntc))
    {
        fprintf(stderr, "--accelerated can only be used with BC7 format when NTC is used.\n");
        return false;
    }

    return true;
}

#define CHECK_NTC_RESULT(fname) \
    if (ntcStatus != ntc::Status::Ok) { \
        fprintf(stderr, "Call to " #fname " failed, code = %s\n%s\n", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage()); \
        return false; \
    }

bool g_Terminate = false;
void SigintHandler(int signal)
{
    printf("\nSIGINT received, stopping...\n\n");
    g_Terminate = true;
}

struct BcFormatDefinition
{
    ntc::BlockCompressedFormat ntcFormat;
    DXGI_FORMAT dxgiFormat;
    DXGI_FORMAT dxgiFormatSrgb;
    nvrhi::Format nvrhiFormat;
    nvrhi::Format blockFormat;
    int bytesPerBlock;
    int channels;
#if NTC_WITH_NVTT
    nvtt::Format nvttFormat;
    nvtt::ValueType nvttValueType;
#endif
};

#if NTC_WITH_NVTT
#define NVTT_FORMATS(x, y) x, y
#else
#define NVTT_FORMATS(x, y)
#endif

static const BcFormatDefinition c_BlockCompressedFormats[] = {
    { ntc::BlockCompressedFormat::BC1, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, nvrhi::Format::BC1_UNORM,   nvrhi::Format::RG32_UINT,      8, 4, NVTT_FORMATS(nvtt::Format::Format_BC1a, nvtt::ValueType::UINT8) },
    { ntc::BlockCompressedFormat::BC2, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM_SRGB, nvrhi::Format::BC2_UNORM,   nvrhi::Format::RGBA32_UINT,   16, 4, NVTT_FORMATS(nvtt::Format::Format_BC2,  nvtt::ValueType::UINT8) },
    { ntc::BlockCompressedFormat::BC3, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB, nvrhi::Format::BC3_UNORM,   nvrhi::Format::RGBA32_UINT,   16, 4, NVTT_FORMATS(nvtt::Format::Format_BC3,  nvtt::ValueType::UINT8) },
    { ntc::BlockCompressedFormat::BC4, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_UNORM,      nvrhi::Format::BC4_UNORM,   nvrhi::Format::RG32_UINT,      8, 1, NVTT_FORMATS(nvtt::Format::Format_BC4,  nvtt::ValueType::UINT8) },
    { ntc::BlockCompressedFormat::BC5, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_UNORM,      nvrhi::Format::BC5_UNORM,   nvrhi::Format::RGBA32_UINT,   16, 2, NVTT_FORMATS(nvtt::Format::Format_BC5,  nvtt::ValueType::UINT8) },
    { ntc::BlockCompressedFormat::BC6, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_UF16,      nvrhi::Format::BC6H_UFLOAT, nvrhi::Format::RGBA32_UINT,   16, 3, NVTT_FORMATS(nvtt::Format::Format_BC6U, nvtt::ValueType::FLOAT32) },
    { ntc::BlockCompressedFormat::BC7, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB, nvrhi::Format::BC7_UNORM,   nvrhi::Format::RGBA32_UINT,   16, 4, NVTT_FORMATS(nvtt::Format::Format_BC7,  nvtt::ValueType::UINT8) },
};

BcFormatDefinition const* GetFormatDef(ntc::BlockCompressedFormat format)
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

donut::app::DeviceCreationParameters GetGraphicsDeviceParameters()
{
    donut::app::DeviceCreationParameters deviceParams;
    deviceParams.infoLogSeverity = donut::log::Severity::None;
    deviceParams.adapterIndex = g_options.adapterIndex;
    deviceParams.enableDebugRuntime = g_options.debug;
    deviceParams.enableNvrhiValidationLayer = g_options.debug;
    return deviceParams;
}

std::unique_ptr<donut::app::DeviceManager> InitGraphicsDevice()
{
    using namespace donut::app;

    nvrhi::GraphicsAPI const graphicsApi = g_options.useVulkan
        ? nvrhi::GraphicsAPI::VULKAN
        : nvrhi::GraphicsAPI::D3D12;
    
    auto deviceManager = std::unique_ptr<DeviceManager>(DeviceManager::Create(graphicsApi));

    DeviceCreationParameters const deviceParams = GetGraphicsDeviceParameters();

    if (!deviceManager->CreateHeadlessDevice(deviceParams))
    {
        fprintf(stderr, "Cannot initialize a %s device.\n", nvrhi::utils::GraphicsAPIToString(graphicsApi));
        return nullptr;
    }

    printf("Using %s with %s API.\n", deviceManager->GetRendererString(), nvrhi::utils::GraphicsAPIToString(graphicsApi));

    return std::move(deviceManager);
}

bool InitNtcContext(nvrhi::IDevice* device, ntc::ContextWrapper& context)
{
    // Initialize the NTC context with the graphics device
    ntc::ContextParameters contextParams;
    contextParams.graphicsApi = device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12
        ? ntc::GraphicsAPI::D3D12
        : ntc::GraphicsAPI::Vulkan;

    contextParams.d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    contextParams.vkInstance = device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
    contextParams.vkPhysicalDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
    contextParams.vkDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_Device);

    ntc::Status ntcStatus = ntc::CreateContext(context.ptr(), contextParams);
    if (ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable)
    {
        fprintf(stderr, "Failed to create an NTC context, code = %s: %s\n",
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    return true;
}

std::vector<fs::path> EnumerateSourceFiles()
{
    std::vector<fs::path> result;
    for (auto entry : fs::recursive_directory_iterator(g_options.sourcePath))
    {
        if (!entry.is_regular_file())
            continue;

        std::string extension = entry.path().extension().string();
        LowercaseString(extension);

        if (extension != ".png" && extension != ".jpg" && extension != ".tga" && extension != ".exr")
            continue;

        result.push_back(entry.path());
    }
    return result;
}

struct ImageData
{
    int width = 0;
    int height = 0;
    int widthInBlocks = 0;
    int heightInBlocks = 0;
    int channels = 0;
    bool isHDR = false;
    stbi_uc* data = nullptr;
    fs::path name;

    nvrhi::TextureHandle originalTexture;
    nvrhi::TextureHandle blockTexture;
    nvrhi::TextureHandle compressedTexture;
    nvrhi::BufferHandle modeBuffer;
    nvrhi::StagingTextureHandle stagingTexture;
    
    ImageData()
    { }

    ImageData(ImageData& other) = delete;
    ImageData(ImageData&& other) = delete;

    ~ImageData()
    {
        if (data)
            stbi_image_free((void*)data);
        data = nullptr;
    }

    bool InitTextures(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, BcFormatDefinition const& formatDef)
    {
        nvrhi::TextureDesc originalTextureDesc = nvrhi::TextureDesc()
            .setDebugName(name.generic_string())
            .setWidth(width)
            .setHeight(height)
            .setFormat(isHDR ? nvrhi::Format::RGBA32_FLOAT : nvrhi::Format::RGBA8_UNORM)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true);
        originalTexture = device->createTexture(originalTextureDesc);
        if (!originalTexture)
            return false;

        nvrhi::TextureDesc blockTextureDesc = nvrhi::TextureDesc()
            .setDebugName("Block Texture")
            .setWidth(widthInBlocks)
            .setHeight(heightInBlocks)
            .setFormat(formatDef.blockFormat)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true);
        blockTexture = device->createTexture(blockTextureDesc);
        if (!blockTexture)
            return false;

        nvrhi::TextureDesc compressedTextureDesc = nvrhi::TextureDesc()
            .setDebugName("Compressed Texture")
            .setWidth(width)
            .setHeight(height)
            .setFormat(formatDef.nvrhiFormat)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true);
        compressedTexture = device->createTexture(compressedTextureDesc);
        if (!compressedTexture)
            return false;

        nvrhi::TextureDesc stagingTextureDesc = blockTextureDesc;
        stagingTextureDesc.setIsUAV(false);
        stagingTexture = device->createStagingTexture(stagingTextureDesc, nvrhi::CpuAccessMode::Read);
        if (!stagingTexture)
            return false;

        size_t const bytesPerPixel = isHDR ? 16 : 4;

        commandList->open();
        commandList->writeTexture(originalTexture, 0, 0, data, width * bytesPerPixel);
        commandList->close();
        device->executeCommandList(commandList);
        device->waitForIdle();

        return true;
    }
};

fs::path GetRelativePath(fs::path const& baseDir, fs::path const& filePath)
{
    // Iterate over components of both paths while the components match
    auto basePathIt = baseDir.begin();
    auto filePathIt = filePath.begin();
    while (basePathIt != baseDir.end() && filePathIt != filePath.end() && *basePathIt == *filePathIt)
    {
        ++basePathIt;
        ++filePathIt;
    }

    // If we haven't consumed the entire base path, the paths are unrelated, return full path
    if (basePathIt != baseDir.end())
        return filePath;

    // Construct a relative path from the remaining components of the file path
    fs::path result;
    while (filePathIt != filePath.end())
    {
        if (result.empty())
            result = *filePathIt;
        else
            result /= *filePathIt;
        ++filePathIt;
    }

    return result;
}

std::shared_ptr<ImageData> LoadImage(fs::path const& fileName)
{
    auto imageData = std::make_shared<ImageData>();

    std::string extension = fileName.extension().string();
    LowercaseString(extension);
    
    imageData->isHDR = extension == ".exr";

    // Load the image data
    if (imageData->isHDR)
    {
        LoadEXR((float**)&imageData->data, &imageData->width, &imageData->height,
            fileName.generic_string().c_str(), nullptr);
        imageData->channels = 4;
    }
    else
    {   
        imageData->data = stbi_load(fileName.string().c_str(), &imageData->width,
            &imageData->height, &imageData->channels, 4);
    }

    imageData->widthInBlocks = (imageData->width + 3) / 4;
    imageData->heightInBlocks = (imageData->height + 3) / 4;
    
    if (!imageData->data)
        return nullptr;

    // Make the image name a relative path, starting from --source
    imageData->name = GetRelativePath(g_options.sourcePath, fileName);
    
    return imageData;
}

void WriteDdsHeader(FILE* ddsFile, int width, int height, int mipLevels, BcFormatDefinition const& formatDef, bool sRGB)
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
    dx10header.dxgiFormat = sRGB ? formatDef.dxgiFormatSrgb : formatDef.dxgiFormat;

    uint32_t ddsMagic = DDS_MAGIC;
    fwrite(&ddsMagic, sizeof(ddsMagic), 1, ddsFile);
    fwrite(&ddsHeader, sizeof(ddsHeader), 1, ddsFile);
    fwrite(&dx10header, sizeof(dx10header), 1, ddsFile);
}

bool WriteDdsFile(fs::path const& outputFileName, int width, int height, BcFormatDefinition const& formatDef, uint8_t const* pData, size_t rowPitch, bool sRGB)
{
    if (!pData)
        return false;

    fs::path outputPath = outputFileName.parent_path();
    if (!fs::is_directory(outputPath))
    {
        if (!fs::create_directories(outputPath))
            return false;
    }

    FILE* ddsFile = fopen(outputFileName.string().c_str(), "wb");
    if (!ddsFile)
        return false;

    WriteDdsHeader(ddsFile, width, height, 1, formatDef, sRGB);

    int widthInBlocks = (width + 3) / 4;
    int heightInBlocks = (height + 3) / 4;

    for (int row = 0; row < heightInBlocks; ++row)
    {
        fwrite(pData + rowPitch * row, formatDef.bytesPerBlock, widthInBlocks, ddsFile);
    }

    fclose(ddsFile);
    return true;
}

bool CompressWithNtc(
    ImageData const& imageData,
    BcFormatDefinition const& formatDef,
    ntc::IContext* context,
    GraphicsBlockCompressionPass& blockCompressionPass,
    GraphicsImageDifferencePass& imageDifferencePass,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    nvrhi::ITimerQuery* timerQuery,
    float& outPsnr,
    float& outRmse,
    float& outGPixelsPerSecond)
{
    float const alphaThreshold = 1.f / 255.f;

    ntc::MakeBlockCompressionComputePassParameters compressionParams;
    compressionParams.srcRect.width = imageData.width;
    compressionParams.srcRect.height = imageData.height;
    compressionParams.dstFormat = formatDef.ntcFormat;
    compressionParams.alphaThreshold = alphaThreshold;
    compressionParams.modeBufferSource = imageData.modeBuffer != nullptr ? 
        ntc::BlockCompressionModeBufferSource::Custom : 
        ntc::BlockCompressionModeBufferSource::None;
    compressionParams.modeBufferInfo.custom.widthInBlocks = imageData.widthInBlocks;
    compressionParams.modeBufferInfo.custom.heightInBlocks = imageData.heightInBlocks;
    ntc::ComputePassDesc blockCompressionComputePass;
    ntc::Status ntcStatus = context->MakeBlockCompressionComputePass(compressionParams, &blockCompressionComputePass);
    CHECK_NTC_RESULT(MakeBlockCompressionComputePass)

    ntc::MakeImageDifferenceComputePassParameters differenceParams;
    differenceParams.extent.width = imageData.width;
    differenceParams.extent.height = imageData.height;
    differenceParams.useAlphaThreshold = formatDef.ntcFormat == ntc::BlockCompressedFormat::BC1;
    differenceParams.alphaThreshold = alphaThreshold;
    differenceParams.useMSLE = imageData.isHDR;
    ntc::ComputePassDesc imageDifferenceComputePass;
    ntcStatus = context->MakeImageDifferenceComputePass(differenceParams, &imageDifferenceComputePass);
    CHECK_NTC_RESULT(MakeImageDifferenceComputePass)


    commandList->open();
    commandList->beginTimerQuery(timerQuery);
    if (!blockCompressionPass.ExecuteComputePass(commandList, blockCompressionComputePass,
        imageData.originalTexture, nvrhi::Format::UNKNOWN, 0,
        imageData.modeBuffer,
        imageData.blockTexture, 0))
    {
        commandList->endTimerQuery(timerQuery);
        commandList->close();
        return false;
    }
    commandList->endTimerQuery(timerQuery);
    auto srcSlice = nvrhi::TextureSlice().setWidth(imageData.widthInBlocks).setHeight(imageData.heightInBlocks);
    auto dstSlice = nvrhi::TextureSlice().setWidth(imageData.widthInBlocks * 4).setHeight(imageData.heightInBlocks * 4);
    commandList->copyTexture(imageData.compressedTexture, dstSlice, imageData.blockTexture, srcSlice);
    if (!imageDifferencePass.ExecuteComputePass(commandList, imageDifferenceComputePass,
        imageData.originalTexture, 0, imageData.compressedTexture, 0, 0))
    {
        commandList->close();
        return false;
    }
    commandList->copyTexture(imageData.stagingTexture, nvrhi::TextureSlice(), imageData.blockTexture, nvrhi::TextureSlice());
    commandList->close();

    device->executeCommandList(commandList);
    device->waitForIdle();
    device->runGarbageCollection();

    float const timeSeconds = device->getTimerQueryTime(timerQuery);
    if (timeSeconds > 0.f)
        outGPixelsPerSecond = 1e-9f * float(imageData.width * imageData.height) / timeSeconds;
    else
        outGPixelsPerSecond = 0.f;

    float mse = 0.f;
    imageDifferencePass.ReadResults();
    imageDifferencePass.GetQueryResult(0, nullptr, &mse, &outPsnr, formatDef.channels);
    
    outRmse = sqrtf(mse);

    // Note: for HDR images, these dB values are fake/false because we use MSLE and not MSE!
    // Also, they are calculated as if the maximum value of log(color + 1) was 1.0, and it's actually 11.09 for FP16/BC6.
    // This way, we're getting "sane" dB values like 40, but they're only useful for relative comparison in the same
    // framework.
    printf("[%s]  %s: %.2f %sdB, %.3f Gpix/s\n", imageData.modeBuffer ? "NTCa" : "NTC ", imageData.name.generic_string().c_str(),
        outPsnr, imageData.isHDR ? "false " : "", outGPixelsPerSecond);

    if (g_options.outputPath)
    {
        fs::path ddsName = imageData.name;
        ddsName.replace_extension(std::string(".") + g_options.format + ".NTC.dds");
        fs::path outputFileName = fs::path(g_options.outputPath) / ddsName;
        size_t rowPitch;
        uint8_t const* compressedData = (uint8_t const*)device->mapStagingTexture(imageData.stagingTexture, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);

        if (WriteDdsFile(outputFileName, imageData.width, imageData.height, formatDef, compressedData, rowPitch, false))
            printf("Saved '%s'\n", outputFileName.string().c_str());
        else
            fprintf(stderr, "Failed to save '%s'\n", outputFileName.string().c_str());

        device->unmapStagingTexture(imageData.stagingTexture);
    }

    return true;
}

bool MakeBC7ModeBuffer(
    ImageData& imageData,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList)
{
    size_t rowPitch;
    uint8_t const* blockData = (uint8_t const*)device->mapStagingTexture(imageData.stagingTexture,
        nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);
    if (!blockData)
        return false;

    size_t const modeBufferSize = ntc::GetBC7ModeBufferSize(imageData.widthInBlocks, imageData.heightInBlocks);
    std::vector<uint16_t> modeData(modeBufferSize);

    ntc::Status ntcStatus = ntc::MakeBC7ModeBuffer(imageData.widthInBlocks, imageData.heightInBlocks, blockData,
        rowPitch * imageData.heightInBlocks, rowPitch, modeData.data(), modeData.size());

    device->unmapStagingTexture(imageData.stagingTexture);
    blockData = nullptr;

    CHECK_NTC_RESULT(MakeBC7ModeBuffer);

    nvrhi::BufferDesc modeBufferDesc = nvrhi::BufferDesc()
        .setDebugName("BC7 Mode Buffer")
        .setByteSize(modeBufferSize)
        .setCanHaveRawViews(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true);

    imageData.modeBuffer = device->createBuffer(modeBufferDesc);
    if (!imageData.modeBuffer)
        return false;
    
    commandList->open();
    commandList->writeBuffer(imageData.modeBuffer, modeData.data(), modeBufferSize);
    commandList->close();
    
    device->executeCommandList(commandList);

    return true;
}

#if NTC_WITH_NVTT
bool CompressWithNvtt(
    ImageData const& imageData,
    BcFormatDefinition const& formatDef,
    ntc::IContext* context,
    GraphicsImageDifferencePass& imageDifferencePass,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    float& outPsnr,
    float& outRmse)
{
    float const alphaThreshold = 1.f / 255.f;

    nvtt::RefImage image;
    image.width = imageData.width;
    image.height = imageData.height;
    image.num_channels = 4;
    image.data = imageData.data;
    nvtt::CPUInputBuffer inputBuff(&image, formatDef.nvttValueType);
    nvtt::EncodeSettings eset;
    eset.SetFormat(formatDef.nvttFormat)
        .SetOutputToGPUMem(false)
        .SetUseGPU(true)
        .SetQuality(nvtt::Quality_Normal);
    
    std::vector<uint8_t> blockData(imageData.widthInBlocks * imageData.heightInBlocks * formatDef.bytesPerBlock);
    bool success = nvtt::nvtt_encode(inputBuff, blockData.data(), eset);

    if (!success)
    {
        fprintf(stderr, "Call to nvtt_encode failed.\n");
        return false;
    }

    ntc::MakeImageDifferenceComputePassParameters differenceParams;
    differenceParams.extent.width = imageData.width;
    differenceParams.extent.height = imageData.height;
    differenceParams.useAlphaThreshold = formatDef.ntcFormat == ntc::BlockCompressedFormat::BC1;
    differenceParams.alphaThreshold = alphaThreshold;
    differenceParams.useMSLE = imageData.isHDR;
    ntc::ComputePassDesc imageDifferenceComputePass;
    ntc::Status ntcStatus = context->MakeImageDifferenceComputePass(differenceParams, &imageDifferenceComputePass);
    CHECK_NTC_RESULT(MakeImageDifferenceComputePass)

    size_t const rowPitch = imageData.widthInBlocks * formatDef.bytesPerBlock;

    commandList->open();
    commandList->writeTexture(imageData.compressedTexture, 0, 0, blockData.data(), rowPitch);
    if (!imageDifferencePass.ExecuteComputePass(commandList, imageDifferenceComputePass,
        imageData.originalTexture, 0, imageData.compressedTexture, 0, 0))
    {
        commandList->close();
        return false;
    }
    commandList->close();

    device->executeCommandList(commandList);
    device->waitForIdle();
    device->runGarbageCollection();
    
    float mse = 0.f;
    imageDifferencePass.ReadResults();
    imageDifferencePass.GetQueryResult(0, nullptr, &mse, &outPsnr, formatDef.channels);
    
    outRmse = sqrtf(mse);
    
    // See the comment in CompressWithNtc(...) near similar printf on why the HDR dB values are fake.
    printf("[NVTT] %s: %.2f %sdB\n", imageData.name.generic_string().c_str(),
        outPsnr, imageData.isHDR ? "false " : "");

    if (g_options.outputPath)
    {
        fs::path ddsName = imageData.name;
        ddsName.replace_extension(std::string(".") + g_options.format + ".NVTT.dds");
        fs::path const outputFileName = fs::path(g_options.outputPath) / ddsName;


        if (WriteDdsFile(outputFileName, imageData.width, imageData.height, formatDef, blockData.data(), rowPitch, false))
            printf("Saved '%s'\n", outputFileName.string().c_str());
        else
            fprintf(stderr, "Failed to save '%s'\n", outputFileName.string().c_str());
    }

    return true;
}
#endif

struct Result
{
    fs::path name;
    float ntcPsnr = 0;
    float ntcRmse = 0;
    float baselineNtcPsnr = 0;
    float nvttPsnr = 0;
    float nvttRmse = 0;
    float ntcGPixelsPerSecond = 0;
    bool ntcAcceleratedMatch = false;
};

// Splits the comma separated string into a vector of its components.
std::vector<std::string> SplitString(std::string const& s)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t comma = 0;
    while ((comma = s.find(',', start)) != std::string::npos)
    {
        result.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    if (start < s.size())
        result.push_back(s.substr(start, s.size() - start));
    return result;
}

// Returns the index of the given string in a vector of strings.
int FindColumn(std::vector<std::string> const& header, char const* name)
{
    auto it = std::find(header.begin(), header.end(), name);
    if (it == header.end())
        return -1;
    return it - header.begin();
}

// Converts a string into a float, with support for the 'inf' literal that sometimes appears in our data.
float ParseFloatInf(char const* s)
{
    if (strcasecmp(s, "inf") == 0)
        return std::numeric_limits<float>::infinity();
    return atof(s);
}

bool LoadBaseline(char const* fileName, std::vector<Result>& outResults)
{
    outResults.clear();

    std::ifstream file(fileName);
    if (!file.is_open())
    {
        fprintf(stderr, "Cannot open file '%s'\n", fileName);
        return false;
    }

    // Read the file line by line
    std::string line;
    int lineno = 0;
    int nameCol = -1;
    int nvttCol = -1;
    int ntcCol = -1;
    int ntcPerfCol = -1;
    while (std::getline(file, line))
    {
        ++lineno;
        std::vector<std::string> parts = SplitString(line);
        if (lineno == 1)
        {
            // First line contains the headers: find the indices of interesting columns there.
            nameCol = FindColumn(parts, "Name");
            nvttCol = FindColumn(parts, "NVTT dB");
            ntcCol = FindColumn(parts, "NTC dB");
            ntcPerfCol = FindColumn(parts, "NTC Gpix/s");
            if (nameCol < 0)
            {
                fprintf(stderr, "There is no Name column in the input CSV file '%s'", fileName);
                return false;
            }
        }
        else
        {
            // Other lines contain numeric data: extract the data.
            Result result;
            result.name = parts[nameCol];
            if (nvttCol >= 0 && nvttCol < int(parts.size()))
                result.nvttPsnr = ParseFloatInf(parts[nvttCol].c_str());
            if (ntcCol >= 0 && ntcCol < int(parts.size()))
                result.ntcPsnr = ParseFloatInf(parts[ntcCol].c_str());
            if (ntcPerfCol >= 0 && ntcPerfCol < int(parts.size()))
                result.ntcGPixelsPerSecond = ParseFloatInf(parts[ntcPerfCol].c_str());
            outResults.push_back(std::move(result));
        }
    }

    return true;
}

// A class that takes a sequence of numbers and computes statistical metrics of them:
// mean, standard deviation, minimum and maximum.
class Statistic
{
public:
    void Append(float value)
    {
        if (std::isnan(value) || std::isinf(value))
            return;
        m_sum += value;
        m_sumSquares += value * value;
        m_min = std::min(m_min, double(value));
        m_max = std::max(m_max, double(value));
        ++m_count;
    }

    bool Empty() const
    {
        return m_count == 0;
    }

    double GetMean() const
    {
        if (m_count == 0)
            return 0;
        return m_sum / m_count;
    }

    double GetStdDev() const
    {
        if (m_count == 0)
            return 0;
        double mean = m_sum / m_count;
        double l2 = m_sumSquares / m_count;
        return sqrt(l2 - mean * mean);
    }

    double GetMin() const
    {
        return m_min;
    }

    double GetMax() const
    {
        return m_max;
    }

private:
    double m_sum = 0.f;
    double m_sumSquares = 0.f;
    double m_min = std::numeric_limits<double>::max();
    double m_max = -std::numeric_limits<double>::max();
    int m_count = 0;
};

bool RunTests(std::vector<fs::path> sourceFiles, std::vector<Result>& results, ntc::BlockCompressedFormat format,
    ntc::IContext* context, nvrhi::IDevice* device)
{
    BcFormatDefinition const* pFormatDef = GetFormatDef(format);

    // Pre-initialize shared graphics passes

    GraphicsBlockCompressionPass blockCompressionPass(device, true);
    if (!blockCompressionPass.Init())
        return false;

    GraphicsImageDifferencePass imageDifferencePass(device);
    if (!imageDifferencePass.Init())
        return false;

    nvrhi::CommandListHandle commandList = device->createCommandList();
    nvrhi::TimerQueryHandle timerQuery = device->createTimerQuery();

    // The runner uses multiple threads to load source images because decoding PNG or JPG takes a long time.
    // The source image paths are placed into sourceFileQueue, and the threads pull tasks from that queue.
    // Once loaded, ImageData objects are placed into imageQueue. The main thread pulls images from that queue.

    std::queue<fs::path> sourceFileQueue;
    for (fs::path const& path : sourceFiles)
        sourceFileQueue.push(path);
    std::queue<std::shared_ptr<ImageData>> imageQueue;
    std::mutex sourceMutex;
    std::mutex imageMutex;

    std::vector<std::shared_ptr<std::thread>> threads;
    int numThreads = g_options.threads > 0 ? g_options.threads : std::thread::hardware_concurrency();
    numThreads = std::min(int(sourceFiles.size()), numThreads);

    // Using a live thread counter to find out when all files have been processed.
    // This can't be done by just looking at either queue because a task can be in-flight when both queues are empty.
    std::atomic<int> liveThreads = numThreads;

    // Start the decoding threads
    for (int i = 0; i < numThreads; ++i)
    {
        auto thread = std::make_shared<std::thread>([&sourceFileQueue, &sourceMutex, &imageQueue, &imageMutex, &liveThreads]()
        {
            while(!g_Terminate)
            {
                // Pull a task from sourceFileQueue
                fs::path fileName;
                {
                    std::lock_guard<std::mutex> lock(sourceMutex);
                    if (sourceFileQueue.empty())
                        break;
                    fileName = sourceFileQueue.front();
                    sourceFileQueue.pop();
                }

                // Process the task
                std::shared_ptr<ImageData> imageData = LoadImage(fileName);

                // If decoding was successful, put the image data into imageQueue
                if (imageData)
                {
                    std::lock_guard<std::mutex> lock(imageMutex);
                    imageQueue.push(imageData);
                }
            }
            --liveThreads;
        });
        threads.push_back(thread);
    }

    // Main loop that pulls images from imageQueue and runs the compression tests on them
    while (!g_Terminate)
    {
        std::shared_ptr<ImageData> imageData;
        {
            std::lock_guard<std::mutex> lock(imageMutex);
            if (!imageQueue.empty())
            {
                imageData = imageQueue.front();
                imageQueue.pop();
            }
        }

        // If we couldn't pull a task from the queue, it means either something is still decoding or we're done
        if (!imageData)
        {
            // All the threads finished means there are no more tasks
            if (liveThreads <= 0)
                break;

            // There are more tasks: sleep a bit and try again
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Create the graphics texture objects and upload data to the GPU
        if (!imageData->InitTextures(device, commandList, *pFormatDef))
            continue;

        Result result;
        result.name = imageData->name;

        if (g_options.ntc)
        {
            CompressWithNtc(*imageData, *pFormatDef, context, blockCompressionPass, imageDifferencePass,
                device, commandList, timerQuery, result.ntcPsnr, result.ntcRmse,
                result.ntcGPixelsPerSecond);

            if (format == ntc::BlockCompressedFormat::BC7 && g_options.acceleratedMode)
            {
                MakeBC7ModeBuffer(*imageData, device, commandList);
                
                float acceleratedPsnr = 0.f;
                CompressWithNtc(*imageData, *pFormatDef, context, blockCompressionPass, imageDifferencePass,
                    device, commandList, timerQuery, acceleratedPsnr, result.ntcRmse,
                    result.ntcGPixelsPerSecond);

                result.ntcAcceleratedMatch = acceleratedPsnr == result.ntcPsnr;
            }
        }

#if NTC_WITH_NVTT
        if (g_options.nvtt)
        {
            CompressWithNvtt(*imageData, *pFormatDef, context, imageDifferencePass,
                device, commandList, result.nvttPsnr, result.nvttRmse);
        }
#endif

        results.push_back(result);
    }

    // Wait until all threads have finished
    for (auto& thread : threads)
        thread->join();

    return !g_Terminate;
}

// Calculates the truncated mean of the values in the input vector. 
// The discardLow and discardHigh parameters control how much to truncate from each end of the set, [0 - 0.5]
static float TruncatedMean(std::vector<float>& items, float discardLow, float discardHigh)
{
    if (items.empty())
        return std::numeric_limits<float>::quiet_NaN();
    std::sort(items.begin(), items.end());
    size_t first = size_t(items.size() * discardLow);
    size_t last = std::max(size_t(items.size() * (1.f - discardHigh)), first + 1);
    float sum = std::accumulate(items.begin() + first, items.begin() + last, 0.f);
    return sum / float(last - first);
}

bool ProcessResults( ntc::BlockCompressedFormat format, std::vector<Result> const& baselineResults, std::vector<Result>& results)
{
    std::sort(results.begin(), results.end(), [](Result const& a, Result const& b)
    {
        return a.name < b.name;
    });


    Statistic ntcBaselineDiff;
    Statistic ntcNvttDiff;
    std::vector<float> currentNtcGpixPerSecond;
    std::vector<float> baselineNtcGpixPerSecond;
    int matchCount = 0;

    // Go over all the new results and:
    //  a) Collate them to baseline results;
    //  b) Compute the statistical values on image quality differences.
    for (Result& result : results)
    {
        if (!baselineResults.empty())
        {
            auto baselineResult = std::find_if(baselineResults.begin(), baselineResults.end(), [&result](Result const& a)
            {
                return a.name == result.name;
            });

            if (baselineResult != baselineResults.end())
            {
                if (!g_options.ntc)
                    result.ntcPsnr = baselineResult->ntcPsnr;
                else
                    result.baselineNtcPsnr = baselineResult->ntcPsnr;
#if NTC_WITH_NVTT
                if (!g_options.nvtt)
                    result.nvttPsnr = baselineResult->nvttPsnr;
#endif

                baselineNtcGpixPerSecond.push_back(baselineResult->ntcGPixelsPerSecond);
            }
        }

        if (result.ntcPsnr != 0.f && result.baselineNtcPsnr != 0.f)
            ntcBaselineDiff.Append(result.ntcPsnr - result.baselineNtcPsnr);

#if NTC_WITH_NVTT
        if (result.ntcPsnr != 0.f && result.nvttPsnr != 0.f)
            ntcNvttDiff.Append(result.ntcPsnr - result.nvttPsnr);
#endif

        currentNtcGpixPerSecond.push_back(result.ntcGPixelsPerSecond);

        if (result.ntcAcceleratedMatch)
            ++matchCount;
        else
        {
            if (g_options.acceleratedMode)
                printf("NTC accelerated mode mismatch: %s\n", result.name.generic_string().c_str());
        }
    }

    // Use truncated mean to calculate the average perf.
    // The data is very noisy with lots of outliers, so truncate a lot from both ends,
    // sort of like using a stabilized median.
    float const discardLow = 0.2f;
    float const discardHigh = 0.2f;
    float meanNtcGpixPerSecond = currentNtcGpixPerSecond.empty() ? 0.f : TruncatedMean(currentNtcGpixPerSecond, discardLow, discardHigh);
    float meanBaselineNtcGpixPerSecond = baselineNtcGpixPerSecond.empty() ? 0.f : TruncatedMean(baselineNtcGpixPerSecond, discardLow, discardHigh);

    if (!currentNtcGpixPerSecond.empty())
        printf("Average NTC encoding perf: %.3f Gpix/s\n", meanNtcGpixPerSecond);
    
    if (g_options.acceleratedMode)
    {
        printf("NTC accelerated mode match: %d / %d\n", matchCount, int(results.size()));
    }

    // Print out the quality statistics
    if (!ntcBaselineDiff.Empty())
    {
        float speedup = meanBaselineNtcGpixPerSecond > 0.f 
            ? 100.f * (meanNtcGpixPerSecond - meanBaselineNtcGpixPerSecond) / meanBaselineNtcGpixPerSecond
            : std::numeric_limits<float>::quiet_NaN();

        printf("(NTC - BaselineNTC): Mean = %.3f dB, StdDev = %.3f dB, Min = %.3f dB, Max = %.3f dB, Speedup = %.2f%%\n",
            ntcBaselineDiff.GetMean(), ntcBaselineDiff.GetStdDev(),
            ntcBaselineDiff.GetMin(), ntcBaselineDiff.GetMax(),
            speedup);
    }
    
#if NTC_WITH_NVTT
    if (!ntcNvttDiff.Empty())
    {
        printf("(NTC - NVTT):        Mean = %.3f dB, StdDev = %.3f dB, Min = %.3f dB, Max = %.3f dB\n",
            ntcNvttDiff.GetMean(), ntcNvttDiff.GetStdDev(),
            ntcNvttDiff.GetMin(), ntcNvttDiff.GetMax());
    }
#endif


    // Save the results into a CSV file, if requested by the user
    if (g_options.csvOutputPath)
    {
        fs::path csvParent = fs::path(g_options.csvOutputPath).parent_path();
        if (!csvParent.empty() && !fs::is_directory(csvParent))
            fs::create_directories(csvParent);

        FILE* csvFile = fopen(g_options.csvOutputPath, "w");
        if (csvFile)
        {
            fprintf(csvFile, "Name,NTC dB,NTC RMS(L)E,NTC Gpix/s,Baseline NTC dB,NVTT dB,NVTT RMS(L)E,NTC - NVTT dB,NTC Improvement dB\n");
            for (Result const& result : results)
            {
                fprintf(csvFile, "%s,%.3f,%.5f,%.3f,%.3f,%.3f,%.5f,%.3f,%.3f\n", result.name.generic_string().c_str(),
                    result.ntcPsnr, result.ntcRmse, result.ntcGPixelsPerSecond, result.baselineNtcPsnr,
                    result.nvttPsnr, result.nvttRmse,
                    result.ntcPsnr - result.nvttPsnr, result.ntcPsnr - result.baselineNtcPsnr);
            }
            fclose(csvFile);
        }
        else
        {
            fprintf(stderr, "Cannot open file '%s'\n", g_options.csvOutputPath);
            return false;
        }
    }

    return true;
}


int main(int argc, const char** argv)
{
    donut::log::ConsoleApplicationMode();
    donut::log::SetMinSeverity(donut::log::Severity::Warning);

    if (!ProcessCommandLine(argc, argv))
        return 1;

    std::vector<Result> baselineResults;
    if (g_options.loadBaselinePath)
    {
        if (!LoadBaseline(g_options.loadBaselinePath, baselineResults))
            return 1;
        printf("Loaded %d baseline results from '%s'\n", int(baselineResults.size()), g_options.loadBaselinePath);
    }

    std::unique_ptr<donut::app::DeviceManager> deviceManager = InitGraphicsDevice();
    if (!deviceManager)
        return 1;

    ntc::ContextWrapper context;
    if (!InitNtcContext(deviceManager->GetDevice(), context))
        return 1;

    signal(SIGINT, SigintHandler);

    ntc::BlockCompressedFormat format = ParseBlockCompressedFormat(g_options.format).value_or(ntc::BlockCompressedFormat::None);
    std::vector<fs::path> sourceFiles = EnumerateSourceFiles();
    std::vector<Result> results;
    if (!RunTests(sourceFiles, results, format, context, deviceManager->GetDevice()))
        return 1;

    if (!ProcessResults(format, baselineResults, results))
        return 1;

    return 0;
}