/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvrhi/utils.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/TextureCache.h>
#include <donut/app/DeviceManager.h>
#include <ntc-utils/GraphicsImageDifferencePass.h>
#include <argparse.h>
#include <cmath>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

struct
{
    std::vector<char const*> sources;
    bool useVulkan = false;
    bool useDX12 = false;
    bool debug = false;
    int adapterIndex = -1;
    int numChannels = 0;
} g_options;

bool ProcessCommandLine(int argc, const char** argv)
{
    struct argparse_option options[] = {
        OPT_HELP(),
#if DONUT_WITH_VULKAN
        OPT_BOOLEAN(0, "vk", &g_options.useVulkan, "Use Vulkan API"),
#endif
#if DONUT_WITH_DX12
        OPT_BOOLEAN(0, "dx12", &g_options.useDX12, "Use D3D12 API"),
#endif
        OPT_BOOLEAN(0, "debug", &g_options.debug, "Enable debug features such as Vulkan validation layer or D3D12 debug runtime"),
        OPT_INTEGER(0, "adapter", &g_options.adapterIndex, "Index of the graphics adapter to use"),
        OPT_INTEGER(0, "channels", &g_options.numChannels, "Number of channels to compare (0 = auto-detect, default)"),
        OPT_END()
    };

    static const char* usages[] = {
        "imagediff.exe <paths...> [options...]",
        nullptr
    };

    struct argparse argparse {};
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "\nImage comparison tool.\nInput files must be provided in pairs.", nullptr);
    argparse_parse(&argparse, argc, argv);

    if (g_options.useVulkan && g_options.useDX12)
    {
        fprintf(stderr, "Only one of --vk or --dx12 options can be specified.\n");
        return false;
    }

    if (!g_options.useVulkan && !g_options.useDX12)
    {
        #if DONUT_WITH_VULKAN
        g_options.useVulkan = true;
        #else
        g_options.useDX12 = true;
        #endif
        assert(g_options.useDX12 || g_options.useVulkan);
    }
    
    if (g_options.numChannels < 0 || g_options.numChannels > 4)
    {
        fprintf(stderr, "The --channels value must be between 0 and 4.\n");
        return false;
    }

    // Process positional arguments
    for (int i = 0; argparse.out[i]; ++i)
    {
        char const* arg = argparse.out[i];
        if (!arg[0])
            continue;
        g_options.sources.push_back(arg);
    }

    if (g_options.sources.size() == 0 || (g_options.sources.size() & 1) != 0)
    {
        fprintf(stderr, "An even number of input paths must be specified.\n");
        return false;
    }

    return true;
}

#define CHECK_NTC_RESULT(fname) \
    if (ntcStatus != ntc::Status::Ok) { \
        fprintf(stderr, "Call to " #fname " failed, code = %s\n%s\n", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage()); \
        return false; \
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

int GetChannelCountForFormat(nvrhi::Format format)
{
    nvrhi::FormatInfo const& formatInfo = nvrhi::getFormatInfo(format);
    int numChannels = 0;
    if (formatInfo.hasRed)   ++numChannels;
    if (formatInfo.hasGreen) ++numChannels;
    if (formatInfo.hasBlue)  ++numChannels;
    if (formatInfo.hasAlpha) ++numChannels;
    return numChannels;
}

bool CompareTwoImages(nvrhi::IDevice* device, nvrhi::ICommandList* commandList,
    donut::engine::TextureCache& textureCache,
    ntc::IContext* ntcContext,
    int pairIndex,
    char const* source1, char const* source2)
{
    commandList->open();
    
    std::shared_ptr<donut::engine::LoadedTexture> texture1 = textureCache.LoadTextureFromFile(
        source1, false, nullptr, commandList);

    bool const sRGB = (texture1->texture && nvrhi::getFormatInfo(texture1->texture->getDesc().format).isSRGB);

    std::shared_ptr<donut::engine::LoadedTexture> texture2 = textureCache.LoadTextureFromFile(
        source2, sRGB, nullptr, commandList);

    commandList->close();
    device->executeCommandList(commandList);

    if (!texture1->texture)
    {
        fprintf(stderr, "Failed to load texture from %s\n", source1);
        return false;
    }

    if (!texture2->texture)
    {
        fprintf(stderr, "Failed to load texture from %s\n", source2);
        return false;
    }

    nvrhi::TextureDesc const& desc1 = texture1->texture->getDesc();
    nvrhi::TextureDesc const& desc2 = texture2->texture->getDesc();

    if (desc1.width != desc2.width || desc1.height != desc2.height)
    {
        fprintf(stderr, "Input images have different dimensions: %ux%u and %ux%u\n",
            desc1.width, desc1.height, desc2.width, desc2.height);
        return false;
    }

    int const mipLevels = std::min(desc1.mipLevels, desc2.mipLevels);
    if (desc1.mipLevels != desc2.mipLevels)
    {
        fprintf(stderr, "Warning: Input images have different mip level counts: %u and %u. Using the smaller count.\n",
            desc1.mipLevels, desc2.mipLevels);
    }

    int numChannels = g_options.numChannels;
    if (numChannels <= 0)
    {
        int const numChannels1 = GetChannelCountForFormat(desc1.format);
        int const numChannels2 = GetChannelCountForFormat(desc2.format);
        numChannels = std::min(numChannels1, numChannels2);
        if (numChannels1 != numChannels2)
        {
            fprintf(stderr, "Warning: Input images have different channel counts: %d and %d. Using the smaller count.\n",
                numChannels1, numChannels2);
        }
    }

    GraphicsImageDifferencePass imageDifferencePass(device, mipLevels);
    if (!imageDifferencePass.Init())
    {
        fprintf(stderr, "Failed to initialize the image difference pass.\n");
        return false;
    }

    commandList->open();

    for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        uint32_t mipWidth = std::max(1u, desc1.width >> mipLevel);
        uint32_t mipHeight = std::max(1u, desc1.height >> mipLevel);

        ntc::MakeImageDifferenceComputePassParameters imageDifferenceParams = {};
        imageDifferenceParams.extent.width = int(mipWidth);
        imageDifferenceParams.extent.height = int(mipHeight);
        ntc::ComputePassDesc computePass = {};
        ntc::Status ntcStatus = ntcContext->MakeImageDifferenceComputePass(imageDifferenceParams, &computePass);
        CHECK_NTC_RESULT(MakeImageDifferenceComputePass);

        imageDifferencePass.ExecuteComputePass(
            commandList, computePass,
            texture1->texture, mipLevel,
            texture2->texture, mipLevel,
            /* queryIndex = */ mipLevel);
    }

    commandList->close();
    device->executeCommandList(commandList);

    if (!imageDifferencePass.ReadResults())
    {
        fprintf(stderr, "Failed to read image difference results from the GPU.\n");
        return false;
    }

    for (int mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        float mse = 0;
        float psnr = 0;
        if (!imageDifferencePass.GetQueryResult(mipLevel, nullptr, &mse, &psnr, numChannels))
        {
            fprintf(stderr, "Failed to get image difference results for mip level %d.\n", mipLevel);
            return 1;
        }

        printf("PAIR %d MIP %2d: MSE = %.4f, PSNR = %.2f dB\n", pairIndex, mipLevel, mse, psnr);
    }

    return true;
}

int main(int argc, const char** argv)
{
    donut::log::ConsoleApplicationMode();
    donut::log::SetMinSeverity(donut::log::Severity::Error);

    if (!ProcessCommandLine(argc, argv))
        return 1;

    std::unique_ptr<donut::app::DeviceManager> deviceManager = InitGraphicsDevice();
    if (!deviceManager)
        return 1;

    nvrhi::DeviceHandle device = deviceManager->GetDevice();
    nvrhi::CommandListHandle commandList = device->createCommandList();

    ntc::ContextWrapper ntcContext;
    if (!InitNtcContext(device, ntcContext))
        return 1;

    std::shared_ptr<donut::vfs::IFileSystem> fileSystem = std::make_shared<donut::vfs::NativeFileSystem>();
    std::shared_ptr<donut::engine::TextureCache> textureCache = std::make_shared<donut::engine::TextureCache>(
        device, fileSystem, nullptr);
    textureCache->SetGenerateMipmaps(false);
    textureCache->SetMaxTextureSize(16384);

    for (size_t inputIndex = 0; inputIndex < g_options.sources.size(); inputIndex += 2)
    {
        if (!CompareTwoImages(device, commandList, *textureCache, ntcContext,
            int(inputIndex / 2), g_options.sources[inputIndex], g_options.sources[inputIndex + 1]))
            return 1;
    }

    return 0;
}