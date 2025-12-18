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

#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/Scene.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/ThreadPool.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/core/string_utils.h>
#include <nvrhi/utils.h>
#include <ntc-utils/GraphicsDecompressionPass.h>
#include <ntc-utils/Manifest.h>
#include <ntc-utils/DeviceUtils.h>
#include <ntc-utils/Misc.h>
#include <ntc-utils/Semantics.h>
#include <filesystem>
#include <argparse.h>
#include <stb_image.h>
#include <sstream>
#include <tinyexr.h>
#include <unordered_set>
#include <libntc/ntc.h>
#include <imgui_internal.h>
#include <cuda_runtime_api.h>


#include "FlatImageView.h"
#include "ModelView.h"
#include "ImGuiExtensions.h"

using namespace donut;
using namespace std::chrono;

namespace fs = std::filesystem;

static const char* g_ApplicationName = "Neural Texture Compression Explorer";

struct
{
    ToolInputType inputType = ToolInputType::None;
    std::vector<char const*> sourcePaths;
    bool debug = false;
    bool noshared = false;
    bool captureMode = false;
    bool hdr = false;
    bool useVulkan = false;
    bool useDX12 = false;
    bool compare = false;
    bool enableCoopVec = true;
    int adapterIndex = -1;
    int cudaDevice = 0;
} g_options;

bool ProcessCommandLine(int argc, const char** argv)
{
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_BOOLEAN(0, "debug", &g_options.debug, "Enable graphics debug runtime"),
        OPT_BOOLEAN(0, "noshared", &g_options.noshared, "Disable the use of shared textures (CUDA/Graphics interop)"),
        OPT_INTEGER(0, "adapter", &g_options.adapterIndex, "Index of the graphics adapter to use (use ntc-cli.exe --dx12|vk --listAdapters to find out)"),
        OPT_INTEGER(0, "cudaDevice", &g_options.cudaDevice, "Index of the CUDA device to use (use ntc-cli.exe --listCudaDevices to find out)"),
        OPT_BOOLEAN(0, "coopVec", &g_options.enableCoopVec, "Enable CoopVec extensions (default on, use --no-coopVec)"),
        OPT_BOOLEAN(0, "captureMode", &g_options.captureMode, "Trace capture mode - run Graphics decompression in a loop"),
        OPT_BOOLEAN(0, "compare", &g_options.compare, "Use Explorer to compare two images specified on the command line"),
        OPT_BOOLEAN(0, "hdr", &g_options.hdr, "Use an HDR (FP16) swap chain"),
#if NTC_WITH_VULKAN
        OPT_BOOLEAN(0, "vk", &g_options.useVulkan, "Use Vulkan API"),
#endif
#if NTC_WITH_DX12
        OPT_BOOLEAN(0, "dx12", &g_options.useDX12, "Use DX12 API"),
#endif
        OPT_END()
    };

    static const char* usages[] = {
        "ntc-explorer.exe [options...] [<source-folder|source-manifest.json|compressed-file.ntc>]",
        nullptr
    };

    struct argparse argparse {};
    argparse_init(&argparse, options, usages, ARGPARSE_USE_MESSAGE_BUFFER | ARGPARSE_NEVER_EXIT);
    argparse_describe(&argparse, nullptr, "\nNeural texture compression and decompression tool.\n");
    int argparse_result = argparse_parse(&argparse, argc, argv);
    if (argparse_result < 0)
    {
        if (argparse.messages)
        {
            bool isError = argparse_result != ARGPARSE_HELP;
#ifdef _WIN32
            MessageBoxA(NULL, argparse.messages, g_ApplicationName, MB_OK | (isError ? MB_ICONERROR : 0));
#else
            log::error("%s\n", argparse.messages);
#endif
        }
        argparse_cleanup(&argparse);
        return false;
    }

    // Process positional arguments and detect their input types
    for (int i = 0; argparse.out[i]; ++i)
    {
        char const* arg = argparse.out[i];
        if (!arg[0])
            continue;

        fs::path argPath = arg;
        if (fs::is_directory(argPath))
        {
            UpdateToolInputType(g_options.inputType, ToolInputType::Directory);
        }
        else if (fs::exists(argPath))
        {
            std::string extension = argPath.extension().string();
            LowercaseString(extension);

            if (extension == ".json")
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::ManifestFile);
            }
            else if (extension == ".ntc")
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::CompressedTextureSet);
            }
            else if (IsSupportedImageFileExtension(extension))
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::Images);
            }
            else
            {
                log::error("Unknown input file type '%s'.", extension.c_str());
                argparse_cleanup(&argparse);
                return false;
            }
        }
        else
        {
            log::error("The specified file or folder '%s' does not exist.", arg);
            argparse_cleanup(&argparse);
            return false;
        }
        
        g_options.sourcePaths.push_back(arg);
    }

    argparse_cleanup(&argparse);

    if (g_options.useDX12 && g_options.useVulkan)
    {
        log::error("Options --vk and --dx12 cannot be used at the same time.");
        return false;
    }

#if NTC_WITH_DX12 && NTC_WITH_VULKAN
    if (!g_options.useDX12 && !g_options.useVulkan)
    {
        // When both DX12 and Vulkan are supported, prefer Vulkan.
        g_options.useVulkan = true;
    }
#elif NTC_WITH_DX12 && !NTC_WITH_VULKAN
    g_options.useDX12 = true;
    g_options.useVulkan = false;
#elif !NTC_WITH_DX12 && NTC_WITH_VULKAN
    g_options.useDX12 = false;
    g_options.useVulkan = true;
#endif

    if (g_options.inputType == ToolInputType::Mixed)
    {
        log::error("Cannot process inputs of mismatching types (image files, directories, manifests, "
            "compressed texture sets) or multiple inputs of the same type except for images.");
        return false;
    }

    if (g_options.compare)
    {
        if (g_options.inputType != ToolInputType::Images || g_options.sourcePaths.size() != 2)
        {
            log::error("The --compare option requires exactly two image files as input.");
            return false;
        }
    }

    return true;
}

static int GetNumMipLevels(int width, int height)
{
    return int(floorf(std::log2f(float(std::max(width, height)))) + 1);
}

struct MaterialImage
{
    int width = 0;
    int height = 0;
    int channels = 0;
    int firstChannel = 0;
    int manifestIndex = 0;
    std::shared_ptr<uint8_t> data = nullptr;
    std::shared_ptr<uint8_t> decompressedData = nullptr;
    std::string name;
    ntc::ChannelFormat format = ntc::ChannelFormat::UNORM8;
    ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
    std::vector<float> lossFunctionScales;
    bool isSRGB = false;
    bool referenceMipsValid = false;
    bool textureSetDataValid = false;
    size_t uncompressedSize = 0;
    size_t uncompressedSizeWithMips = 0;
    size_t bcSize = 0;
    size_t bcSizeWithMips = 0;
    std::vector<ImageSemanticBinding> manifestSemantics;

    nvrhi::TextureHandle referenceTexture;
    nvrhi::TextureHandle decompressedTextureLeft;
    nvrhi::TextureHandle decompressedTextureRight;
    ntc::ISharedTexture* referenceTextureShared = nullptr;
    ntc::ISharedTexture* decompressedTextureLeftShared = nullptr;
    ntc::ISharedTexture* decompressedTextureRightShared = nullptr;

    bool CreateTextures(nvrhi::IDevice* device, bool createReferenceTexture, bool useSharedTextures, int decompressedWidth, int decompressedHeight, int decompressedMips)
    {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.mipLevels = GetNumMipLevels(width, height);
        switch(format)
        {
            case ntc::ChannelFormat::UNORM8:
                desc.format = nvrhi::Format::RGBA8_UNORM;
                break;
            case ntc::ChannelFormat::UNORM16:
                desc.format = nvrhi::Format::RGBA16_UNORM;
                break;
            case ntc::ChannelFormat::FLOAT16:
                desc.format = nvrhi::Format::RGBA16_FLOAT;
                break;
            case ntc::ChannelFormat::FLOAT32:
                desc.format = nvrhi::Format::RGBA32_FLOAT;
                break;
            case ntc::ChannelFormat::UINT32:
                desc.format = nvrhi::Format::RGBA32_UINT;
                break;
            default: assert(false);
        }
        desc.debugName = name;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.arraySize = 1;
        desc.sharedResourceFlags = useSharedTextures ? nvrhi::SharedResourceFlags::Shared : nvrhi::SharedResourceFlags::None;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;
        desc.isRenderTarget = true;
        desc.isTypeless = true;
        if (createReferenceTexture)
        {
            referenceTexture = device->createTexture(desc);
            if (!referenceTexture)
                return false;
        }

        desc.width = decompressedWidth;
        desc.height = decompressedHeight;
        desc.mipLevels = decompressedMips;
        desc.isUAV = true;
        desc.debugName = name + " (Decompressed Left)";
        decompressedTextureLeft = device->createTexture(desc);
        if (!decompressedTextureLeft)
            return false;

        desc.debugName = name + " (Decompressed Right)";
        decompressedTextureRight = device->createTexture(desc);
        if (!decompressedTextureRight)
            return false;

        return true;
    }

    void ComputeUncompressedSize()
    {
        size_t bytesPerPixel = 0;

        switch(format)
        {
            case ntc::ChannelFormat::UNORM8:
                bytesPerPixel = channels;
                break;

            case ntc::ChannelFormat::UNORM16:
            case ntc::ChannelFormat::FLOAT16:
                bytesPerPixel = 2 * channels;
                break;

            case ntc::ChannelFormat::UINT32:
            case ntc::ChannelFormat::FLOAT32:
                bytesPerPixel = 4 * channels;
                break;
        }
        assert(bytesPerPixel != 0);

        size_t bytesPerBlock = 0;

        switch(bcFormat)
        {
            case ntc::BlockCompressedFormat::BC1:
            case ntc::BlockCompressedFormat::BC4:
                bytesPerBlock = 8;
                break;
            default:
                bytesPerBlock = 16;
                break;
        }

        int const mipLevels = GetNumMipLevels(width, height);
        for (int mip = 0; mip < mipLevels; ++mip)
        {
            int const mipWidth = std::max(1, width >> mip);
            int const mipHeight = std::max(1, height >> mip);

            size_t const uncompressedMipSize = size_t(mipWidth) * size_t(mipHeight) * channels * bytesPerPixel;

            if (mip == 0)
                uncompressedSize = uncompressedMipSize;
            
            uncompressedSizeWithMips += uncompressedMipSize;

            if (bcFormat != ntc::BlockCompressedFormat::None)
            {
                int const mipWidthBlocks = (mipWidth + 3) / 4;
                int const mipHeightBlocks = (mipHeight + 3) / 4;

                size_t const bcMipSize = size_t(mipWidthBlocks) * size_t(mipHeightBlocks) * bytesPerBlock;
                
                if (mip == 0)
                    bcSize = bcMipSize;

                bcSizeWithMips += bcMipSize;
            }
        }

        if (bcFormat == ntc::BlockCompressedFormat::None)
        {
            bcSize = uncompressedSize;
            bcSizeWithMips = uncompressedSizeWithMips;
        }
    }
};

struct CompressionResult
{
    ntc::CompressionSettings compressionSettings;
    ntc::LatentShape latentShape;
    bool compressMipChain = false;
    std::optional<bool> useGDeflate = std::nullopt;
    float bitsPerPixel = 0.f;
    float overallPSNR = 0.f;
    float perMipPSNR[NTC_MAX_MIPS]{};
    float perChannelMSE[NTC_MAX_CHANNELS]{};
    int ordinal = 0;
    float timeSeconds = 0.f;
    float experimentalKnob = 0.f;
    std::shared_ptr<std::vector<uint8_t>> compressedData;
    fs::path sourceFileName;
};

class Application : public app::ImGui_Renderer
{
private:
    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<engine::BindingCache> m_bindingCache;
    nvrhi::CommandListHandle m_commandList;
    nvrhi::CommandListHandle m_uploadCommandList;
    
    engine::ThreadPool m_threadPool;
    std::mutex m_mutex;

    bool m_cudaAvailable = false;
    ntc::ContextWrapper m_ntcContext;
    ntc::ITextureSet* m_textureSet = nullptr;
    std::vector<MaterialImage> m_images;
    int m_totalPixels = 0;
    
    int m_texturesToLoad = 0;
    int m_texturesLoaded = 0;
    int m_errors = 0;
    bool m_loading = false;
    bool m_compressing = false;
    bool m_cancel = false;
    bool m_loadedManifestFile = false;
    bool m_sharedTexturesAvailable = false;
    bool m_compareMode = false;

    std::shared_ptr<FlatImageView> m_flatImageView;
    std::shared_ptr<ModelView> m_modelView;
    std::shared_ptr<app::RegisteredFont> m_PrimaryFont = nullptr;
    std::shared_ptr<app::RegisteredFont> m_LargerFont = nullptr;
    
    int m_selectedImage = 0;
    int m_editingImage = -1;

    ntc::TextureSetDesc m_textureSetDesc;
    int m_maxOriginalWidth = 0;
    int m_maxOriginalHeight = 0;
    ntc::LatentShape m_latentShape;
    ntc::CompressionSettings m_compressionSettings;
    ntc::CompressionStats m_compressionStats;
    std::string m_leftImageName = "Reference";
    std::string m_rightImageName = "Reference";
    bool m_useLeftDecompressedImage = false;
    bool m_useRightDecompressedImage = false;
    bool m_compressedTextureSetAvailable = false;
    bool m_showCompressionProgress = true;
    int m_compressionCounter = 0;
    std::vector<CompressionResult> m_compressionResults;
    CompressionResult m_selectedCompressionResult;
    bool m_selectedCompressionResultValid = false;
    int m_alphaMaskChannelIndex = -1;
    bool m_useAlphaMaskChannel = false;
    bool m_discardMaskedOutPixels = false;
    int m_numTextureSetMips = 0;
    std::vector<SemanticBinding> m_semanticBindings;
    float m_experimentalKnob = 0.f;
    bool m_developerUI = false;
    bool m_enableGDeflate = false;

    bool m_useInt8Decompression = false;
    bool m_useGapiDecompression = false;
    bool m_useGapiDecompressionRect = false;
    ntc::Rect m_gapiDecompressionRect;
    GraphicsDecompressionPass m_decompressionPass;
    nvrhi::TimerQueryHandle m_timerQuery;
    std::optional<Manifest> m_manifest;


public:
    Application(app::DeviceManager* deviceManager)
        : ImGui_Renderer(deviceManager)
        , m_decompressionPass(GetDevice(), NTC_MAX_CHANNELS * NTC_MAX_MIPS)
    {
        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), nullptr, fs::path());

        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);

        m_bindingCache = std::make_shared<engine::BindingCache>(GetDevice());

        m_flatImageView = std::make_shared<FlatImageView>(m_bindingCache, m_commonPasses, m_shaderFactory, GetDevice());

        m_modelView = std::make_shared<ModelView>(m_commonPasses, m_shaderFactory, GetDevice());

        auto commandListParams = nvrhi::CommandListParameters()
            .setEnableImmediateExecution(false);
        m_commandList = GetDevice()->createCommandList(commandListParams);
        m_uploadCommandList = GetDevice()->createCommandList(commandListParams);

        m_timerQuery = GetDevice()->createTimerQuery();

        ImGui::GetIO().IniFilename = nullptr;
    }

    ~Application() override
    {
        m_cancel = true;
        m_threadPool.WaitForTasks();
        
        GetDevice()->waitForIdle();

        ClearImages();

        if (m_textureSet)
            m_ntcContext->DestroyTextureSet(m_textureSet);
    }
    
    bool Init()
    {
        ntc::ContextParameters contextParams;
        contextParams.cudaDevice = g_options.cudaDevice;
        contextParams.graphicsApi = GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12
            ? ntc::GraphicsAPI::D3D12
            : ntc::GraphicsAPI::Vulkan;

        bool const osSupportsCoopVec = (contextParams.graphicsApi == ntc::GraphicsAPI::D3D12)
            ? IsDX12DeveloperModeEnabled()
            : true;

        contextParams.d3d12Device = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        contextParams.vkInstance = GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
        contextParams.vkPhysicalDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
        contextParams.vkDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_Device);
        contextParams.enableCooperativeVector = osSupportsCoopVec && g_options.enableCoopVec;

        ntc::Status ntcStatus = ntc::CreateContext(m_ntcContext.ptr(), contextParams);
        if (ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable)
        {
            log::error("Failed to create an NTC context, code = %s: %s",
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        if (ntcStatus == ntc::Status::Ok)
            m_cudaAvailable = true;
        else
            m_useGapiDecompression = true;

        if (!ImGui_Renderer::Init(m_shaderFactory))
            return false;

        void const* pFontData;
        size_t fontSize;
        GetNvidiaSansFont(&pFontData, &fontSize);
        m_PrimaryFont = CreateFontFromMemoryCompressed(pFontData, fontSize, 16.f);
        m_LargerFont = CreateFontFromMemoryCompressed(pFontData, fontSize, 22.f);

        m_compareMode = g_options.compare;

        // Begin loading the inputs specified on the command line.
        // The type of inputs and their consistency is validated in ProcessCommandLine.
        switch (g_options.inputType)
        {
            case ToolInputType::Directory:
                assert(!g_options.sourcePaths.empty());
                BeginLoadingImagesFromDirectory(g_options.sourcePaths[0]);
                break;
            case ToolInputType::Images:
                BeginLoadingImagesFromFileList(g_options.sourcePaths);
                break;
            case ToolInputType::ManifestFile:
                assert(!g_options.sourcePaths.empty());
                BeginLoadingImagesFromManifest(g_options.sourcePaths[0]);
                break;
            case ToolInputType::CompressedTextureSet:
                assert(!g_options.sourcePaths.empty());
                m_manifest = std::nullopt;
                CompressionResult* result = LoadCompressedTextureSet(g_options.sourcePaths[0], true);
                if (result)
                {
                    RestoreCompressedTextureSet(*result, /* useRightTextures = */ false);
                }
                break;
        }

        return true;
    }

    bool CreateImagesFromCompressedTextureSet(ntc::ITextureSetMetadata* textureSetMetadata)
    {
        assert(textureSetMetadata);
        ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
        int const numTextures = textureSetMetadata->GetTextureCount();

        ClearImages();
        
        bool useSharedTextures = !g_options.noshared;

        for (int index = 0; index < numTextures; ++index)
        {
            ntc::ITextureMetadata const* textureMetadata = textureSetMetadata->GetTexture(index);

            MaterialImage image{};
            image.name = textureMetadata->GetName();
            image.isSRGB = textureMetadata->GetRgbColorSpace() == ntc::ColorSpace::sRGB;
            textureMetadata->GetChannels(image.firstChannel, image.channels);
            image.format = textureMetadata->GetChannelFormat();
            image.bcFormat = textureMetadata->GetBlockCompressedFormat();
            image.width = textureSetDesc.width;
            image.height = textureSetDesc.height;
            
            int firstChannel, numChannels;
            textureMetadata->GetChannels(firstChannel, numChannels);
            image.channels = numChannels;

            image.ComputeUncompressedSize();

            if (!image.CreateTextures(GetDevice(), /* createReferenceTexture = */ false, !g_options.noshared,
                image.width, image.height, textureSetDesc.mips))
                return false;
            
            if (useSharedTextures)
            {
                if (!RegisterSharedTextures(image))
                    useSharedTextures = false;
            }

            m_images.push_back(std::move(image));
        }

        m_sharedTexturesAvailable = useSharedTextures;

        return true;
    }

    CompressionResult* LoadCompressedTextureSet(const char* fileName, bool createImagesIfEmpty)
    {
        ntc::FileStreamWrapper inputFile(m_ntcContext);
        ntc::Status ntcStatus = m_ntcContext->OpenFile(fileName, false, inputFile.ptr());
        if (ntcStatus != ntc::Status::Ok)
        {
            log::error("Failed to open input file '%s', error code = %s: %s", fileName,
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return nullptr;
        }
        
        ntc::TextureSetMetadataWrapper metadata(m_ntcContext);
        ntcStatus = m_ntcContext->CreateTextureSetMetadataFromStream(inputFile, metadata.ptr());
        if (ntcStatus != ntc::Status::Ok)
        {
            log::error("Failed to load input file '%s', error code = %s: %s", fileName,
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return nullptr;
        }

        int const maxImageDimension = 16384;
        ntc::TextureSetDesc const& textureSetDesc = metadata->GetDesc();
        if (textureSetDesc.width > maxImageDimension || textureSetDesc.height > maxImageDimension)
        {
            log::error("Cannot load input file '%s' because the textures stored in it are too large for "
                "graphics API usage. The texture set is %dx%d pixels, and maximum supported size is %dx%d.",
                fileName, textureSetDesc.width, textureSetDesc.height, maxImageDimension, maxImageDimension);
            return nullptr;
        }

        if (!m_images.empty())
        {
            std::unordered_set<std::string> missingImageNames;
            std::unordered_set<std::string> extraImageNames;
            for (MaterialImage const& image : m_images)
                missingImageNames.insert(image.name);
                
            int const texturesInSet = metadata->GetTextureCount();
            for (int index = 0; index < texturesInSet; ++index)
            {
                std::string textureName = metadata->GetTexture(index)->GetName();
                auto it = missingImageNames.find(textureName);
                if (it == missingImageNames.end())
                    extraImageNames.insert(textureName);
                else
                    missingImageNames.erase(it);
            }

            if (!extraImageNames.empty() || !missingImageNames.empty())
            {
                std::stringstream ss;
                ss << "The compressed texture set contains textures that do not match the loaded reference images.\n";
                if (!extraImageNames.empty())
                {
                    ss << "Extra textures:\n";
                    for (std::string const& name : extraImageNames)
                    {
                        ss << " - " << name << "\n";
                    }
                }
                if (!missingImageNames.empty())
                {
                    ss << "Missing textures:\n";
                    for (std::string const& name : missingImageNames)
                    {                        
                        ss << " - " << name << "\n";
                    }
                }

                log::error("%s", ss.str().c_str());
                return nullptr;
            }
        }

        ntc::TextureSetDesc const& desc = metadata->GetDesc();

        if (createImagesIfEmpty && m_images.empty())
        {
            if (!CreateImagesFromCompressedTextureSet(metadata))
                return nullptr;
                
            m_textureSetDesc = desc;
        }

        uint64_t const fileSize = inputFile->Size();
        CompressionResult result;
        result.compressedData = std::make_shared<std::vector<uint8_t>>(fileSize);
        inputFile->Seek(0);
        inputFile->Read(result.compressedData->data(), fileSize);
        result.compressMipChain = desc.mips > 1;
        result.bitsPerPixel = float(fileSize) / float(desc.width * desc.height);
        if (result.compressMipChain)
            result.bitsPerPixel /= 1.333f;
        result.latentShape = metadata->GetLatentShape();
        result.ordinal = ++m_compressionCounter;
        result.sourceFileName = fileName;
        m_compressionResults.push_back(result);
        return &m_compressionResults.back();
    }

    bool BeginLoadingImagesFromDirectory(const char* path)
    {
        Manifest manifest;
        GenerateManifestFromDirectory(path, false, manifest);
        if (manifest.textures.empty())
        {
            log::error("The folder '%s' contains no compatible image files.", path);
            return false;
        }

        if (manifest.textures.size() > NTC_MAX_CHANNELS)
        {
            log::error("Too many images (%d) found in the input folder. At most %d channels are supported.\n"
                "Note: when loading images from a folder, a single material with all images is created. "
                "To load a material with only some images from a folder, use manifest files.",
                int(manifest.textures.size()), NTC_MAX_CHANNELS);
            return false;
        }

        m_loadedManifestFile = false;
        BeginLoadingImages(manifest);
        return true;
    }

    bool BeginLoadingImagesFromFileList(std::vector<char const*> const& files)
    {
        Manifest manifest;
        GenerateManifestFromFileList(files, manifest);

        if (manifest.textures.size() > NTC_MAX_CHANNELS)
        {
            log::error("Too many images (%d) specified. At most %d channels are supported.",
                int(manifest.textures.size()), NTC_MAX_CHANNELS);
            return false;
        }

        m_loadedManifestFile = false;
        BeginLoadingImages(manifest);
        return true;
    }

    bool BeginLoadingImagesFromManifest(const char* manifestFileName)
    {
        Manifest manifest;
        std::string errorMessage;
        if (!ReadManifestFromFile(manifestFileName, manifest, errorMessage))
        {
            log::error("%s", errorMessage.c_str());
            return false;
        }

        if (manifest.textures.size() > NTC_MAX_CHANNELS)
        {
            log::error("Too many images (%d) specified in the manifest. At most %d channels are supported.",
                int(manifest.textures.size()), NTC_MAX_CHANNELS);
            return false;
        }

        m_loadedManifestFile = true;
        BeginLoadingImages(manifest);
        return true;
    }

    bool ProcessChannelSwizzle(MaterialImage& image, std::string const& channelSwizzle)
    {
        if (channelSwizzle.empty())
            return true;
        
        // Init the channel map, 4 means "store 0"
        int swizzle[4] = { 4, 4, 4, 4 };
        // Size of the 'srcPixel' arrays below, 5 because element 4 stores 0
        constexpr int srcPixelSize = 5;

        for (size_t i = 0; i < channelSwizzle.size(); ++i)
        {
            // Decode the channel letter into an offset using a lookup string
            char const* channelMap = "RGBA";
            char const* channelPos = strchr(channelMap, channelSwizzle[i]);
            
            if (!channelPos)
            {
                // The format of 'channelSwizzle' is validated when the manifest is loaded,
                // so 'channelPos' should never be NULL here.
                assert(false);
                return false;
            }

            swizzle[i] = channelPos - channelMap;
        }
        
        // We always create 4-channel images because we upload to 4-component textures later
        int const oldChannels = 4;
        int const newChannels = 4;

        // Swizzle the image data in-place.
        // We can do this because we always use 4 components per pixel, and don't change the component format.
        size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(image.format);
        switch(bytesPerComponent)
        {
            case 1: {
                uint8_t* imageData = image.data.get();
                uint8_t srcPixel[srcPixelSize] = { };
                for (int row = 0; row < image.height; ++row)
                {
                    for (int col = 0; col < image.width; ++col)
                    {
                        for (int c = 0; c < oldChannels; ++c)
                        {
                            srcPixel[c] = imageData[c];
                        }
                        for (int c = 0; c < newChannels; ++c)
                        {
                            int sw = swizzle[c];
                            imageData[c] = srcPixel[sw];
                        }

                        imageData += 4;
                    }
                }
                break;
            }
            case 2: {
                uint16_t* imageData = (uint16_t*)image.data.get();
                uint16_t srcPixel[srcPixelSize] = { };
                for (int row = 0; row < image.height; ++row)
                {
                    for (int col = 0; col < image.width; ++col)
                    {
                        for (int c = 0; c < oldChannels; ++c)
                        {
                            srcPixel[c] = imageData[c];
                        }
                        for (int c = 0; c < newChannels; ++c)
                        {
                            int sw = swizzle[c];
                            ((uint16_t*)imageData)[c] = srcPixel[sw];
                        }

                        imageData += 4;
                    }
                }
                break;
            }
            case 4: {
                uint32_t* imageData = (uint32_t*)image.data.get();
                uint32_t srcPixel[srcPixelSize] = { };
                for (int row = 0; row < image.height; ++row)
                {
                    for (int col = 0; col < image.width; ++col)
                    {
                        for (int c = 0; c < oldChannels; ++c)
                        {
                            srcPixel[c] = imageData[c];
                        }
                        for (int c = 0; c < newChannels; ++c)
                        {
                            int sw = swizzle[c];
                            ((uint32_t*)imageData)[c] = srcPixel[sw];
                        }

                        imageData += 4;
                    }
                }
                break;
            }
            default:
                assert(false); // What is component size that is not 1, 2 or 4 bytes?
                break;
        }

        // Store the actual number of valid channels in the image
        image.channels = int(channelSwizzle.size());

        return true;
    }

    void VerticalFlip(MaterialImage& image)
    {
        size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(image.format);
        
        // Note: allocating for 4 components because we always use 4-component images here
        size_t const rowPitch = bytesPerComponent * size_t(image.width) * 4;
        
        // Allocate memory for flipped image data. Can't (quickly) flip in-place.
        std::shared_ptr<uint8_t> newData = std::shared_ptr<uint8_t>((uint8_t*)malloc(rowPitch * size_t(image.height)));

        // Copy image rows into new locations
        for (int row = 0; row < image.height; ++row)
        {
            uint8_t const* src = image.data.get() + row * rowPitch;
            uint8_t* dst = newData.get() + (image.height - row - 1) * rowPitch;
            memcpy(dst, src, rowPitch);
        }

        // Replace the image data with flipped data
        image.data = newData;
    }

    void BeginLoadingImages(Manifest const& manifest)
    {
        m_loading = true;
        ClearImages();
        bool isFirstFile = true;

        m_manifest = manifest;
        int manifestIndex = 0;

        for (ManifestEntry const& entry : manifest.textures)
        {
            ++m_texturesToLoad;

            m_threadPool.AddTask([this, entry, manifestIndex]()
            {
                MaterialImage image{};

                std::string extension = fs::path(entry.fileName).extension().generic_string();
                LowercaseString(extension);
                if (extension == ".exr")
                {
                    LoadEXR((float**)&image.data, &image.width, &image.height, entry.fileName.c_str(), nullptr);
                    image.channels = 4;
                    image.format = ntc::ChannelFormat::FLOAT32;
                }
                else
                {
                    FILE* imageFile = fopen(entry.fileName.c_str(), "rb");
                    if (imageFile)
                    {
                        bool is16bit = stbi_is_16_bit_from_file(imageFile);
                        fseek(imageFile, 0, SEEK_SET);

                        if (is16bit)
                        {
                            image.data = std::shared_ptr<stbi_uc>((stbi_uc*)stbi_load_from_file_16(imageFile, &image.width, &image.height, &image.channels, STBI_rgb_alpha));
                            image.format = ntc::ChannelFormat::UNORM16;
                        }
                        else
                        {
                            image.data = std::shared_ptr<stbi_uc>(stbi_load_from_file(imageFile, &image.width, &image.height, &image.channels, STBI_rgb_alpha));
                            image.format = ntc::ChannelFormat::UNORM8;
                        }

                        fclose(imageFile);
                    }
                }

                // The rest of this function is interlocked with other threads
                std::lock_guard lockGuard(m_mutex);

                if (!image.data)
                {
                    log::warning("Failed to read image '%s'.\n", entry.fileName.c_str());
                    ++m_errors;
                    return;
                }

                // Apply channel swizzle during loading, not with WriteChannels tricks like ntc-cli does:
                // we want the reference graphics texture to also be swizzled.
                if (!ProcessChannelSwizzle(image, entry.channelSwizzle))
                {
                    ++m_errors;
                    return;
                }

                if (entry.verticalFlip)
                {
                    // Apply vertical flip during loading, not using the NTC WriteChannels feature:
                    // we want the reference graphics texture to also be flipped.
                    VerticalFlip(image);
                }

                image.name = entry.entryName;
                image.isSRGB = entry.isSRGB;
                image.manifestIndex = manifestIndex;
                image.manifestSemantics = entry.semantics;
                image.bcFormat = entry.bcFormat;
                image.ComputeUncompressedSize();

                // Process loss function scales and expand to per-channel if needed
                if (entry.lossFunctionScales.size() == 1)
                {
                    image.lossFunctionScales.resize(image.channels, entry.lossFunctionScales[0]);
                }
                else if (entry.lossFunctionScales.size() == size_t(image.channels))
                {
                    image.lossFunctionScales = entry.lossFunctionScales;
                }
                else
                {
                    image.lossFunctionScales.resize(image.channels, 1.f);

                    if (!entry.lossFunctionScales.empty())
                    {
                        log::warning("Ignoring invalid loss function scales for image '%s'.\n", entry.fileName.c_str());
                    }
                }
                assert(image.lossFunctionScales.size() == size_t(image.channels));

                m_images.push_back(std::move(image));

                ++m_texturesLoaded;
            });

            ++manifestIndex;
        }
    }

    bool IsModelViewActive() const
    {
        return m_selectedImage < 0;
    }

    void NewTexturesLoaded()
    {
        // Make the 2D view fit the new textures to the window
        m_flatImageView->Reset(m_compareMode);

        // Select the albedo texture, if this semantic is defined, otherwise the first one
        m_selectedImage = 0;
        if (!m_compareMode)
        {
            for (auto& semantic : m_semanticBindings)
            {
                if (semantic.label == SemanticLabel::Albedo)
                    m_selectedImage = semantic.imageIndex;
            }
        }

        m_editingImage = -1;
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        // ImGui doesn't recognize the keypad Enter key, and that's annoying.
        // Map it to the regular Enter key.
        if (key == GLFW_KEY_KP_ENTER)
            key = GLFW_KEY_ENTER;

        return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        if (ImGui_Renderer::MousePosUpdate(xpos, ypos))
            return true;

        if (IsModelViewActive())
            return m_modelView->MousePosUpdate(xpos, ypos);

        return m_flatImageView->MousePosUpdate(xpos, ypos);
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        if (ImGui_Renderer::MouseButtonUpdate(button, action, mods))
            return true;

        if (IsModelViewActive())
            return m_modelView->MouseButtonUpdate(button, action, mods);

        return m_flatImageView->MouseButtonUpdate(button, action, mods);
    }

    bool MouseScrollUpdate(double xoffset, double yoffset) override
    {
        if (ImGui_Renderer::MouseScrollUpdate(xoffset, yoffset))
            return true;

        if (IsModelViewActive())
            return m_modelView->MouseScrollUpdate(xoffset, yoffset);

        return m_flatImageView->MouseScrollUpdate(xoffset, yoffset);
    }

    void BackBufferResizing() override
    { 
        ImGui_Renderer::BackBufferResizing();
    }

    void GenerateReferenceMips(nvrhi::ICommandList* commandList, nvrhi::ITexture* texture, bool isSRGB)
    {
        if (!texture)
            return;

        const auto& desc = texture->getDesc();

        nvrhi::Format nvrhiFormat = desc.format;
        if (isSRGB && nvrhiFormat == nvrhi::Format::RGBA8_UNORM)
            nvrhiFormat = nvrhi::Format::SRGBA8_UNORM;
        
        for (int mip = 1; mip < int(desc.mipLevels); ++mip)
        {
            const nvrhi::FramebufferDesc framebufferDesc = nvrhi::FramebufferDesc()
                .addColorAttachment(nvrhi::FramebufferAttachment()
                    .setTexture(texture)
                    .setSubresources(nvrhi::TextureSubresourceSet(mip, 1, 0, 1))
                    .setFormat(nvrhiFormat));

            nvrhi::FramebufferHandle framebuffer = GetDevice()->createFramebuffer(framebufferDesc);

            engine::BlitParameters blitParams;
            blitParams.sourceTexture = texture;
            blitParams.sourceMip = mip - 1;
            blitParams.sourceFormat = nvrhiFormat;
            blitParams.targetFramebuffer = framebuffer;
            blitParams.targetViewport.maxX = float(std::max(desc.width >> mip, 1u));
            blitParams.targetViewport.maxY = float(std::max(desc.height >> mip, 1u));

            m_commonPasses->BlitTexture(commandList, blitParams, m_bindingCache.get());
        }
    }

    bool RegisterSharedTextures(MaterialImage& image)
    {
        if (g_options.noshared || m_compareMode)
            return false;

        if (!m_cudaAvailable)
            return false;

        if (image.referenceTextureShared)
        {
            m_ntcContext->ReleaseSharedTexture(image.referenceTextureShared);
            image.referenceTextureShared = nullptr;
        }

        if (image.decompressedTextureLeftShared)
        {
            m_ntcContext->ReleaseSharedTexture(image.decompressedTextureLeftShared);
            image.decompressedTextureLeftShared = nullptr;
        }

        if (image.decompressedTextureRightShared)
        {
            m_ntcContext->ReleaseSharedTexture(image.decompressedTextureRightShared);
            image.decompressedTextureRightShared = nullptr;
        }

        ntc::SharedTextureDesc sharedTextureDesc;
        sharedTextureDesc.channels = 4;
        sharedTextureDesc.format = image.format;
        sharedTextureDesc.dedicatedResource = true;
#ifdef _WIN32
        sharedTextureDesc.handleType = GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN
            ? ntc::SharedHandleType::OpaqueWin32
            : ntc::SharedHandleType::D3D12Resource;
#else
        sharedTextureDesc.handleType = ntc::SharedHandleType::OpaqueFd;
#endif

        if (image.referenceTexture)
        {
            const nvrhi::TextureDesc& referenceDesc = image.referenceTexture->getDesc();
            sharedTextureDesc.width = referenceDesc.width;
            sharedTextureDesc.height = referenceDesc.height;
            sharedTextureDesc.mips = referenceDesc.mipLevels;

            // Register the reference texture

            sharedTextureDesc.sizeInBytes = GetDevice()->getTextureMemoryRequirements(image.referenceTexture).size;
            sharedTextureDesc.sharedHandle = image.referenceTexture->getNativeObject(nvrhi::ObjectTypes::SharedHandle).integer;
                    
            ntc::Status ntcStatus = m_ntcContext->RegisterSharedTexture(sharedTextureDesc, &image.referenceTextureShared);
            if (ntcStatus != ntc::Status::Ok)
            {
                log::warning("Call to RegisterSharedTexture failed, code = %s: %s", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                return false;
            }
        }
                
        // Register the decompressed textures

        const nvrhi::TextureDesc& decompressedDesc = image.decompressedTextureLeft->getDesc();
        sharedTextureDesc.width = decompressedDesc.width;
        sharedTextureDesc.height = decompressedDesc.height;
        sharedTextureDesc.mips = decompressedDesc.mipLevels;
        sharedTextureDesc.sizeInBytes = GetDevice()->getTextureMemoryRequirements(image.decompressedTextureLeft).size;
        sharedTextureDesc.sharedHandle = image.decompressedTextureLeft->getNativeObject(nvrhi::ObjectTypes::SharedHandle).integer;
                
        ntc::Status ntcStatus = m_ntcContext->RegisterSharedTexture(sharedTextureDesc, &image.decompressedTextureLeftShared);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Call to RegisterSharedTexture failed, code = %s: %s", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        sharedTextureDesc.sizeInBytes = GetDevice()->getTextureMemoryRequirements(image.decompressedTextureRight).size;
        sharedTextureDesc.sharedHandle = image.decompressedTextureRight->getNativeObject(nvrhi::ObjectTypes::SharedHandle).integer;

        ntcStatus = m_ntcContext->RegisterSharedTexture(sharedTextureDesc, &image.decompressedTextureRightShared);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Call to RegisterSharedTexture failed, code = %s: %s", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        return true;
    }

    void ClearImages()
    {
        m_semanticBindings.clear();
        m_compressionResults.clear();
        m_bindingCache->Clear();
        m_useLeftDecompressedImage = false;
        m_useRightDecompressedImage = false;
        m_texturesLoaded = 0;
        m_texturesToLoad = 0;
        m_selectedImage = 0;
        m_editingImage = -1;
        m_compressionCounter = 0;
        m_manifest = std::nullopt;
        
        for (auto& image : m_images)
        {
            if (image.referenceTextureShared)
                m_ntcContext->ReleaseSharedTexture(image.referenceTextureShared);

            if (image.decompressedTextureLeftShared)
                m_ntcContext->ReleaseSharedTexture(image.decompressedTextureLeftShared);

            if (image.decompressedTextureRightShared)
                m_ntcContext->ReleaseSharedTexture(image.decompressedTextureRightShared);
        }

        m_images.clear();
    }

    void UploadTextures()
    {
        std::sort(m_images.begin(), m_images.end(), [](const MaterialImage& a, const MaterialImage& b) {
            return a.name < b.name;
        });

        m_textureSetDesc.channels = 0;
        m_maxOriginalWidth = 0;
        m_maxOriginalHeight = 0;

        // Gather the texture dimensions to determine the texture set parameters.
        // This should be done before creating the texture objects because the decompressed textures
        // must have the same dimensions as the texture set, not as the reference textures.
        for (MaterialImage& image : m_images)
        {
            image.firstChannel = m_textureSetDesc.channels;
            m_textureSetDesc.channels += image.channels;
            m_maxOriginalWidth = std::max(image.width, m_maxOriginalWidth);
            m_maxOriginalHeight = std::max(image.height, m_maxOriginalHeight);
        }

        // Override the texture set dimensions from the manifest, if specified
        if (m_manifest.has_value())
        {
            m_textureSetDesc.width = m_manifest->width.value_or(m_maxOriginalWidth);
            m_textureSetDesc.height = m_manifest->height.value_or(m_maxOriginalHeight);
        }

        m_numTextureSetMips = GetNumMipLevels(m_textureSetDesc.width, m_textureSetDesc.height);
        SetCompressMipChain(false);

        bool useSharedTextures = !g_options.noshared && !m_compareMode;

        // Create the texture objects and upload data into the reference textures.
        int imageIndex = 0;
        for (MaterialImage& image : m_images)
        {
            if (!m_loadedManifestFile)
            {
                // When we've enumerated files in a folder, guess the sRGB colorspace and semantics.
                GuessImageSemantics(image.name, image.channels, image.format, imageIndex,
                    image.isSRGB, m_semanticBindings);
            }
            else
            {
                // When we've used a manifest file, take the semantics from that file.
                for (ImageSemanticBinding const& binding : image.manifestSemantics)
                {
                    m_semanticBindings.push_back({ binding.label, imageIndex, binding.firstChannel });
                }
            }
            
            image.CreateTextures(GetDevice(), /* createReferenceTextures = */ true, useSharedTextures,
                m_textureSetDesc.width, m_textureSetDesc.height, m_numTextureSetMips);
            
            nvrhi::Format const textureFormat = image.referenceTexture->getDesc().format;

            m_uploadCommandList->open();
            m_uploadCommandList->writeTexture(image.referenceTexture, 0, 0, image.data.get(),
                nvrhi::getFormatInfo(textureFormat).bytesPerBlock * image.width);

            GenerateReferenceMips(m_uploadCommandList, image.referenceTexture, image.isSRGB);
            image.referenceMipsValid = true;

            m_uploadCommandList->close();

            GetDevice()->executeCommandList(m_uploadCommandList);
            GetDevice()->waitForIdle();
            GetDevice()->runGarbageCollection();

            if (useSharedTextures)
            {
                if (!RegisterSharedTextures(image))
                {
                    // If one texture failed to register, don't try others - we'll not use sharing anyway,
                    // and the user will get fewer error messages.
                    useSharedTextures = false;
                }
            }

            ++imageIndex;
        }

        if (m_compareMode)
        {
            {
                MaterialImage& leftImage = m_images[0];
                MaterialImage& rightImage = m_images[1];
                leftImage.decompressedTextureLeft = leftImage.referenceTexture;
                leftImage.decompressedTextureRight = rightImage.referenceTexture;
                leftImage.referenceTexture = nullptr;
                m_leftImageName = leftImage.name;
                m_rightImageName = rightImage.name;
            }

            m_useLeftDecompressedImage = true;
            m_useRightDecompressedImage = true;

            m_images.resize(1);
        }

        m_sharedTexturesAvailable = useSharedTextures;
    }

    void SetCompressMipChain(bool compress)
    {
        m_textureSetDesc.mips = compress ? m_numTextureSetMips : 1;

        // Find out the total number of pixels in all mips to calculate the compression ratios later
        m_totalPixels = 0;
        for (int mip = 0; mip < m_textureSetDesc.mips; ++mip)
        {
            const int mipWidth = std::max(m_textureSetDesc.width >> mip, 1);
            const int mipHieght = std::max(m_textureSetDesc.height >> mip, 1);
            m_totalPixels += mipWidth * mipHieght;
        }
    }

#define CHECK_NTC_RESULT(fname) \
    if (ntcStatus != ntc::Status::Ok) { \
        log::error("Call to " #fname " failed, code = %s: %s\n", ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage()); \
        return false; }
//#end-define
#define CHECK_CANCEL(doAbort) if (m_cancel) { \
    if ((doAbort) && m_textureSet) \
        m_textureSet->AbortCompression(); \
        return false; }
//#end-define

    ntc::Status DecompressWithGapi(ntc::IStream* inputStream, size_t inputSize, bool useRightTextures)
    {
        ntc::TextureSetMetadataWrapper metadata(m_ntcContext);

        ntc::Status ntcStatus = m_ntcContext->CreateTextureSetMetadataFromStream(inputStream, metadata.ptr());

        if (ntcStatus != ntc::Status::Ok)
            return ntcStatus;

        if (!m_decompressionPass.Init())
            return ntc::Status::InternalError;

        // Write UAV descriptors for all necessary mip levels into the descriptor table
        for (int mipLevel = 0; mipLevel < metadata->GetDesc().mips; ++mipLevel)
        {
            for (int index = 0; index < int(m_images.size()); ++index)
            {
                nvrhi::TextureHandle texture = useRightTextures
                    ? m_images[index].decompressedTextureRight
                    : m_images[index].decompressedTextureLeft;

                const auto bindingSetItem = nvrhi::BindingSetItem::Texture_UAV(
                    mipLevel * int(m_images.size()) + index,
                    texture,
                    nvrhi::Format::UNKNOWN,
                    nvrhi::TextureSubresourceSet(mipLevel, 1, 0, 1));

                m_decompressionPass.WriteDescriptor(bindingSetItem);
            }
        }

        ntc::TextureSetDesc const& textureSetDesc = metadata->GetDesc();

        ntc::InferenceWeightType weightType = metadata->GetBestSupportedWeightType();
        if (weightType == ntc::InferenceWeightType::Unknown)
            return ntc::Status::Unsupported;
        
        // Upload the latents and weights
        if (!m_decompressionPass.SetLatentDataFromTextureSet(m_commandList, m_ntcContext.Get(),
            nullptr, inputStream, metadata))
        {
            return ntc::Status::InternalError;
        }
        m_commandList->open();
        if (!m_decompressionPass.SetWeightsFromTextureSet(m_commandList, metadata, weightType))
        {
            m_commandList->close();
            return ntc::Status::InternalError;
        }

        // Begin the decompression region
        m_commandList->beginMarker("Decompress");
        m_commandList->beginTimerQuery(m_timerQuery);
        
        // Decompress each mip level in a loop
        for (int mipLevel = 0; mipLevel < metadata->GetDesc().mips; ++mipLevel)
        {
            // Obtain the compute pass description and constant buffer data from NTC
            ntc::ComputePassDesc computePass{};
            ntc::MakeDecompressionComputePassParameters params;
            params.textureSetMetadata = metadata;
            params.mipLevel = mipLevel;
            params.firstOutputDescriptorIndex = mipLevel * int(m_images.size());
            params.pSrcRect = m_useGapiDecompressionRect ? &m_gapiDecompressionRect : nullptr;
            params.weightType = weightType;
            ntcStatus = m_ntcContext->MakeDecompressionComputePass(params, &computePass);

            // On failure, close/abandon the command list and return
            if (ntcStatus != ntc::Status::Ok)
            {
                m_commandList->endTimerQuery(m_timerQuery);
                m_commandList->close();
                return ntcStatus;
            }

            // Set a marker around the mip level, if the level is large enough.
            // Small mips can be evaluated simultaneously by the GPU, but markers prevent that.
            int const mipWidth = metadata->GetDesc().width >> mipLevel;
            int const mipHeight = metadata->GetDesc().height >> mipLevel;
            bool const useMarker = mipWidth * mipHeight > 512 * 512;
            if (useMarker)
            {
                char markerName[16];
                snprintf(markerName, sizeof(markerName), "Mip %d", mipLevel);
                m_commandList->beginMarker(markerName);
            }

            if (!m_decompressionPass.ExecuteComputePass(m_commandList, computePass))
            {
                m_commandList->endTimerQuery(m_timerQuery);
                m_commandList->close();
                return ntc::Status::InternalError;
            }

            if (useMarker)
                m_commandList->endMarker();
        }

        // End the timer query, close and execute the CL
        m_commandList->endTimerQuery(m_timerQuery);
        m_commandList->endMarker();
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        float seconds = GetDevice()->getTimerQueryTime(m_timerQuery);
        log::info("Decompression time: %.2f ms", seconds * 1e3f);

        if (useRightTextures)
            m_useRightDecompressedImage = true;
        else
            m_useLeftDecompressedImage = true;

        return ntc::Status::Ok;
    }

    ntc::LosslessCompressionSettings GetLosslessCompressionParams() const
    {
        ntc::LosslessCompressionSettings params;
        params.algorithm = ntc::CompressionType::GDeflate;
        params.compressBCModeBuffers = true; // We don't make those here anyway
        params.compressLatents = m_enableGDeflate;
        return params;
    }

    bool DecompressIntoTextures(bool recordResults, bool useRightTextures, bool enableInt8, time_point<steady_clock> beginTime)
    {
        if (!m_cudaAvailable)
            return false;

        m_textureSet->SetExperimentalKnob(m_experimentalKnob);

        ntc::DecompressionStats stats;
        ntc::Status ntcStatus = m_textureSet->Decompress(&stats, m_useInt8Decompression && enableInt8);
        CHECK_NTC_RESULT(Decompress);
        CHECK_CANCEL(false);

        if (recordResults)
        {
            const auto& textureSetDesc = m_textureSet->GetDesc();

            CompressionResult result;
            result.latentShape = m_textureSet->GetLatentShape();
            result.overallPSNR = ntc::LossToPSNR(stats.overallLoss);
            result.compressionSettings = m_compressionSettings;
            result.compressMipChain = m_textureSetDesc.mips > 1;
            for (int mip = 0; mip < m_textureSetDesc.mips; ++mip)
                result.perMipPSNR[mip] = ntc::LossToPSNR(stats.perMipLoss[mip]);
            for (int channel = 0; channel < textureSetDesc.channels; ++channel)
                result.perChannelMSE[channel] = stats.perChannelLoss[channel];
            result.experimentalKnob = m_experimentalKnob;
            result.ordinal = ++m_compressionCounter;
            
            time_point endTime = steady_clock::now();
            result.timeSeconds = float(duration_cast<microseconds>(endTime - beginTime).count()) * 1e-6f;

            size_t bufferSize = m_textureSet->GetOutputStreamSize();
            result.compressedData = std::make_shared<std::vector<uint8_t>>(bufferSize);

            result.useGDeflate = m_enableGDeflate;

            ntcStatus = m_textureSet->ConfigureLosslessCompression(GetLosslessCompressionParams());
            CHECK_NTC_RESULT(ConfigureLosslessCompression);

            ntcStatus = m_textureSet->SaveToMemory(result.compressedData->data(), &bufferSize);
            CHECK_NTC_RESULT(SaveToMemory);

            // Trim the buffer to the actual size of the saved data
            result.compressedData->resize(bufferSize);
            result.bitsPerPixel = float(double(bufferSize) * 8.0 / double(m_totalPixels));
            
            // The rest of this function is interlocked with other threads
            std::lock_guard lock(m_mutex);
            m_compressionResults.push_back(result);
        }
        
        const bool useSharedTextures = !g_options.noshared;

        int const texturesInSet = m_textureSet->GetTextureCount();
        assert(texturesInSet == m_images.size()); // Validated when loading the file, or equal by definition if the texture was just compressed

        for (auto& image : m_images)
        {
            size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(image.format);
            size_t const pixelStride = 4 * bytesPerComponent;

            nvrhi::TextureHandle decompressedTexture = useRightTextures ? image.decompressedTextureRight : image.decompressedTextureLeft;
            ntc::ISharedTexture*& decompressedTextureSharedRef = useRightTextures ? image.decompressedTextureRightShared : image.decompressedTextureLeftShared;

            ntc::ITextureMetadata const* compressedTexture = nullptr;
            for (int index = 0; index < texturesInSet; ++index)
            {
                ntc::ITextureMetadata const* compressedTextureCandidate = m_textureSet->GetTexture(index);
                if (image.name == compressedTextureCandidate->GetName())
                {
                    compressedTexture = compressedTextureCandidate;
                    break;
                }
            }

            if (!compressedTexture)
            {
                log::error("Cannot find a texture with name '%s' in the texture set. This should never happen though...", image.name.c_str());
                return false;
            }

            int compressedFirstChannel, compressedNumChannels;
            compressedTexture->GetChannels(compressedFirstChannel, compressedNumChannels);

            nvrhi::TextureDesc const& textureDesc = decompressedTexture->getDesc();
            int const effectiveMips = std::min(m_textureSetDesc.mips, int(textureDesc.mipLevels));
            
            ntc::ColorSpace const rgbColorSpace = image.isSRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
            ntc::ColorSpace const alphaColorSpace = ntc::ColorSpace::Linear;
            ntc::ColorSpace const colorSpaces[4] = { rgbColorSpace, rgbColorSpace, rgbColorSpace, alphaColorSpace };

            if (useSharedTextures && decompressedTextureSharedRef)
            {
                for (int mip = 0; mip < effectiveMips; ++mip)
                {
                    ntc::ReadChannelsIntoTextureParameters params;
                    params.page = ntc::TextureDataPage::Output;
                    params.mipLevel = mip;
                    params.firstChannel = compressedFirstChannel;
                    params.numChannels = compressedNumChannels;
                    params.texture = decompressedTextureSharedRef;
                    params.textureMipLevel = mip;
                    params.dstRgbColorSpace = rgbColorSpace;
                    params.dstAlphaColorSpace = alphaColorSpace;
                    params.useDithering = true;

                    ntcStatus = m_textureSet->ReadChannelsIntoTexture(params);
                    CHECK_NTC_RESULT(ReadChannelsIntoTexture);
                    CHECK_CANCEL(false);
                }
            }
            else
            {
                if (!image.decompressedData)
                    image.decompressedData = std::shared_ptr<uint8_t>((uint8_t*)malloc(textureDesc.width * textureDesc.height * pixelStride));

                m_uploadCommandList->open();

                for (int mip = 0; mip < effectiveMips; ++mip)
                {
                    int const mipWidth = std::max(int(textureDesc.width) >> mip, 1);
                    int const mipHeight = std::max(int(textureDesc.height) >> mip, 1);

                    ntc::ReadChannelsParameters params;
                    params.page = ntc::TextureDataPage::Output;
                    params.mipLevel = mip;
                    params.firstChannel = compressedFirstChannel;
                    params.numChannels = compressedNumChannels;
                    params.pOutData = image.decompressedData.get();
                    params.addressSpace = ntc::AddressSpace::Host;
                    params.width = mipWidth;
                    params.height = mipHeight;
                    params.pixelStride = pixelStride;
                    params.rowPitch = size_t(mipWidth) * pixelStride;
                    params.channelFormat = image.format;
                    params.dstColorSpaces = colorSpaces;
                    params.useDithering = true;

                    ntcStatus = m_textureSet->ReadChannels(params);
                    CHECK_NTC_RESULT(ReadChannels);
                    CHECK_CANCEL(false);
                    
                    m_uploadCommandList->writeTexture(decompressedTexture, 0, mip, image.decompressedData.get(), pixelStride * mipWidth);
                }
                m_uploadCommandList->close();

                GetDevice()->executeCommandList(m_uploadCommandList);
                GetDevice()->waitForIdle();
                GetDevice()->runGarbageCollection();
            }

            CHECK_CANCEL(false);
        }

        if (useRightTextures)
            m_useRightDecompressedImage = true;
        else
            m_useLeftDecompressedImage = true;

        return true;
    }

    ntc::TextureSetFeatures GetTextureSetFeatures(bool needStagingUpload)
    {
        bool sharedTexturesAvailable = true;
        for (const auto& image : m_images)
        {
            if (!image.referenceTextureShared)
            {
                sharedTexturesAvailable = false;
                break;
            }
        }

        ntc::TextureSetFeatures textureSetFeatures;
        textureSetFeatures.stagingBytesPerPixel = sharedTexturesAvailable ? 0 : sizeof(float) * 4;
        textureSetFeatures.stagingWidth = needStagingUpload ? m_maxOriginalWidth : 0;
        textureSetFeatures.stagingHeight = needStagingUpload ? m_maxOriginalHeight : 0;
        textureSetFeatures.separateRefOutData = true;

        return textureSetFeatures;
    }

    bool UploadReferenceImages(bool uploadAllTextures)
    {
        m_textureSet->ClearTextureMetadata();

        // Upload the reference texture data. This only needs to be done once
        // because NTC never overwrites the data when TextureSetFeatures::separateRefOutData is true.
        int index = 0;
        bool needToGenerateMips = false;
        ntc::Status ntcStatus;
        for (auto& image : m_images)
        {
            ntc::ColorSpace const srcRgbColorSpace = image.isSRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
            ntc::ColorSpace const dstRgbColorSpace = image.format == ntc::ChannelFormat::FLOAT32 ? ntc::ColorSpace::HLG : srcRgbColorSpace;
            ntc::ColorSpace const srcAlphaColorSpace = ntc::ColorSpace::Linear;
            ntc::ColorSpace const dstAlphaColorSpace = image.format == ntc::ChannelFormat::FLOAT32 ? ntc::ColorSpace::HLG : srcAlphaColorSpace;
            ntc::ColorSpace const srcColorSpaces[4] = { srcRgbColorSpace, srcRgbColorSpace, srcRgbColorSpace, srcAlphaColorSpace };
            ntc::ColorSpace const dstColorSpaces[4] = { dstRgbColorSpace, dstRgbColorSpace, dstRgbColorSpace, dstAlphaColorSpace };

            // Upload when we've just created the texture set, or when the user has changed the texture format
            if (uploadAllTextures || !image.textureSetDataValid)
            {
                size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(image.format);
                size_t const pixelStride = 4 * bytesPerComponent;

                if (image.referenceTextureShared)
                {
                    ntc::WriteChannelsFromTextureParameters params;
                    params.mipLevel = 0;
                    params.firstChannel = image.firstChannel;
                    params.numChannels = image.channels;
                    params.texture = image.referenceTextureShared;
                    params.textureMipLevel = 0;
                    params.srcRgbColorSpace = srcRgbColorSpace;
                    params.srcAlphaColorSpace = srcAlphaColorSpace;
                    params.dstRgbColorSpace = dstRgbColorSpace;
                    params.dstAlphaColorSpace = dstAlphaColorSpace;

                    ntcStatus = m_textureSet->WriteChannelsFromTexture(params);
                    CHECK_NTC_RESULT(WriteChannelsFromTexture);
                }
                else
                {
                    ntc::WriteChannelsParameters params;
                    params.mipLevel = 0;
                    params.firstChannel = image.firstChannel;
                    params.numChannels = image.channels;
                    params.pData = image.data.get();
                    params.addressSpace = ntc::AddressSpace::Host;
                    params.width = image.width;
                    params.height = image.height;
                    params.pixelStride = pixelStride;
                    params.rowPitch = size_t(image.width) * pixelStride;
                    params.channelFormat = image.format;
                    params.srcColorSpaces = srcColorSpaces;
                    params.dstColorSpaces = dstColorSpaces;
                    
                    ntcStatus = m_textureSet->WriteChannels(params);
                    CHECK_NTC_RESULT(WriteChannels);
                }

                image.textureSetDataValid = true;
                needToGenerateMips = true;
            }

            // Refresh all texture metadata since we've just cleared it above
            ntc::ITextureMetadata* textureMetadata = m_textureSet->AddTexture();
            assert(textureMetadata);
            textureMetadata->SetName(image.name.c_str());
            textureMetadata->SetChannels(image.firstChannel, image.channels);
            textureMetadata->SetChannelFormat(image.format);
            textureMetadata->SetRgbColorSpace(srcRgbColorSpace);
            textureMetadata->SetAlphaColorSpace(srcAlphaColorSpace);
            CHECK_CANCEL(false);

            ++index;
        }

        // (Re-)generate mips if we've just uploaded some textures
        if (needToGenerateMips)
        {
            ntcStatus = m_textureSet->GenerateMips();
            CHECK_NTC_RESULT(GenerateMips);
            CHECK_CANCEL(false);
        }

        return true;
    }

    void RestoreReferenceTextureView(bool rightTexture)
    {
        if (rightTexture)
        {
            m_useRightDecompressedImage = false;
            m_rightImageName = "Reference";
        }
        else
        {
            m_useLeftDecompressedImage = false;
            m_leftImageName = "Reference";
        }
    }

    void SetRestoredRunName(CompressionResult const& result, bool useRightTextures)
    {
        char textureName[32];
        if (result.sourceFileName.empty())
            snprintf(textureName, sizeof textureName, "Run #%d", result.ordinal);
        else
            snprintf(textureName, sizeof textureName, "File #%d", result.ordinal);
        if (useRightTextures)
            m_rightImageName = textureName;
        else
            m_leftImageName = textureName;
    }

    bool RestoreCompressedTextureSet(const CompressionResult& result, bool useRightTextures)
    {
        ntc::MemoryStreamWrapper inputStream(m_ntcContext);
        ntc::Status ntcStatus = m_ntcContext->OpenReadOnlyMemory(result.compressedData->data(),
            result.compressedData->size(), inputStream.ptr());
        CHECK_NTC_RESULT(OpenReadOnlyMemory);

        auto reportError = [this, &result, &ntcStatus]()
        {
            log::error("Failed to load compressed texture data from run #%d, code = %s: %s",
                result.ordinal, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        };

        if (m_useGapiDecompression)
        {
            ntcStatus = DecompressWithGapi(inputStream, result.compressedData->size(), useRightTextures);

            if (ntcStatus != ntc::Status::Ok)
            {
                reportError();
                return false;
            }

            SetRestoredRunName(result, useRightTextures);

            return true;
        }
        
        if (m_textureSet)
        {
            ntcStatus = m_textureSet->LoadFromStream(inputStream);
            if (ntcStatus == ntc::Status::FileIncompatible)
            {
                m_ntcContext->DestroyTextureSet(m_textureSet);
                m_textureSet = nullptr;
            }
            else if (ntcStatus != ntc::Status::Ok)
            {
                // Reset the network and assume it's empty
                m_textureSet->AbortCompression();

                reportError();
                return false;
            }
        }

        if (!m_textureSet)
        {
            // Reset the stream to the beginning in case we tried and failed to load it above
            inputStream->Seek(0);
            
            ntcStatus = m_ntcContext->CreateCompressedTextureSetFromStream(inputStream,
                GetTextureSetFeatures(false), &m_textureSet);
            if (ntcStatus != ntc::Status::Ok)
            {
                reportError();
                return false;
            }

            m_textureSetDesc = m_textureSet->GetDesc();

            // Make sure to re-upload all images' reference data before the next compression run
            for (auto& image : m_images)
            {
                image.textureSetDataValid = false;
            }
        }

        inputStream.Close();
        
        // Make sure to restore with the same exp.knob that was used for compression
        m_textureSet->SetExperimentalKnob(result.experimentalKnob);

        if (!DecompressIntoTextures(false, useRightTextures, true, steady_clock::now()))
            return false;

        SetRestoredRunName(result, useRightTextures);

        return true;
    }

    bool CompressionThreadProc()
    {
        ntc::Status ntcStatus;

        bool uploadAllTextures = false;
        if (!m_textureSet)
        {
            ntcStatus = m_ntcContext->CreateTextureSet(m_textureSetDesc, GetTextureSetFeatures(true), &m_textureSet);
            CHECK_NTC_RESULT(CreateTextureSet);
            CHECK_CANCEL(false);

            uploadAllTextures = true;
        }
        
        if (!UploadReferenceImages(uploadAllTextures))
            return false;

        m_textureSet->SetMaskChannelIndex(m_alphaMaskChannelIndex, m_discardMaskedOutPixels);
        m_textureSet->SetExperimentalKnob(m_experimentalKnob);

        ntcStatus = m_textureSet->SetLatentShape(m_latentShape);
        CHECK_NTC_RESULT(SetLatentShape);
        
        // Apply the per-image loss function scales to the appropriate channels
        for (auto const& image : m_images)
        {
            for (int channel = 0; channel < image.channels; ++channel)
            {
                m_compressionSettings.lossFunctionScales[image.firstChannel + channel] = image.lossFunctionScales[channel];
            }
        }
        
        time_point beginTime = steady_clock::now();

        ntcStatus = m_textureSet->BeginCompression(m_compressionSettings);
        CHECK_NTC_RESULT(BeginCompression);
        CHECK_CANCEL(true);

        ntc::CompressionStats stats;
        char textureName[32];

        do
        {
            ntcStatus = m_textureSet->RunCompressionSteps(&stats);
            CHECK_CANCEL(true);
            if (ntcStatus == ntc::Status::Incomplete || ntcStatus == ntc::Status::Ok)
            {
                if (m_showCompressionProgress && ntcStatus == ntc::Status::Incomplete)
                {
                    if (!DecompressIntoTextures(false, true, false, beginTime))
                    {
                        // If the user clicks Cancel while decompression is running, DecompressIntoTextures(...)
                        // doesn't call AbortCompression() - do that here to avoid leaving the texture set
                        // in an incorrect state, which prevents further compression attempts.
                        m_textureSet->AbortCompression();
                        return false;
                    }
                }

                snprintf(textureName, sizeof textureName, "[%d%%]", (stats.currentStep * 100) / m_compressionSettings.trainingSteps);
                
                std::lock_guard guard(m_mutex);
                m_compressionStats = stats;
                if (m_showCompressionProgress)
                {
                    m_rightImageName = textureName;
                }
                GetDeviceManager()->RenderNextFrameWhileUnfocused();
            }
        } while (ntcStatus == ntc::Status::Incomplete);
        CHECK_NTC_RESULT(RunCompressionSteps);

        ntcStatus = m_textureSet->FinalizeCompression();
        CHECK_NTC_RESULT(FinalizeCompression);
        CHECK_CANCEL(false);

        m_compressedTextureSetAvailable = true;
        
        bool success = DecompressIntoTextures(true, true, false, beginTime);
        if (success)
        {
            snprintf(textureName, sizeof textureName, "Run #%d", m_compressionResults[m_compressionResults.size() - 1].ordinal);

            std::lock_guard guard(m_mutex);
            m_rightImageName = textureName;
        }
        GetDeviceManager()->RenderNextFrameWhileUnfocused();

        return success;

    }
#undef CHECK_NTC_RESULT
#undef CHECK_CANCEL

    void BeginCompression()
    {
        assert(m_ntcContext);

        if (!m_cudaAvailable)
            return;
        
        if (m_textureSet && m_textureSet->GetDesc() != m_textureSetDesc)
        {
            m_ntcContext->DestroyTextureSet(m_textureSet);
            m_textureSet = nullptr;
            m_compressedTextureSetAvailable = false;
        }

        m_compressing = true;
        m_compressionStats = ntc::CompressionStats();

        m_alphaMaskChannelIndex = -1;
        if (m_useAlphaMaskChannel)
        {
            for (auto& semanticBinding : m_semanticBindings)
                if (semanticBinding.label == SemanticLabel::AlphaMask)
                {
                    m_alphaMaskChannelIndex = m_images[semanticBinding.imageIndex].firstChannel + semanticBinding.firstChannel;
                    break;
                }
        }

        m_threadPool.AddTask([this](){
            CompressionThreadProc();
            m_compressing = false;
            m_cancel = false;
        });
    }

    void SaveCompressedTextureSet(const char* fileName) const
    {
        ntc::Status ntcStatus = m_textureSet->ConfigureLosslessCompression(GetLosslessCompressionParams());
        if (ntcStatus != ntc::Status::Ok)
        {
            log::error("Call to ConfigureLosslessCompression failed, code = %s: %s",
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return;
        }

        ntcStatus = m_textureSet->SaveToFile(fileName);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::error("Failed to save texture set to file '%s', code = %s: %s", fileName,
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return;
        }
    }

    void SaveManifestToFile(const char* fileName) const
    {
        if (!m_manifest.has_value())
            return;

        Manifest manifest = m_manifest.value();
        
        // Patch the manifest with the current image properties
        int manifestIndex = 0;
        for (ManifestEntry& texture : manifest.textures)
        {
            // Find the image corresponding to this manifest entry
            for (MaterialImage const& image : m_images)
            {
                if (image.manifestIndex == manifestIndex)
                {
                    texture.isSRGB = image.isSRGB;
                    
                    float scale = image.lossFunctionScales[0];
                    bool scalesAreUniform = true;
                    for (int channel = 1; channel < image.channels; ++channel)
                    {
                        if (image.lossFunctionScales[channel] != scale)
                        {
                            scalesAreUniform = false;
                            break;
                        }
                    }
                    if (!scalesAreUniform || scale != 1.f)
                        texture.lossFunctionScales = image.lossFunctionScales;
                        
                    break;
                }
            }

            // Replace the semantics with the current ones, potentially modified by the user
            texture.semantics.clear();
            for (const auto& binding : m_semanticBindings)
            {
                if (m_images[binding.imageIndex].manifestIndex == manifestIndex)
                {
                    ImageSemanticBinding imageBinding;
                    imageBinding.label = binding.label;
                    imageBinding.firstChannel = binding.firstChannel;
                    texture.semantics.push_back(imageBinding);
                }
            }

            ++manifestIndex;
        }

        std::string errorMessage;
        if (!WriteManifestToFile(fileName, manifest, errorMessage))
        {
            log::error("%s", errorMessage.c_str());
            return;
        }
    }

    void Animate(float elapsedTimeSeconds) override
    {
        ImGui_Renderer::Animate(elapsedTimeSeconds);
        m_modelView->Animate(elapsedTimeSeconds);
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (!m_flatImageView->Init(framebuffer->getFramebufferInfo()))
            return;

        if (!m_modelView->Init(framebuffer->getFramebufferInfo()))
            return;

        if (m_loading)
        {
            if (m_texturesLoaded + m_errors == m_texturesToLoad)
            {
                m_threadPool.WaitForTasks();
                m_loading = false;

                UploadTextures();
                NewTexturesLoaded();
            }
        }

        const auto& fbinfo = framebuffer->getFramebufferInfo();
        m_flatImageView->SetTextureSize(m_textureSetDesc.width, m_textureSetDesc.height, m_textureSetDesc.mips);
        m_flatImageView->SetViewport(dm::float2(0.f), dm::float2(float(fbinfo.width), float(fbinfo.height)));
        m_modelView->SetViewport(dm::float2(0.f), dm::float2(float(fbinfo.width), float(fbinfo.height)));
        
        m_commandList->open();
        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));
        
        if (!m_loading && !m_images.empty())
        {
            for (auto& image : m_images)
            {
                if (!image.referenceMipsValid)
                {
                    GenerateReferenceMips(m_commandList, image.referenceTexture, image.isSRGB);
                    image.referenceMipsValid = true;
                }
            }

            if (m_selectedImage < 0)
            {
                int imageIndex = 0;
                for (auto& image : m_images)
                {
                    m_modelView->SetTexture(m_useLeftDecompressedImage ? image.decompressedTextureLeft : image.referenceTexture ? image.referenceTexture : image.decompressedTextureRight, image.isSRGB, imageIndex, false);
                    m_modelView->SetTexture(m_useRightDecompressedImage ? image.decompressedTextureRight : image.referenceTexture ? image.referenceTexture : image.decompressedTextureLeft, image.isSRGB, imageIndex, true);

                    ++imageIndex;
                }

                m_modelView->SetNumTextureMips(m_textureSetDesc.mips);

                m_modelView->SetSemanticBindings(m_semanticBindings.data(), m_semanticBindings.size());

                m_modelView->SetDecompressedImagesAvailable(m_useRightDecompressedImage);

                m_commandList->beginMarker("ModelView");
                m_modelView->Render(m_commandList, framebuffer);
                m_commandList->endMarker();
            }
            else
            {
                const MaterialImage& selectedImage = m_images[m_selectedImage];
                m_flatImageView->SetTextures(
                    m_useLeftDecompressedImage ? selectedImage.decompressedTextureLeft : selectedImage.referenceTexture ? selectedImage.referenceTexture : selectedImage.decompressedTextureRight,
                    m_useRightDecompressedImage ? selectedImage.decompressedTextureRight : selectedImage.referenceTexture ? selectedImage.referenceTexture : selectedImage.decompressedTextureLeft,
                    selectedImage.channels,
                    selectedImage.isSRGB);

                m_commandList->beginMarker("FlatImageView");
                m_flatImageView->Render(m_commandList, framebuffer);
                m_commandList->endMarker();
            }
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        ImGui_Renderer::Render(framebuffer);

        if (!m_loading && m_selectedImage >= 0)
            m_flatImageView->ReadPixel();
    }

    void BuildSettingsDialog(float fontSize, bool& openImageSettings)
    {
        ImGui::SetNextWindowPos(ImVec2(fontSize * 0.6f, fontSize * 2.f), 0);
        ImGui::SetNextWindowSizeConstraints(ImVec2(), ImVec2(FLT_MAX, ImGui::GetIO().DisplaySize.y - fontSize * 3.f));
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::PushItemWidth(fontSize * 9.f);
            
            ImGui::TextUnformatted("View:");
            if (ImGui::RadioButton("3D Model", m_selectedImage < 0))
                m_selectedImage = -1;

            int index = 0;
            for (auto& image : m_images)
            {
                ImGui::PushID(index);

                if (ImGui::RadioButton(image.name.c_str(), index == m_selectedImage))
                    m_selectedImage = index;

                ImGui::SameLine(fontSize * 12.f);
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(128, 128, 128, 255));
                if (image.bcFormat != ntc::BlockCompressedFormat::None)
                {
                    ImGui::TextUnformatted(ntc::BlockCompressedFormatToString(image.bcFormat));
                }
                else
                {
                    char const* shortFormat = "";
                    switch(image.format)
                    {
                        case ntc::ChannelFormat::UNORM8: shortFormat = "un8"; break;
                        case ntc::ChannelFormat::UNORM16: shortFormat = "un16"; break;
                        case ntc::ChannelFormat::UINT32: shortFormat = "u32"; break;
                        case ntc::ChannelFormat::FLOAT16: shortFormat = "f16"; break;
                        case ntc::ChannelFormat::FLOAT32: shortFormat = "f32"; break;
                    }
                    ImGui::Text("%sx%d", shortFormat, image.channels);
                }
                ImGui::PopStyleColor();

                ImGui::SameLine(fontSize * 16.f);
                if (ImGui::Button(" ... "))
                {
                    openImageSettings = true;
                    m_editingImage = index;
                }

                ImGui::PopID();
                ++index;
            }
            
            ImGui::Separator();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Semantics:");

            ImGui::SameLine();
            if (ImGui::Button("Add"))
                m_semanticBindings.push_back(SemanticBinding());

            ImGui::TooltipMarker("Define the interpretation of texture channels.\n"
                "This information is used for the 3D view, and the Alpha Mask channel can be used for compression.");

            const auto getImageChannelLabel = [this](int imageIndex, int firstChannel, int numChannels)
            {
                static const std::string channels = "RGBA";
                return m_images[imageIndex].name + "." + channels.substr(firstChannel, numChannels);
            };

            int bindingIndex = 0;
            int deleteBindingIndex = -1;
            for (auto& semanticBinding : m_semanticBindings)
            {
                ImGui::PushID(bindingIndex);

                ImGui::PushItemWidth(fontSize * 7.5f);
                if (ImGui::BeginCombo("##SemanticLabel", SemanticLabelToString(semanticBinding.label)))
                {
                    for (int label = 0; label < int(SemanticLabel::Count); ++label)
                    {
                        bool selected = int(semanticBinding.label) == label;
                        ImGui::Selectable(SemanticLabelToString(SemanticLabel(label)), &selected);

                        if (selected)
                        {
                            ImGui::SetItemDefaultFocus();
                            semanticBinding.label = SemanticLabel(label);
                        }
                    }
                    ImGui::EndCombo();
                }
                
                ImGui::SameLine();

                const int numChannels = GetSemanticChannelCount(semanticBinding.label);
                if (ImGui::BeginCombo("##SemanticImage", getImageChannelLabel(semanticBinding.imageIndex, semanticBinding.firstChannel, numChannels).c_str()))
                {
                    int imageIndex = 0;
                    for (auto& image : m_images)
                    {
                        for (int firstChannel = 0; firstChannel <= image.channels - numChannels; firstChannel += numChannels)
                        {
                            bool selected = semanticBinding.imageIndex == imageIndex && semanticBinding.firstChannel == firstChannel;
                            ImGui::Selectable(getImageChannelLabel(imageIndex, firstChannel, numChannels).c_str(), &selected);

                            if (selected)
                            {
                                ImGui::SetItemDefaultFocus();
                                semanticBinding.imageIndex = imageIndex;
                                semanticBinding.firstChannel = firstChannel;
                            }
                        }
                        ++imageIndex;
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopItemWidth();

                ImGui::SameLine();
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.f);
                if (ImGui::Button("X"))
                    deleteBindingIndex = bindingIndex;

                ImGui::PopStyleVar();
                ImGui::PopID();
                ++bindingIndex;
            }

            if (deleteBindingIndex >= 0)
                m_semanticBindings.erase(m_semanticBindings.begin() + deleteBindingIndex);

            bool const referenceAvailable = !m_images.empty() && !!m_images[0].referenceTexture;
            if (referenceAvailable)
            {
                ImGui::Separator();
                ImGui::TextUnformatted("Encoding Parameters:");

                float currentBpp = ntc::GetLatentShapeBitsPerPixel(m_latentShape);
                if (ImGui::SliderFloat("Bits per Pixel", &currentBpp, 1.0f, 20.f, "%.3f", ImGuiSliderFlags_Logarithmic))
                {
                    ntc::PickLatentShape(currentBpp, currentBpp, m_latentShape);
                }
                ImGui::TooltipMarker("The bitrate to aim for in a single MIP level.\n"
                    "When compressing the entire MIP chain, overall bitrate will be lower.");

                if (m_developerUI)
                {
                    ImGui::SliderInt("Grid Size Scale", &m_latentShape.gridSizeScale, 1, 6);
                    
                    if (ImGui::SliderInt("Number of Features", &m_latentShape.numFeatures, NTC_FEATURES_PER_LAYER, NTC_MLP_FEATURES))
                    {
                        // Round to a multiple of NTC_FEATURES_PER_LAYER
                        m_latentShape.numFeatures = ((m_latentShape.numFeatures + 1) / NTC_FEATURES_PER_LAYER) * NTC_FEATURES_PER_LAYER;
                    }
                }
                
                bool compressMipChain = m_textureSetDesc.mips > 1;
                if (ImGui::Checkbox("Compress MIP Chain", &compressMipChain))
                {
                    SetCompressMipChain(compressMipChain);
                }
                ImGui::TooltipMarker("Controls whether all MIP levels should be encoded within the NTC file.\n"
                    "This is useful for partial decompression, such as streaming lower quality mips first, "
                    "or for decompress-on-sample.");
                
                size_t estimatedFileSize = 0;
                if (ntc::EstimateCompressedTextureSetSize(m_textureSetDesc, m_latentShape, estimatedFileSize) == ntc::Status::Ok)
                {
                    size_t uncompressedTextureSize = 0;
                    bool useBcFormats = false;
                    for (const auto& image : m_images)
                    {
                        if (compressMipChain)
                            uncompressedTextureSize += image.bcSizeWithMips;
                        else
                            uncompressedTextureSize += image.bcSize;

                        if (image.bcFormat != ntc::BlockCompressedFormat::None)
                            useBcFormats = true;
                    }
                    
                    const double fileSizeMegabytes = double(estimatedFileSize) / 1'048'576.0;
                    const double compressedBitsPerPixel = double(estimatedFileSize) * 8.0 / double(m_totalPixels);
                    const double compressionRatio = double(uncompressedTextureSize) / double(estimatedFileSize);
                    ImGui::PushFont(m_LargerFont->GetScaledFont());
                    ImGui::Text("File Size: %.2f MB", fileSizeMegabytes);
                    ImGui::Text("Effective bitrate: %.2f bpp", compressedBitsPerPixel);
                    if (useBcFormats)
                        ImGui::Text("Compression ratio: %.2fx over BCn", compressionRatio);
                    else
                        ImGui::Text("Compression ratio: %.2fx over raw", compressionRatio);
                    ImGui::PopFont();
                        
                    ImGui::TooltipMarker("The ratio is calculated using the BC formats specified in the manifest.\n"
                        "If images were loaded without manifest or no BC formats were specified, the original bitrates are used instead.");
                }
                
                ImGui::Separator();
                ImGui::TextUnformatted("Compression Settings:");

                ImGui::DragInt("Training Steps", &m_compressionSettings.trainingSteps, 100.f, 1, 1'000'000);
                ImGui::TooltipMarker("The number of steps to train the neural network and latents.\n"
                    "Higher step count yields higher image quality.");

                ImGui::DragInt("kPixels Per Batch", &m_compressionSettings.kPixelsPerBatch, 1.f, 1, NTC_MAX_KPIXELS_PER_BATCH);
                ImGui::TooltipMarker("The number of kilopixels to use in each training step.\n"
                    "Higher pixel count yields higher image quality, up to a certain point.");

                if (m_developerUI)
                {
                    ImGui::DragInt("Reporting Steps", &m_compressionSettings.stepsPerIteration, 10.f, 1, 10'000);
                    ImGui::DragFloat("Network Learning Rate", &m_compressionSettings.networkLearningRate, 0.0001f, 0.0001f, 0.2f, "%.4f");
                    ImGui::DragFloat("Grid Learning Rate", &m_compressionSettings.gridLearningRate, 0.0001f, 0.0001f, 0.2f, "%.4f");
                }

                ImGui::DragInt("Random Seed", (int*)&m_compressionSettings.randomSeed, 1.f, 0, 65535);
                ImGui::TooltipMarker("Random number generator seed for training during compression.");

                if (m_compressionSettings.randomSeed == 0)
                {
                    ImGui::BeginDisabled();
                    m_compressionSettings.stableTraining = false;
                }
                ImGui::Checkbox("Stable Training", &m_compressionSettings.stableTraining);
                if (m_compressionSettings.randomSeed == 0)
                    ImGui::EndDisabled();
                ImGui::TooltipMarker("Use a more expensive but more numerically stable training algorithm \n"
                    "for reproducible results. Requires nonzero Random Seed.");
                
                ImGui::Checkbox("Use Alpha Mask Channel", &m_useAlphaMaskChannel);
                ImGui::TooltipMarker("Enable special processing for the alpha mask channel.\n"
                    "The 0.0 and 1.0 values in the mask channel will be preserved with higher accuracy.\n"
                    "Requires the alpha mask channel to be specified in the Semantics list above.");
                    
                if (!m_useAlphaMaskChannel)
                    ImGui::BeginDisabled();
                ImGui::Checkbox("Discard Masked Out Pixels", &m_discardMaskedOutPixels);
                if (!m_useAlphaMaskChannel)
                    ImGui::EndDisabled();
                ImGui::TooltipMarker("Ignore the data in all other channels for pixels where alpha mask is 0.\n"
                    "Requires the Use Alpha Mask Channel option to be active.");
                
                ImGui::Checkbox("Use GDeflate", &m_enableGDeflate);
                ImGui::TooltipMarker("Enables the use of GDeflate compression for latent data and BC7 mode buffers.\n"
                    "Note that this setting makes final files somewhat smaller, but it does not affect the estimated size.");

                if (m_developerUI)
                {
                    ImGui::Checkbox("Enable Int8 restore", &m_useInt8Decompression);
                    ImGui::Checkbox("Restore with GAPI Decompression", &m_useGapiDecompression);
                    ImGui::Checkbox("Decompress sub-rect (for testing)", &m_useGapiDecompressionRect);
                    if (m_useGapiDecompressionRect)
                    {
                        ImGui::DragInt4("Decompression rect", &m_gapiDecompressionRect.left, 1.f, 0, std::max(m_textureSetDesc.width, m_textureSetDesc.height));
                    }
                    ImGui::DragFloat("Experimental Knob", &m_experimentalKnob, 0.01f);
                }

                ImGui::Separator();
                if (!m_compressing)
                {
                    if (ImGui::Button("Compress!"))
                        BeginCompression();
                }
                else
                {
                    char buf[32];
                    float progress = float(m_compressionStats.currentStep) / float(m_compressionSettings.trainingSteps);
                    snprintf(buf, sizeof buf, "%d / %d", m_compressionStats.currentStep, m_compressionSettings.trainingSteps);

                    ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), buf);
                    ImGui::SameLine();
                    if (ImGui::Button("Cancel"))
                        m_cancel = true;

                    ImGui::Text("In-progress PSNR: %.2f dB", ntc::LossToPSNR(m_compressionStats.loss));
                    ImGui::Text("Compression performance: %.2f ms/step", m_compressionStats.millisecondsPerStep);
                }

                if (!m_sharedTexturesAvailable)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
                    ImGui::TextUnformatted("CUDA texture sharing unavailable!");
                    ImGui::PopStyleColor(1);
                }
            }
            
            if (!m_compressionResults.empty())
            {
                ImGui::Separator();
                ImGui::TextUnformatted("Compression Results:");
                ImGui::TooltipMarker("Drag any compression result onto either of the channel slots " 
                    "in the bottom dialog to restore it into that channel.");

                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 10, 2 });
                ImGui::BeginTable("Results", 6);
                ImGui::TableSetupColumn("#");
                ImGui::TableSetupColumn("bpp");
                ImGui::TableSetupColumn("Steps");
                ImGui::TableSetupColumn("Batch");
                ImGui::TableSetupColumn("Time");
                ImGui::TableSetupColumn("PSNR");
                ImGui::TableHeadersRow();
                
                for (auto result = m_compressionResults.rbegin(); result != m_compressionResults.rend(); ++result)
                {
                    const int kiloSteps = result->compressionSettings.trainingSteps / 1000;
                    const int minutes = int(floorf(result->timeSeconds / 60.f));
                    const float seconds = result->timeSeconds - float(minutes * 60);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    char buf[32];
                    snprintf(buf, sizeof buf, "%d", result->ordinal);
                    if (ImGui::Selectable(buf, false, ImGuiSelectableFlags_SpanAllColumns))
                    {
                        m_selectedCompressionResult = *result;
                        m_selectedCompressionResultValid = true;
                    }
                    if (!m_compressing && ImGui::BeginDragDropSource(ImGuiDragDropFlags_None))
                    {
                        ImGui::SetDragDropPayload("CompressionRun", &result->ordinal, sizeof(int));
                        ImGui::Text("Run #%d", result->ordinal);
                        ImGui::EndDragDropSource();
                    }
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2f", result->bitsPerPixel);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%dk", kiloSteps);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%dk", result->compressionSettings.kPixelsPerBatch);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%d:%04.1f", minutes, seconds);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.2f dB", result->overallPSNR);
                }

                ImGui::EndTable();
                ImGui::PopStyleVar();

                if (ImGui::Button("Clear Results"))
                {
                    m_compressionResults.clear();
                    RestoreReferenceTextureView(false);
                }
                ImGui::SameLine();
                if (ImGui::Button("Restore Reference"))
                {
                    RestoreReferenceTextureView(false);
                }
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None))
                {
                    const int ordinal = 0;
                    ImGui::SetDragDropPayload("CompressionRun", &ordinal, sizeof(int));
                    ImGui::TextUnformatted("Reference");
                    ImGui::EndDragDropSource();
                }
                ImGui::TooltipMarker("Drag the Restore Reference button onto either of the channel slots " 
                    "in the bottom dialog to put the reference images into that channel.");
            }
            
            ImGui::PopItemWidth();
        }
        // End of window
        ImGui::End();
    }

    void buildUI() override
    {
        if (m_loading || m_images.empty())
        {
            ImGui::PushFont(m_PrimaryFont->GetScaledFont());
            BeginFullScreenWindow();
            if (m_loading)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "Loading images: %d/%d, %d errors", m_texturesLoaded, m_texturesToLoad, m_errors);
                DrawScreenCenteredText(buf);
            }
            else
            {
                DrawScreenCenteredText("No images loaded.");
            }
            EndFullScreenWindow();
            ImGui::PopFont();

            if (m_loading)
                return;
        }

        // Various UI-related things are written from the compression thread
        std::lock_guard lock(m_mutex);

        ImGui::PushFont(m_PrimaryFont->GetScaledFont());
        float const fontSize = ImGui::GetFontSize();

        bool openViewerHelp = false;

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Load Images from Folder..."))
                {
                    static std::string defaultPath = app::GetDirectoryWithExecutable().string();
                    std::string path;
                    if (app::FolderDialog("Select a folder with images", defaultPath.c_str(), path))
                    {
                        defaultPath = path;
                        m_compareMode = false;
                        BeginLoadingImagesFromDirectory(path.c_str());
                    }
                }
                if (ImGui::MenuItem("Load Images with Manifest..."))
                {
                    std::string fileName;
                    if (app::FileDialog(true, "JSON manifest files\0*.json\0All files\0*.*\0", fileName))
                    {
                        m_compareMode = false;
                        BeginLoadingImagesFromManifest(fileName.c_str());
                    }
                }
                if (ImGui::MenuItem("Load Compressed File..."))
                {
                    std::string fileName;
                    if (app::FileDialog(true, "NTC files\0*.ntc\0All files\0*.*\0", fileName))
                    {
                        if (m_compareMode)
                        {
                            ClearImages();
                            m_compareMode = false;
                        }
                        
                        bool const imagesWereEmpty = m_images.empty();
                        CompressionResult* result = LoadCompressedTextureSet(fileName.c_str(), true);
                        if (result)
                        {
                            RestoreCompressedTextureSet(*result, /* useRightTextures = */ !imagesWereEmpty);
                            if (imagesWereEmpty)
                                NewTexturesLoaded();
                        }
                    }
                }
                if (ImGui::MenuItem("Compare Two Images..."))
                {
                    char const* filter = "Image files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.hdr;*.exr\0All files\0*.*\0";
                    std::string fileNameA;
                    if (app::FileDialog(true, filter, fileNameA))
                    {
                        std::string fileNameB;
                        if (app::FileDialog(true, filter, fileNameB))
                        {
                            m_compareMode = true;
                            BeginLoadingImagesFromFileList({ fileNameA.c_str(), fileNameB.c_str() });
                        }
                    }
                }
                if (ImGui::MenuItem("Save Compressed File...", nullptr, nullptr, m_compressedTextureSetAvailable))
                {
                    std::string fileName;
                    if (app::FileDialog(false, "NTC files \0*.ntc\0All files\0*.*\0", fileName))
                    {
                        SaveCompressedTextureSet(fileName.c_str());
                    }
                }
                if (ImGui::MenuItem("Save Manifest...", nullptr, nullptr, m_manifest.has_value()))
                {
                    std::string fileName;
                    if (app::FileDialog(false, "JSON manifest files\0*.json\0All files\0*.*\0", fileName))
                    {
                        SaveManifestToFile(fileName.c_str());
                    }
                }
                if (ImGui::MenuItem("Unload Images", nullptr, nullptr, !m_images.empty()))
                {
                    ClearImages();
                    m_compareMode = false;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Options"))
            {
                ImGui::MenuItem("Show Compression Progress", nullptr, &m_showCompressionProgress);
                ImGui::MenuItem("Developer UI", nullptr, &m_developerUI);
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Help"))
            {
                if (ImGui::MenuItem("Using the Viewer"))
                {
                    openViewerHelp = true;
                }
                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        if (openViewerHelp)
            ImGui::OpenPopup("ViewerHelp");

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopup("ViewerHelp", ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::BeginTable("Help", 2);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("2D Viewer:");

            ImGui::Indent();
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Pan the image");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("LMB or touchpad scroll");

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Zoom");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Mouse wheel or touchpad zoom gesture");
            
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Move the A/B slider");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("RMB or Shift+LMB");
            ImGui::Unindent();

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("3D Viewer:");

            ImGui::Indent();
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Rotate the camera");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("LMB");

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Zoom");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Mouse wheel or touchpad zoom gesture");
            
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Move the A/B slider");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("RMB or Shift+LMB");

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Move the light");
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Ctrl+LMB");
            ImGui::Unindent();

            ImGui::EndTable();
            ImGui::Dummy(ImVec2(0.f, fontSize));

            ImGuiStyle& style = ImGui::GetStyle();
            float avail = ImGui::GetContentRegionAvail().x;
            float buttonWidth = fontSize * 8.f;
            float offset = (avail - buttonWidth) * 0.5f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);

            if (ImGui::Button("OK", ImVec2(buttonWidth, 0)))
                ImGui::CloseCurrentPopup();

            ImGui::End();
        }
        
        if (m_images.empty())
        {
            ImGui::PopFont();
            return;
        }

        bool openImageSettings = false;

        if (!m_compareMode)
        {
            BuildSettingsDialog(fontSize, openImageSettings);
        }

        if (openImageSettings)
            ImGui::OpenPopup("ImageSettings");

        if (ImGui::BeginPopup("ImageSettings"))
        {
            if (m_editingImage >= 0 && m_editingImage < int(m_images.size()))
            {
                MaterialImage& image = m_images[m_editingImage];

                ImGui::Text("Image settings for '%s'", image.name.c_str());
                ImGui::Separator();

                if (image.format == ntc::ChannelFormat::UNORM8 || image.format == ntc::ChannelFormat::UNORM16)
                {
                    if (ImGui::Checkbox("sRGB", &image.isSRGB))
                    {
                        image.referenceMipsValid = false;
                        image.textureSetDataValid = false;
                    }
                }

                float const minScale = 0.1f;
                float const maxScale = 10.f;
                ImGui::PushItemWidth(fontSize * 5.f * float(image.channels));
                ImGui::SliderScalarN("Loss scale(s)", ImGuiDataType_Float, image.lossFunctionScales.data(),
                    image.channels, &minScale, &maxScale, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::PopItemWidth();
                
                ImGui::TooltipMarker("Loss function scale for the channels of this texture.\n"
                    "Higher values make the channels compress more accurately than other channels in the texture set.");

                if (ImGui::Button("Reset loss scales"))
                {
                    for (float& scale : image.lossFunctionScales)
                        scale = 1.f;
                }
            }

            ImGui::EndPopup();
        }

        m_modelView->SetImageName(false, m_leftImageName);
        m_modelView->SetImageName(true, m_rightImageName);
        m_flatImageView->SetImageName(false, m_leftImageName);
        m_flatImageView->SetImageName(true, m_rightImageName);

        static int restoreRunOrdinal = 0;
        static bool restoreRightTexture = false;
        static bool requestingRestore = false;

        // When in capture mode, keep running the restore operation until application exits.
        if (!requestingRestore || !g_options.captureMode)
        {
            if (IsModelViewActive())
            {
                m_modelView->BuildControlDialog();
                requestingRestore = m_modelView->IsRequestingRestore(restoreRunOrdinal, restoreRightTexture);
            }
            else
            {
                m_flatImageView->BuildControlDialog();
                requestingRestore = m_flatImageView->IsRequestingRestore(restoreRunOrdinal, restoreRightTexture);
            }
        }

        if (requestingRestore)
        {
            if (restoreRunOrdinal == 0)
            {
                // Ordinal 0 means reference, see the "Restore Reference" button above.
                RestoreReferenceTextureView(restoreRightTexture);
            }
            else if (!m_compressing)
            {
                for (auto& result : m_compressionResults)
                {
                    if (result.ordinal == restoreRunOrdinal)
                    {
                        RestoreCompressedTextureSet(result, restoreRightTexture);
                        break;
                    }
                }
            }
        }
        
        if (m_selectedCompressionResultValid)
        {
            int width, height;
            GetDeviceManager()->GetWindowDimensions(width, height);
            ImGui::SetNextWindowPos({ float(width / 2), float(height / 2) }, ImGuiCond_Appearing, { 0.5f, 0.5f });
            ImGui::Begin("Result Details", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);

            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, { 10, 2 });
            ImGui::BeginTable("Result Values", 2);
            ImGui::TableSetupColumn("Parameter");
            ImGui::TableSetupColumn("Value");
            ImGui::TableHeadersRow();

            auto setupRow = [](const char* name)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Selectable(name, false, ImGuiSelectableFlags_SpanAllColumns);
                ImGui::TableNextColumn();
            };
            setupRow("Result Ordinal");             ImGui::Text("#%d", m_selectedCompressionResult.ordinal);
            ImGui::Separator();

            setupRow("Bits per pixel");             ImGui::Text("%.2f", m_selectedCompressionResult.bitsPerPixel);
            setupRow("Stored texture size");        ImGui::Text("%.2f MB", float(m_selectedCompressionResult.compressedData->size()) / 1'048'576.f);
            setupRow("Compress MIP chain");         ImGui::Text("%s", m_selectedCompressionResult.compressMipChain ? "YES" : "NO");
            setupRow("GDeflate");                   ImGui::Text("%s", m_selectedCompressionResult.useGDeflate.has_value() 
                                                                        ? (m_selectedCompressionResult.useGDeflate.value() ? "YES" : "NO")
                                                                        : "N/A");
            setupRow("Random seed");                ImGui::Text("%d", m_selectedCompressionResult.compressionSettings.randomSeed);
            setupRow("Stable training");            ImGui::Text("%s", m_selectedCompressionResult.compressionSettings.stableTraining ? "YES" : "NO");
            setupRow("Grid size scale");            ImGui::Text("%d", m_selectedCompressionResult.latentShape.gridSizeScale);
            setupRow("Number of features");         ImGui::Text("%d", m_selectedCompressionResult.latentShape.numFeatures);
            setupRow("Compression steps");          ImGui::Text("%d", m_selectedCompressionResult.compressionSettings.trainingSteps);
            setupRow("kPixels per batch");          ImGui::Text("%d", m_selectedCompressionResult.compressionSettings.kPixelsPerBatch);
            setupRow("Network learning rate");      ImGui::Text("%.4f", m_selectedCompressionResult.compressionSettings.networkLearningRate);
            setupRow("Grid learning rate");         ImGui::Text("%.4f", m_selectedCompressionResult.compressionSettings.gridLearningRate);
            setupRow("Experimental knob");          ImGui::Text("%.3f", m_selectedCompressionResult.experimentalKnob);

            ImGui::Separator();
            setupRow("Overall PSNR");               ImGui::Text("%.2f dB", m_selectedCompressionResult.overallPSNR);

            int const mips = m_selectedCompressionResult.compressMipChain ? m_numTextureSetMips : 1;
            for (int mip = 0; mip < mips; ++mip)
            {
                char buf[32];
                snprintf(buf, sizeof buf, "Mip %d PSNR", mip);
                setupRow(buf);
                ImGui::Text("%.2f dB", m_selectedCompressionResult.perMipPSNR[mip]);
            }

            ImGui::Separator();
            for (size_t i = 0; i < m_images.size(); ++i)
            {
                MaterialImage& image = m_images[i];
                setupRow(image.name.c_str());
                
                std::stringstream ss;
                float textureMSE = 0.f;
                for (int ch = 0; ch < image.channels; ++ch)
                {
                    textureMSE += m_selectedCompressionResult.perChannelMSE[ch + image.firstChannel];
                }
                textureMSE /= float(image.channels);
                ss << std::fixed << std::setprecision(2);
                ss << ntc::LossToPSNR(textureMSE) << " dB";
                if (image.channels > 1)
                {
                    ss << " [";
                    for (int ch = 0; ch < image.channels; ++ch)
                    {
                        if (ch > 0)
                            ss << ", ";
                        float channelPSNR = ntc::LossToPSNR(m_selectedCompressionResult.perChannelMSE[ch + image.firstChannel]);
                        ss << channelPSNR;
                    }
                    ss << "]";
                }
                ImGui::TextUnformatted(ss.str().c_str());
            }

            ImGui::EndTable();
            ImGui::PopStyleVar();
            ImGui::Separator();

            ImGuiStyle& style = ImGui::GetStyle();
            float avail = ImGui::GetContentRegionAvail().x;
            float buttonWidth = fontSize * 5.f;
            float offset = (avail - buttonWidth * 3.f - style.ItemSpacing.x * 2.f) * 0.5f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);

            ImGui::BeginDisabled(m_compressing);
            if (ImGui::ButtonEx("Restore", { buttonWidth, 0.f }) && !m_compressing)
            {
                m_latentShape = m_selectedCompressionResult.latentShape;
                m_compressionSettings = m_selectedCompressionResult.compressionSettings;

                RestoreCompressedTextureSet(m_selectedCompressionResult, true);
            }
            ImGui::EndDisabled();

            ImGui::SameLine();
            if (ImGui::Button("Copy", { buttonWidth, 0.f }))
            {
                std::stringstream ss;
                ss << "Parameter\tName\n";
                ss << "Ordinal\t" << m_selectedCompressionResult.ordinal << "\n";
                ss << "Bits per pixel\t" << m_selectedCompressionResult.bitsPerPixel << "\n";
                ss << "Experimental knob\t" << m_selectedCompressionResult.experimentalKnob << "\n";
                ss << "Overall PSNR\t" << m_selectedCompressionResult.overallPSNR << "\n";
                for (int mip = 0; mip < mips; ++mip)
                {
                    ss << "Mip " << mip << " PSNR\t" << m_selectedCompressionResult.perMipPSNR[mip] << "\n";
                }
                glfwSetClipboardString(GetDeviceManager()->GetWindow(), ss.str().c_str());
            }

            ImGui::SameLine();
            if (ImGui::Button("Close", { buttonWidth, 0.f }))
                m_selectedCompressionResultValid = false;

            ImGui::End();
        }
        ImGui::PopFont();
    }
};

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    donut::log::SetErrorMessageCaption(g_ApplicationName);

    if (!ProcessCommandLine(__argc, (const char**)__argv))
        return 1;

#if NTC_WITH_DX12 && NTC_WITH_VULKAN
    const nvrhi::GraphicsAPI graphicsApi = g_options.useDX12
        ? nvrhi::GraphicsAPI::D3D12
        : nvrhi::GraphicsAPI::VULKAN;
#elif NTC_WITH_VULKAN
    const nvrhi::GraphicsAPI graphicsApi = nvrhi::GraphicsAPI::VULKAN;
#else
    const nvrhi::GraphicsAPI graphicsApi = nvrhi::GraphicsAPI::D3D12;
#endif
    app::DeviceManager* deviceManager = app::DeviceManager::Create(graphicsApi);

    cudaDeviceProp cudaDeviceProperties{};
    if (g_options.cudaDevice >= 0)
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaSuccess && count > 0)
        {
            cudaGetDeviceProperties(&cudaDeviceProperties, g_options.cudaDevice);
        }
    }

    app::DeviceCreationParameters deviceParams;
    deviceParams.infoLogSeverity = log::Severity::None;
    deviceParams.vsyncEnabled = true;
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;
    deviceParams.adapterIndex = g_options.adapterIndex;
    deviceParams.swapChainFormat = g_options.hdr ? nvrhi::Format::RGBA16_FLOAT : nvrhi::Format::SRGBA8_UNORM;
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.supportExplicitDisplayScaling = true;
    
    if (g_options.debug)
    {
        deviceParams.enableDebugRuntime = true;
        deviceParams.enableNvrhiValidationLayer = true;
    }

    SetNtcGraphicsDeviceParameters(deviceParams, graphicsApi, true, g_options.enableCoopVec, g_ApplicationName);

    if (!deviceManager->CreateInstance(deviceParams))
    {
        log::error("Cannot initialize a %s subsystem.", nvrhi::utils::GraphicsAPIToString(graphicsApi));
        return 1;
    }

    std::vector<donut::app::AdapterInfo> adapters;
    if (!deviceManager->EnumerateAdapters(adapters))
    {
        log::error("Cannot enumerate graphics adapters.");
        return 1;
    }

    // When there is a CUDA device and no graphics adapter is specified, try to find a graphics adapter
    // matching the selected CUDA device.
    if (cudaDeviceProperties.major > 0 && g_options.adapterIndex < 0)
    {
        for (int adapterIndex = 0; adapterIndex < int(adapters.size()); ++adapterIndex)
        {
            donut::app::AdapterInfo const& adapter = adapters[adapterIndex];

            static_assert(sizeof(donut::app::AdapterInfo::UUID) == sizeof(cudaDeviceProperties.uuid));
            static_assert(sizeof(donut::app::AdapterInfo::LUID) == sizeof(cudaDeviceProperties.luid));

            if (adapter.uuid.has_value() && !memcmp(adapter.uuid->data(), cudaDeviceProperties.uuid.bytes, sizeof(cudaDeviceProperties.uuid)) ||
                adapter.luid.has_value() && !memcmp(adapter.luid->data(), cudaDeviceProperties.luid, sizeof(cudaDeviceProperties.luid)) )
            {
                deviceParams.adapterIndex = adapterIndex;
                break;
            }
        }

        if (deviceParams.adapterIndex < 0)
        {
            log::warning("Warning: Couldn't find a matching %s adapter for the selected CUDA device %d (%s).\n",
                nvrhi::utils::GraphicsAPIToString(graphicsApi), g_options.cudaDevice, cudaDeviceProperties.name);
        }
    }

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_ApplicationName))
    {
        log::error("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    char windowTitle[256];
    snprintf(windowTitle, sizeof windowTitle, "%s (%s, %s)", g_ApplicationName,
        nvrhi::utils::GraphicsAPIToString(graphicsApi), deviceManager->GetRendererString());
    deviceManager->SetWindowTitle(windowTitle);
     
    {
        Application app(deviceManager);
        
        if (app.Init())
        {
            deviceManager->AddRenderPassToBack(&app);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&app);
        }
    }

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
