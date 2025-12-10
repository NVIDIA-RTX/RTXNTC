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

#include <argparse.h>
#include <cinttypes>
#include <cmath>
#include <cuda_runtime_api.h>
#include <donut/app/DeviceManager.h>
#include <filesystem>
#include <libntc/ntc.h>
#include <ntc-utils/DeviceUtils.h>
#include <ntc-utils/GraphicsDecompressionPass.h>
#include <ntc-utils/Manifest.h>
#include <ntc-utils/Misc.h>
#include <ntc-utils/Semantics.h>
#include <nvrhi/utils.h>
#include <stb_image.h>
#include <tinyexr.h>
#include "GraphicsPasses.h"
#include "Utils.h"

namespace fs = std::filesystem;

struct
{
    const char* loadImagesPath = nullptr;
    const char* loadManifestFileName = nullptr;
    const char* saveImagesPath = nullptr;
    const char* loadCompressedFileName = nullptr;
    const char* saveCompressedFileName = nullptr;
    const char* saveManifestFileName = nullptr;
    ToolInputType inputType = ToolInputType::None;
    std::vector<char const*> loadImagesList;
    std::optional<ntc::BlockCompressedFormat> bcFormat;
    ImageContainer imageFormat = ImageContainer::Auto;
    bool compress = false;
    bool decompress = false;
    bool readManifestFromStdin = false;
    bool loadMips = false;
    bool saveMips = false;
    bool generateMips = false;
    bool optimizeBC = false;
    bool useVulkan = false;
    bool useDX12 = false;
    bool debug = false;
    bool listAdapters = false;
    bool listCudaDevices = false;
    bool describe = false;
    bool discardMaskedOutPixels = false;
    bool enableCoopVec = true;
    bool enableGpuDeflate = false;
    bool printVersion = false;
    int gridSizeScale = 2;
    int numFeatures = NTC_MLP_FEATURES;
    int adapterIndex = -1;
    int cudaDevice = 0;
    int benchmarkIterations = 1;
    float experimentalKnob = 0.f;
    float bitsPerPixel = NAN; // Use an "undefined" value to tell if something came from the command line
    float targetPsnr = NAN;
    float maxBitsPerPixel = NAN;
    bool matchBcPsnr = false;
    float minBcPsnr = 0.f;
    float maxBcPsnr = INFINITY;
    float bcPsnrOffset = 0.f;
    float bcPsnrThreshold = 0.2f;
    std::optional<int> customWidth;
    std::optional<int> customHeight;
    ntc::CompressionSettings compressionSettings;
    ntc::LosslessCompressionSettings losslessCompression;
} g_options;

bool ProcessCommandLine(int argc, const char** argv)
{
    const char* bcFormatString = nullptr;
    const char* imageFormatString = nullptr;
    const char* dimensionsString = nullptr;
    const char* gdeflateString = nullptr;

    struct argparse_option options[] = {
        OPT_GROUP("Actions:"),
        OPT_BOOLEAN('c', "compress", &g_options.compress, "Perform NTC compression"),
        OPT_BOOLEAN('D', "decompress", &g_options.decompress, "Perform NTC decompression (implied when needed)"),
        OPT_BOOLEAN('d', "describe", &g_options.describe, "Describe the contents of a compressed texture set"),
        OPT_BOOLEAN('g', "generateMips", &g_options.generateMips, "Generate MIP level images before compression"),
        OPT_STRING (0,   "loadCompressed", &g_options.loadCompressedFileName, "Load compressed texture set from the specified file"),
        OPT_STRING (0,   "loadImages", &g_options.loadImagesPath, "Load channel images from the specified folder"),
        OPT_STRING (0,   "loadManifest", &g_options.loadManifestFileName, "Load channel images and their parameters using the specified JSON manifest file"),
        OPT_BOOLEAN(0,   "loadMips", &g_options.loadMips, "Load MIP level images from <loadImages>/mips/<texture>.<mip>.<ext> before compression"),
        OPT_BOOLEAN(0,   "optimizeBC", &g_options.optimizeBC, "Run slow BC compression and store acceleration info in the NTC package"),
        OPT_BOOLEAN(0,   "readManifestFromStdin", &g_options.readManifestFromStdin, "Load channel images using a JSON manifest passed through standard input (stdin)"),
        OPT_STRING ('o', "saveCompressed", &g_options.saveCompressedFileName, "Save compressed texture set into the specified file"),
        OPT_STRING ('i', "saveImages", &g_options.saveImagesPath, "Save channel images into the specified folder"),
        OPT_STRING (0,   "saveManifest", &g_options.saveManifestFileName, "Save a manifest JSON file (only for image inputs)"),
        OPT_BOOLEAN(0,   "saveMips", &g_options.saveMips, "Save MIP level images into <saveImages>/mips/ after decompression"),
        OPT_BOOLEAN(0,   "version", &g_options.printVersion, "Print version information and exit"),
        OPT_HELP(),
        
        OPT_GROUP("Basic compression options:"),
        OPT_FLOAT  ('b', "bitsPerPixel", &g_options.bitsPerPixel, "Request an optimal compression configuration for the provided BPP value"),
        OPT_FLOAT  (0,   "maxBitsPerPixel", &g_options.maxBitsPerPixel, "Maximum BPP value to use in the compression parameter search"),
        OPT_FLOAT  ('p', "targetPsnr", &g_options.targetPsnr, "Perform compression parameter search to reach at least the provided PSNR value"),
        
        OPT_GROUP("Custom latent shape selection:"),
        OPT_INTEGER(0, "gridSizeScale", &g_options.gridSizeScale, "Ratio of source image size to high-resolution feature grid size"),
        OPT_INTEGER(0, "numFeatures", &g_options.numFeatures, "Number of features"),
        
        OPT_GROUP("Training process controls:"),
        OPT_FLOAT  (0,   "gridLearningRate", &g_options.compressionSettings.gridLearningRate, "Maximum learning rate for the feature grid"),
        OPT_INTEGER(0,   "kPixelsPerBatch", &g_options.compressionSettings.kPixelsPerBatch, "Number of kilopixels from the image to process in one training step"),
        OPT_FLOAT  (0,   "networkLearningRate", &g_options.compressionSettings.networkLearningRate, "Maximum learning rate for the MLP weights"),
        OPT_INTEGER(0,   "randomSeed", &g_options.compressionSettings.randomSeed, "Random seed, set to a nonzero value to get more stable compression results"),
        OPT_BOOLEAN(0,   "stableTraining", &g_options.compressionSettings.stableTraining, "Use a more expensive but more numerically stable training algorithm for reproducible results"),
        OPT_INTEGER(0,   "stepsPerIteration", &g_options.compressionSettings.stepsPerIteration, "Training steps between progress reports"),
        OPT_INTEGER('S', "trainingSteps", &g_options.compressionSettings.trainingSteps, "Total training step count"),
        
        OPT_GROUP("Output settings:"),
        OPT_STRING ('B', "bcFormat", &bcFormatString, "Set or override the BCn encoding format, BC1-BC7"),
        OPT_STRING ('F', "imageFormat", &imageFormatString, "Set the output file format for color images: Auto (default), BMP, JPG, TGA, PNG, PNG16, EXR"),
        OPT_STRING (0,   "dimensions", &dimensionsString, "Set the dimensions of the NTC texture set before compression, in the 'WxH' format"),
        
        OPT_GROUP("Advanced settings:"),
        OPT_FLOAT  (0,   "bcPsnrThreshold", &g_options.bcPsnrThreshold, "PSNR loss threshold for BC7 optimization, in dB, default value is 0.2"),
        OPT_INTEGER(0,   "benchmark", &g_options.benchmarkIterations, "Number of iterations to run over compute passes for benchmarking"),
        OPT_BOOLEAN(0,   "discardMaskedOutPixels", &g_options.discardMaskedOutPixels, "Ignore contents of pixels where alpha mask is 0.0 (requires the AlphaMask semantic)"),
        OPT_FLOAT  (0,   "experimentalKnob", &g_options.experimentalKnob, "A parameter for NTC development, normally has no effect"),
        OPT_BOOLEAN(0,   "matchBcPsnr", &g_options.matchBcPsnr, "Perform compression parameter search to reach the PSNR value that BCn encoding provides"),
        OPT_FLOAT  (0,   "minBcPsnr", &g_options.minBcPsnr, "When using --matchBcPsnr, minimum PSNR value to use for NTC compression"),
        OPT_FLOAT  (0,   "maxBcPsnr", &g_options.maxBcPsnr, "When using --matchBcPsnr, maximum PSNR value to use for NTC compression"),
        OPT_FLOAT  (0,   "bcPsnrOffset", &g_options.bcPsnrOffset, "When using --matchBcPsnr, offset to apply to BCn PSNR value before NTC compression"),
        OPT_STRING (0,   "gdeflate", &gdeflateString, "Controls which parts of the texture set file should be compressed with GDeflate (off/none, bc, latents, all; default is bc)"),
        OPT_FLOAT  (0,   "gdeflateThreshold", &g_options.losslessCompression.compressionRatioThreshold,
            "Don't use GDeflate compression if sizeof(compressedData) >= sizeof(uncompressedData) * X, default is 0.95"),
        OPT_INTEGER(0,   "gdeflateLevel", &g_options.losslessCompression.compressionLevel, "GDeflate compression level, 0-12, default is 9"),
        OPT_INTEGER(0,   "gdeflateThreads", &g_options.losslessCompression.compressionThreads, "Number of GDeflate compression threads, 0 means auto, -1 means disable"),
        
        OPT_GROUP("GPU and Graphics API settings:"),
        OPT_INTEGER(0, "adapter", &g_options.adapterIndex, "Index of the graphics adapter to use"),
        OPT_BOOLEAN(0, "coopVec", &g_options.enableCoopVec, "Enable CoopVec extensions (default on, use --no-coopVec)"),
        OPT_BOOLEAN(0, "gpuGDeflate", &g_options.enableGpuDeflate, "Enable GPU-based GDeflate decompression"),
        OPT_INTEGER(0, "cudaDevice", &g_options.cudaDevice, "Index of the CUDA device to use"),
        OPT_BOOLEAN(0, "debug", &g_options.debug, "Enable debug features such as Vulkan validation layer or D3D12 debug runtime"),
#if NTC_WITH_DX12
        OPT_BOOLEAN(0, "dx12", &g_options.useDX12, "Use D3D12 API for graphics operations"),
#endif
        OPT_BOOLEAN(0, "listAdapters", &g_options.listAdapters, "Enumerate the graphics adapters present in the system"),
        OPT_BOOLEAN(0, "listCudaDevices", &g_options.listCudaDevices, "Enumerate the CUDA devices present in the system"),
#if NTC_WITH_VULKAN
        OPT_BOOLEAN(0, "vk", &g_options.useVulkan, "Use Vulkan API for graphics operations"),
#endif
        OPT_END()
    };

    static const char* usages[] = {
        "ntc-cli [input-files|input-directory] <actions...> [options...]",
        nullptr
    };

    struct argparse argparse {};
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse,
        "\n"
        "Neural texture compression and decompression tool.\n"
        "\n"
        "Inputs can be specified as positional arguments, in one of four modes:\n"
        "    - Directory with image files (same as --loadImages)\n"
        "    - Individual image files (.png, .tga, .jpg, .jpeg, .exr)\n"
        "    - Manifest file with .json extension (same as --loadManifest)\n"
        "    - Compressed texture set with .ntc extension (same as --loadCompressed)\n"
        "\n"
        "For the manifest file schema, please refer to docs/Manifest.md in the SDK.",
        nullptr);
    argparse_parse(&argparse, argc, argv);

    bool const useGapi = g_options.useVulkan || g_options.useDX12;
    if (useGapi && g_options.listAdapters)
        return true;

    if (g_options.listCudaDevices || g_options.printVersion)
        return true;

    if (!useGapi && g_options.listAdapters)
    {
        fprintf(stderr, "--listAdapters requires either --dx12 or --vk.\n");
        return false;
    }

    if (g_options.useVulkan && g_options.useDX12)
    {
        fprintf(stderr, "Options --vk and --dx12 cannot be used at the same time.\n");
        return false;
    }

    if (g_options.loadManifestFileName && g_options.readManifestFromStdin)
    {
        fprintf(stderr, "Options --loadManifest and --readManifestFromStdin cannot be used at the same time.\n");
        return false;
    }

    // Process explicit inputs
    if (g_options.loadImagesPath)
    {
        if (!fs::is_directory(g_options.loadImagesPath))
        {
            fprintf(stderr, "Input directory '%s' does not exist or is not a directory.\n", g_options.loadImagesPath);
            return false;
        }

        UpdateToolInputType(g_options.inputType, ToolInputType::Directory);
    }

    if (g_options.loadManifestFileName)
    {
        if (!fs::exists(g_options.loadManifestFileName))
        {
            fprintf(stderr, "Manifest file '%s' does not exist.\n", g_options.loadManifestFileName);
            return false;
        }

        UpdateToolInputType(g_options.inputType, ToolInputType::ManifestFile);
    }

    if (g_options.readManifestFromStdin)
    {
        UpdateToolInputType(g_options.inputType, ToolInputType::ManifestStdin);
    }

    if (g_options.loadCompressedFileName)
    {
        if (!fs::exists(g_options.loadCompressedFileName))
        {
            fprintf(stderr, "Input file '%s' does not exist.\n", g_options.loadCompressedFileName);
            return false;
        }

        UpdateToolInputType(g_options.inputType, ToolInputType::CompressedTextureSet);
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
            g_options.loadImagesPath = arg;
        }
        else if (fs::exists(argPath))
        {
            std::string extension = argPath.extension().string();
            LowercaseString(extension);

            if (extension == ".json")
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::ManifestFile);
                g_options.loadManifestFileName = arg;
            }
            else if (extension == ".ntc")
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::CompressedTextureSet);
                g_options.loadCompressedFileName = arg;
            }
            else if (IsSupportedImageFileExtension(extension))
            {
                UpdateToolInputType(g_options.inputType, ToolInputType::Images);
                g_options.loadImagesList.push_back(arg);
            }
            else
            {
                fprintf(stderr, "Unknown input file type '%s'.\n", extension.c_str());
                return false;
            }
        }
        else
        {
            fprintf(stderr, "The file or directory '%s' specified as a positional argument does not exist.\n", arg);
            return false;
        }
    }

    if (g_options.inputType == ToolInputType::None)
    {
        fprintf(stderr, "No inputs.\n");
        return false;
    }

    if (g_options.inputType == ToolInputType::Mixed)
    {
        fprintf(stderr, "Cannot process inputs of mismatching types (image files, directories, manifests, "
            "compressed texture sets) or multiple inputs of the same type except for images.\n");
        return false;
    }

    g_options.benchmarkIterations = std::max(g_options.benchmarkIterations, 1);

    if (g_options.compress && g_options.inputType == ToolInputType::CompressedTextureSet)
    {
        fprintf(stderr, "Cannot compress an already compressed texture set.\n");
        return false;
    }

    if ((g_options.saveCompressedFileName || g_options.decompress) &&
        !(g_options.compress || g_options.inputType == ToolInputType::CompressedTextureSet))
    {
        fprintf(stderr, "To use --decompress or --saveCompressed, either --compress or --loadCompressed must be used.\n");
        return false;
    }

    if (g_options.saveManifestFileName && g_options.inputType == ToolInputType::CompressedTextureSet)
    {
        fprintf(stderr, "Cannot save a manifest file when the input is a compressed texture set.\n");
        return false;
    }

    if (g_options.saveImagesPath && (g_options.compress || g_options.inputType == ToolInputType::CompressedTextureSet))
    {
        // When saving images from a compressed texture set, --decompress is implied.
        g_options.decompress = true;
    }

    if (g_options.generateMips && g_options.loadMips)
    {
        fprintf(stderr, "Options --generateMips and --loadMips cannot be used at the same time.\n");
        return false;
    }

    if (g_options.generateMips && g_options.inputType == ToolInputType::CompressedTextureSet)
    {
        fprintf(stderr, "To use --generateMips, uncompressed images must be loaded first.\n");
        return false;
    }

    if (g_options.optimizeBC && !useGapi)
    {
        fprintf(stderr, "Option --optimizeBC requires either --vk or --dx12.\n");
        return false;
    }

    if (g_options.optimizeBC && !g_options.decompress)
    {
        fprintf(stderr, "Option --optimizeBC requires --decompress.\n");
        return false;
    }

    if (g_options.bcPsnrThreshold < 0.f || g_options.bcPsnrThreshold > 10.f)
    {
        fprintf(stderr, "The --bcPsnrThreshold value (%f) must be between 0 and 10.\n", g_options.bcPsnrThreshold);
        return false;
    }
        
    if (g_options.matchBcPsnr && !std::isnan(g_options.targetPsnr))
    {
        fprintf(stderr, "The --targetPsnr and --matchBcPsnr options cannot be used at the same time.");
        return false;
    }

    if ((g_options.matchBcPsnr || !std::isnan(g_options.targetPsnr)) && !g_options.compress)
    {
        fprintf(stderr, "The --targetPsnr or --matchBcPsnr options require --compress.");
        return false;
    }

    if (g_options.matchBcPsnr && !useGapi)
    {
        fprintf(stderr, "The --matchBcPsnr option requires either --vk or --dx12 (where available).");
        return false;
    }
    
    if (bcFormatString)
    {
        g_options.bcFormat = ParseBlockCompressedFormat(bcFormatString, /* enableAuto = */ true);
        if (!g_options.bcFormat.has_value())
        {
            fprintf(stderr, "Invalid --bcFormat value '%s'.\n", bcFormatString);
            return false;
        }
    }
    
    if (imageFormatString)
    {
        auto parsedFormat = ParseImageContainer(imageFormatString);
        if (parsedFormat.has_value())
        {
            g_options.imageFormat = parsedFormat.value();
        }
        else
        {
            fprintf(stderr, "Invalid --imageFormat value '%s'.\n", imageFormatString);
            return false;
        }
    }

    if (dimensionsString)
    {
        int width = 0, height = 0;
        if (sscanf(dimensionsString, "%dx%d", &width, &height) != 2)
        {
            fprintf(stderr, "Invalid format for --dimensions '%s', must be 'WxH' where W and H are integers.\n",
                dimensionsString);
            return false;
        }

        if (width <= 0 || height <= 0)
        {
            fprintf(stderr, "Invalid values specified in --dimensions (%dx%d), must be 1x1 or more.\n",
                width, height);
            return false;
        }

        g_options.customWidth = width;
        g_options.customHeight = height;
    }

    if (g_options.saveCompressedFileName)
    {
        fs::path outputPath = fs::path(g_options.saveCompressedFileName).parent_path();
        if (!outputPath.empty() && !fs::is_directory(outputPath) && !fs::create_directories(outputPath))
        {
            fprintf(stderr, "Failed to create directories for '%s'.\n", outputPath.generic_string().c_str());
            return false;
        }
    }

    if (g_options.saveImagesPath)
    {
        fs::path outputPath = fs::path(g_options.saveImagesPath);
        if (!fs::is_directory(outputPath) && !fs::create_directories(outputPath))
        {
            fprintf(stderr, "Failed to create directories for '%s'.\n", g_options.saveImagesPath);
            return false;
        }
    }

    if (gdeflateString)
    {
        if (strcasecmp(gdeflateString, "off") == 0 || strcasecmp(gdeflateString, "none") == 0)
        {
            g_options.losslessCompression.compressBCModeBuffers = false;
            g_options.losslessCompression.compressLatents = false;
        }
        else if (strcasecmp(gdeflateString, "bc") == 0)
        {
            g_options.losslessCompression.compressBCModeBuffers = true;
            g_options.losslessCompression.compressLatents = false;
        }
        else if (strcasecmp(gdeflateString, "latents") == 0)
        {
            g_options.losslessCompression.compressBCModeBuffers = false;
            g_options.losslessCompression.compressLatents = true;
        }
        else if (strcasecmp(gdeflateString, "all") == 0)
        {
            g_options.losslessCompression.compressBCModeBuffers = true;
            g_options.losslessCompression.compressLatents = true;
        }
        else
        {
            fprintf(stderr, "Invalid value '%s' for --gdeflate.\n", gdeflateString);
            return false;
        }
    }

    return true;
}

bool SaveImagesFromTextureSet(ntc::IContext* context, ntc::ITextureSet* textureSet)
{
    const ntc::TextureSetDesc& textureSetDesc = textureSet->GetDesc();
    fs::path const outputPath = g_options.saveImagesPath;
    bool mipsDirCreated = false;

    int numTextures = textureSet->GetTextureCount();
    
    std::mutex mutex;
    bool anyErrors = false;

    const int mips = g_options.saveMips ? textureSetDesc.mips : 1;

    for (int textureIndex = 0; textureIndex < numTextures; ++textureIndex)
    {
        ntc::ITextureMetadata* texture = textureSet->GetTexture(textureIndex);
        assert(texture);

        ntc::BlockCompressedFormat bcFormat = texture->GetBlockCompressedFormat();
        if (bcFormat != ntc::BlockCompressedFormat::None)
            continue;

        if (!mipsDirCreated && g_options.saveMips && textureSetDesc.mips > 1)
        {
            fs::path mipsPath = outputPath / "mips";
            if (!fs::is_directory(mipsPath) && !fs::create_directories(mipsPath))
            {
                fprintf(stderr, "Failed to create directory '%s'.\n", mipsPath.generic_string().c_str());
                return false;
            }
            mipsDirCreated = true;
        }

        const char* textureName = texture->GetName();
        int firstChannel = 0;
        int numChannels = 0;
        texture->GetChannels(firstChannel, numChannels);
        ntc::ChannelFormat channelFormat = texture->GetChannelFormat();
        ntc::ColorSpace rgbColorSpace = texture->GetRgbColorSpace();
        ntc::ColorSpace const alphaColorSpace = texture->GetAlphaColorSpace();

        ImageContainer container = g_options.imageFormat;

        // Select the container from texture's channel format if it wasn't provided explicitly
        if (container == ImageContainer::Auto)
        {
            if (channelFormat == ntc::ChannelFormat::FLOAT16 || channelFormat == ntc::ChannelFormat::FLOAT32)
                container = ImageContainer::EXR;
            else if (channelFormat == ntc::ChannelFormat::UNORM16)
                container = ImageContainer::PNG16;
            else
                container = ImageContainer::PNG;
        }

        // Pick the channel format suitable for our container
        channelFormat = GetContainerChannelFormat(container);

        // EXR uses linear data, request that from NTC
        if (container == ImageContainer::EXR)
            rgbColorSpace = ntc::ColorSpace::Linear;

        ntc::ColorSpace const colorSpaces[4] = { rgbColorSpace, rgbColorSpace, rgbColorSpace, alphaColorSpace };

        size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(channelFormat);

        for (int mip = 0; mip < mips; ++mip)
        {
            int mipWidth = std::max(1, textureSetDesc.width >> mip);
            int mipHeight = std::max(1, textureSetDesc.height >> mip);

            size_t const mipDataSize = size_t(mipWidth) * size_t(mipHeight) * size_t(numChannels) * bytesPerComponent;
            uint8_t* data = (uint8_t*)malloc(mipDataSize);

            ntc::ReadChannelsParameters params;
            params.page = ntc::TextureDataPage::Output;
            params.mipLevel = mip;
            params.firstChannel = firstChannel;
            params.numChannels = numChannels;
            params.pOutData = data;
            params.addressSpace = ntc::AddressSpace::Host;
            params.width = mipWidth;
            params.height = mipHeight;
            params.pixelStride = size_t(numChannels) * bytesPerComponent;
            params.rowPitch = size_t(numChannels * mipWidth) * bytesPerComponent;
            params.channelFormat = channelFormat;
            params.dstColorSpaces = colorSpaces;
            params.useDithering = true;

            ntc::Status ntcStatus = textureSet->ReadChannels(params);

            if (ntcStatus != ntc::Status::Ok)
            {
                fprintf(stderr, "Failed to read texture data for texture %d (%s) MIP %d, code = %s: %s\n",
                    textureIndex, textureName, mip, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                free(data);
                return false;
            }

            std::string outputFileName;
            if (g_options.saveMips && mip > 0)
            {
                outputFileName = (outputPath / "mips" / textureName).generic_string();

                char mipStr[8];
                snprintf(mipStr, sizeof(mipStr), ".%02d", mip);
                outputFileName += mipStr;
            }
            else
            {
                outputFileName = (outputPath / textureName).generic_string();
            }
            
            outputFileName += GetContainerExtension(container);

            StartAsyncTask([&mutex, container, outputFileName, mipWidth, mipHeight, numChannels, channelFormat, data, &anyErrors]()
            {
                bool success = SaveImageToContainer(container, data, mipWidth, mipHeight, numChannels, outputFileName.c_str());
                
                // The rest of this function is interlocked with other threads
                std::lock_guard lockGuard(mutex);

                if (!success)
                {
                    anyErrors = true;
                    fprintf(stderr, "Failed to write a texture into '%s'\n", outputFileName.c_str());
                }
                else
                {
                    printf("Saved image '%s': %dx%d pixels, %d channels, %s.\n", outputFileName.c_str(),
                        mipWidth, mipHeight, numChannels, ntc::ChannelFormatToString(channelFormat));
                }

                free(data);
            });
        }
    }


    WaitForAllTasks();

    if (anyErrors)
        return false;

    return true;
}

bool PickLatentShape(ntc::LatentShape& outShape)
{
    if (!std::isnan(g_options.targetPsnr) || g_options.matchBcPsnr)
    {
        // When doing adaptive compression, start with an empty latent space because the first configuration
        // will be given by the adaptive compression session.
        outShape = ntc::LatentShape::Empty();
    }
    else if (!std::isnan(g_options.bitsPerPixel))
    {
        float selectedBpp = 0.f;
        if (ntc::PickLatentShape(g_options.bitsPerPixel, selectedBpp, outShape) != ntc::Status::Ok)
        {
            fprintf(stderr, "Cannot select a latent shape for %.3f bpp.\n", g_options.bitsPerPixel);
            return false;
        }

        printf("Selected latent shape for %.3f bpp: --gridSizeScale %d --numFeatures %d\n",
            selectedBpp, outShape.gridSizeScale, outShape.numFeatures);
    }
    else
    {
        outShape.gridSizeScale = g_options.gridSizeScale;
        outShape.numFeatures = g_options.numFeatures;
    }
    return true;
}

void OverrideTextureBcFormat(ntc::ITextureMetadata* texture, ManifestEntry* manifestEntry)
{
    // Override the BC format from command line, if specified.
    // Overriding with 'none' is also an option here.
    if (g_options.bcFormat.has_value())
    {
        ntc::BlockCompressedFormat bcFormat = *g_options.bcFormat;

        // Automatic selection of BCn mode based on channel count and HDR-ness
        if (bcFormat == BlockCompressedFormat_Auto)
        {
            ntc::ChannelFormat const channelFormat = texture->GetChannelFormat();
            if (channelFormat == ntc::ChannelFormat::FLOAT16 || channelFormat == ntc::ChannelFormat::FLOAT32)
            {
                // HDR textures use only BC6, no other options
                bcFormat = ntc::BlockCompressedFormat::BC6;
            }
            else
            {
                // Best quality options.
                // If you want more control, use a manifest.
                int const channels = texture->GetNumChannels();
                switch(channels)
                {
                    case 1:  bcFormat = ntc::BlockCompressedFormat::BC4; break;
                    case 2:  bcFormat = ntc::BlockCompressedFormat::BC5; break;
                    default: bcFormat = ntc::BlockCompressedFormat::BC7; break;
                }       
            }
        }

        assert(bcFormat != BlockCompressedFormat_Auto);

        texture->SetBlockCompressedFormat(bcFormat);

        // Apply the same override to the manifest entry, if it's provided
        // (in case the user wants to save the manifest)
        if (manifestEntry)
        {
            manifestEntry->bcFormat = bcFormat;
        }
    }
}

ntc::ITextureSet* LoadImages(ntc::IContext* context, Manifest& manifest, bool manifestIsGenerated)
{
    ntc::TextureSetDesc textureSetDesc{};
    textureSetDesc.mips = 1;

    ntc::LatentShape latentShape;
    if (!PickLatentShape(latentShape))
        return nullptr;

    // Count the number of MIP 0 images in the manifest
    int numMipZeroImages = 0;
    for (auto const& entry : manifest.textures)
    {
        if (entry.mipLevel == 0)
            ++numMipZeroImages;
    }

    if (numMipZeroImages > NTC_MAX_CHANNELS)
    {
        if (g_options.loadImagesPath)
        {
            fprintf(stderr, "Too many images (%d) found in the input folder. At most %d channels are supported.\n"
                "Note: when loading images from a folder, a single material with all images is created. "
                "To load a material with only some images from a folder, use manifest files or specify each image "
                "on the command line separately.",
                int(manifest.textures.size()), NTC_MAX_CHANNELS);
        }
        else
        {
            fprintf(stderr, "Too many images (%d) specified in the manifest. At most %d channels are supported.\n",
                int(manifest.textures.size()), NTC_MAX_CHANNELS);
        }
        return nullptr;
    }

    struct SourceImageData
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        int storedChannels = 0;
        int alphaMaskChannel = -1;
        int firstChannel = -1;
        int manifestIndex = 0;
        bool verticalFlip = false;
        std::string channelSwizzle;
        std::array<stbi_uc*, NTC_MAX_MIPS> data {};
        std::string name;
        ntc::ChannelFormat channelFormat = ntc::ChannelFormat::UNORM8;
        ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
        bool isSRGB = false;
        std::vector<float> lossFunctionScales;

        SourceImageData()
        { }

        SourceImageData(SourceImageData& other) = delete;
        SourceImageData(SourceImageData&& other) = delete;

        ~SourceImageData()
        {
            for (auto mipLevel : data)
            {
                if (mipLevel)
                    stbi_image_free((void*)mipLevel);
            }
            data.fill(nullptr);
        }
    };

    std::vector<std::shared_ptr<SourceImageData>> images;
    
    std::mutex mutex;

    bool anyErrors = false;

    // Load the base images (mip level 0)

    int entryIndex = 0;
    for (const auto& entry : manifest.textures)
    {
        if (entry.mipLevel > 0)
            continue;

        StartAsyncTask([&mutex, &images, entry, entryIndex, &textureSetDesc, &anyErrors]()
        {
            std::shared_ptr<SourceImageData> image = std::make_shared<SourceImageData>();

            fs::path const fileName = entry.fileName;
            std::string extension = fileName.extension().generic_string();
            LowercaseString(extension);

            if (extension == ".exr")
            {
                LoadEXR((float**)&image->data[0], &image->width, &image->height, entry.fileName.c_str(), nullptr);
                image->channels = 4;
                image->channelFormat = ntc::ChannelFormat::FLOAT32;
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
                        image->data[0] = (stbi_uc*)stbi_load_from_file_16(imageFile, &image->width, &image->height, &image->channels, STBI_rgb_alpha);
                        image->channelFormat = ntc::ChannelFormat::UNORM16;
                    }
                    else
                    {
                        image->data[0] = stbi_load_from_file(imageFile, &image->width, &image->height, &image->channels, STBI_rgb_alpha);
                        image->channelFormat = ntc::ChannelFormat::UNORM8;
                    }

                    fclose(imageFile);
                }
            }

            // The rest of this function is interlocked with other threads
            std::lock_guard lockGuard(mutex);

            if (!image->data[0])
            {
                fprintf(stderr, "Failed to read image '%s'.\n", entry.fileName.c_str());
                anyErrors = true;
                return;
            }

            image->name = entry.entryName;
            image->isSRGB = entry.isSRGB;
            image->bcFormat = entry.bcFormat;
            image->firstChannel = entry.firstChannel;
            image->manifestIndex = entryIndex;
            image->verticalFlip = entry.verticalFlip;
            image->lossFunctionScales = entry.lossFunctionScales;

            int numLossScales = int(image->lossFunctionScales.size());
            if (numLossScales == 0)
            {
                // Default to 1.0 for all channels
                image->lossFunctionScales.resize(image->channels, 1.f);
            }
            else if (numLossScales == 1)
            {
                // Expand to all channels
                image->lossFunctionScales.resize(image->channels, image->lossFunctionScales[0]);
            }
            else if (numLossScales != image->channels)
            {
                fprintf(stderr, "Number of loss function scales (%d) does not match the number of channels "
                    "(%d) in image '%s'.\n", numLossScales, image->channels, entry.fileName.c_str());
                anyErrors = true;
                return;
            }
            
            printf("Loaded image '%s': %dx%d pixels, %d channels.\n", fileName.filename().generic_string().c_str(),
                image->width, image->height, image->channels);

            image->channelSwizzle = entry.channelSwizzle;
            if (image->channelSwizzle.empty())
                image->storedChannels = image->channels;
            else
                image->storedChannels = image->channelSwizzle.size();

            // Find the alpha mask semantic in the manifest, store the channel index
            for (ImageSemanticBinding const& binding : entry.semantics)
            {
                if (binding.label == SemanticLabel::AlphaMask)
                {
                    image->alphaMaskChannel = binding.firstChannel; // Default value is -1 which means "none"
                }
            }

            textureSetDesc.width = std::max(image->width, textureSetDesc.width);
            textureSetDesc.height = std::max(image->height, textureSetDesc.height);

            images.push_back(image);
        });

        ++entryIndex;
    }

    WaitForAllTasks();

    if (images.empty())
    {
        fprintf(stderr, "No images loaded, exiting.\n");
        return nullptr;
    }

    // Validate the names of images if there are multiple channels.
    if (images.size() > 1)
    {
        for (size_t i = 0; i < images.size() - 1 && !anyErrors; ++i)
        {
            for (size_t ii = i + 1; ii < images.size(); ++ii)
            {
                if (images[i]->name == images[ii]->name)
                {
                    fprintf(stderr, "Multiple images have the same name '%s'.\n"
                        "Make sure that input files have different and non-empty names (before extension).\n",
                        images[i]->name.c_str());
                    anyErrors = true;
                    break;
                }
            }
        }
    }

    if (anyErrors)
    {
        return nullptr;
    }
    
    // Auto-generate the semantics and sRGB flags after loading the images: this needs per-image channel counts

    if (manifestIsGenerated)
    {
        std::vector<SemanticBinding> semantics;
        for (auto& image : images)
        {
            // We don't (currently) need the global semantic table, but we do look fir the alpha mask below
            semantics.clear();

            GuessImageSemantics(image->name, image->channels, image->channelFormat, image->manifestIndex,
                image->isSRGB, semantics);
            
            // Copy the generated semantics into the manifest in case the user requested to save the manifest
            ManifestEntry& manifestEntry = manifest.textures[image->manifestIndex];
            manifestEntry.isSRGB = image->isSRGB;

            // If one of the channels is the alpha mask, remember that
            for (auto const& binding : semantics)
            {
                if (binding.label == SemanticLabel::AlphaMask)
                    image->alphaMaskChannel = binding.firstChannel;

                ImageSemanticBinding manifestBinding;
                manifestBinding.label = binding.label;
                manifestBinding.firstChannel = binding.firstChannel;
                manifestEntry.semantics.push_back(manifestBinding);
            }
        }
    }

    // Load the other mips

    for (const auto& entry : manifest.textures)
    {
        if (entry.mipLevel == 0)
            continue;

        auto found = std::find_if(images.begin(), images.end(), [&entry](std::shared_ptr<SourceImageData> const& image)
            { return image->name == entry.entryName; });

        if (found == images.end())
            continue;

        std::shared_ptr<SourceImageData> const& image = *found;

        textureSetDesc.mips = std::max(textureSetDesc.mips, entry.mipLevel + 1);

        StartAsyncTask([&mutex, &image, entry, &anyErrors]()
        {
            const fs::path fileName = entry.fileName;
            std::string extension = fileName.extension().generic_string();
            LowercaseString(extension);

            int width = 0, height = 0;
            ntc::ChannelFormat format = ntc::ChannelFormat::UNORM8;
            if (extension == ".exr")
            {
                LoadEXR((float**)&image->data[entry.mipLevel], &width, &height, fileName.generic_string().c_str(), nullptr);
                format = ntc::ChannelFormat::FLOAT32;
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
                        image->data[entry.mipLevel] = (stbi_uc*)stbi_load_from_file_16(imageFile, &width, &height, nullptr, STBI_rgb_alpha);
                        format = ntc::ChannelFormat::UNORM16;
                    }
                    else
                    {
                        image->data[entry.mipLevel] = stbi_load_from_file(imageFile, &width, &height, nullptr, STBI_rgb_alpha);
                        format = ntc::ChannelFormat::UNORM8;
                    }

                    fclose(imageFile);
                }
            }

            // The rest of this function is interlocked with other threads
            std::lock_guard lockGuard(mutex);

            if (!image->data[entry.mipLevel])
            {
                fprintf(stderr, "Failed to read image '%s'.\n", fileName.generic_string().c_str());
                anyErrors = true;
                return;
            }

            if (format != image->channelFormat)
            {
                fprintf(stderr, "Image '%s' has pixel format (%s) that differs from the base MIP's pixel format (%s).\n",
                    fileName.generic_string().c_str(), ntc::ChannelFormatToString(format), ntc::ChannelFormatToString(image->channelFormat));
                anyErrors = true;
                return;
            }

            const int expectedWidth = std::max(1, image->width >> entry.mipLevel);
            const int expectedHeight = std::max(1, image->height >> entry.mipLevel);
            if (width != expectedWidth || height != expectedHeight)
            {
                fprintf(stderr, "Image '%s' has incorrect dimensions for MIP level %d: expected %dx%d, got %dx%d.\n",
                    fileName.generic_string().c_str(), entry.mipLevel, expectedWidth, expectedHeight, width, height);
                anyErrors = true;
                return;
            }

            printf("Loaded image '%s': %dx%d pixels.\n", fileName.filename().generic_string().c_str(),
                width, height);
        });
    }

    WaitForAllTasks();

    if (anyErrors)
    {
        return nullptr;
    }

    // Remember the max size of the input textures to create a staging buffer of sufficient size.

    ntc::TextureSetFeatures textureSetFeatures;
    textureSetFeatures.stagingBytesPerPixel = sizeof(float) * 4; // We might have FLOAT32 data on reads
    textureSetFeatures.stagingWidth = textureSetDesc.width;
    textureSetFeatures.stagingHeight = textureSetDesc.height;

    // Override the dimensions from the manifest or command line, if specified.
    // Command line has higher priority.

    textureSetDesc.width = g_options.customWidth.value_or(manifest.width.value_or(textureSetDesc.width));
    textureSetDesc.height = g_options.customHeight.value_or(manifest.height.value_or(textureSetDesc.height));

    if (textureSetDesc.width * 2 < textureSetFeatures.stagingWidth ||
        textureSetDesc.height * 2 < textureSetFeatures.stagingHeight)
    {
        printf("Warning: Texture set dimensions (%dx%d) are less than 1/2 of the maximum input image dimensions "
               "(%dx%d). The resize operation uses a 2x2 bilinear filter, which may produce low quality output.\n",
               textureSetDesc.width, textureSetDesc.height,
               textureSetFeatures.stagingWidth, textureSetFeatures.stagingHeight);
    }

    // Maybe not loading mips, but generating them later

    if (g_options.generateMips)
    {
        textureSetDesc.mips = int(floorf(std::log2f(float(std::max(textureSetDesc.width, textureSetDesc.height)))) + 1);
    }


    // Verify that we have images for all mips

    if (g_options.loadMips)
    {
        for (auto& image : images)
        {
            for (int mip = 0; mip < textureSetDesc.mips; ++mip)
            {
                if (!image->data[mip])
                {
                    fprintf(stderr, "Channel '%s' doesn't have an image for MIP level %d.\n",
                        image->name.c_str(), mip);
                    anyErrors = true;
                }
            }
        }

        if (anyErrors)
        {
            return nullptr;
        }
    }

    // Sort the images in manifest order, to make channel assignment easy to control.

    std::sort(images.begin(), images.end(), [](std::shared_ptr<SourceImageData> const& a, std::shared_ptr<SourceImageData> const& b) {
        return a->manifestIndex < b->manifestIndex;
    });

    // Assign channels to images:
    // Phase 1 - enumerate the explicitly specified channels and make sure they don't collide.

    uint32_t availableChannels = (1u << NTC_MAX_CHANNELS) - 1u;
    for (std::shared_ptr<SourceImageData> const& image : images)
    {
        if (image->firstChannel < 0)
            continue;

        int const min1 = image->firstChannel;
        int const max1 = image->firstChannel + image->storedChannels - 1;

        if (max1 >= NTC_MAX_CHANNELS)
        {   
            fprintf(stderr, "Texture '%s' uses channels %d-%d, and that is out of range of supported channels (0-%d).\n",
                image->name.c_str(), min1, max1, NTC_MAX_CHANNELS - 1);
            return nullptr;
        }
            
        uint32_t const channelMask = ((1u << image->storedChannels) - 1u) << image->firstChannel;
        if (~availableChannels & channelMask)
        {
            int const min1 = image->firstChannel;
            int const max1 = image->firstChannel + image->storedChannels - 1;

            for (std::shared_ptr<SourceImageData> const& otherImage : images)
            {
                int const min2 = otherImage->firstChannel;
                int const max2 = otherImage->firstChannel + otherImage->storedChannels - 1;

                if (image != otherImage && min1 <= max2 && min2 <= max1)
                {
                    fprintf(stderr, "Texture '%s' uses channels %d-%d, and that range intersects with channels %d-%d used by texture '%s'.\n",
                        image->name.c_str(), min1, max1, min2, max2, otherImage->name.c_str());
                    return nullptr;
                }
            }
            assert(false); // We should never get here, if two textures collide, the loop above will find that
            return nullptr;
        }

        availableChannels &= ~channelMask;
    }

    // Phase 2 - assign channels to images that don't have an explicit firstChannel attribute.

    for (std::shared_ptr<SourceImageData> const& image : images)
    {
        if (image->firstChannel >= 0)
            continue;

        uint32_t channelMask = (1u << image->storedChannels) - 1u;
        for (int firstChannel = 0; firstChannel + image->storedChannels <= NTC_MAX_CHANNELS; ++firstChannel)
        {
            if ((availableChannels & channelMask) == channelMask)
            {
                image->firstChannel = firstChannel;
                availableChannels &= ~channelMask;
                break;
            }
            channelMask <<= 1;
        }

        if (image->firstChannel < 0)
        {
            fprintf(stderr, "Failed to allocate %d channel(s) for texture '%s'.\n",
                image->storedChannels, image->name.c_str());
            return nullptr;
        }
    }

    // Derive the texture set's channel count from the highest zero bit in 'availableChannels'

    for (int channelCount = NTC_MAX_CHANNELS; channelCount > 0; --channelCount)
    {
        if ((availableChannels & (1u << (channelCount - 1))) == 0)
        {
            textureSetDesc.channels = channelCount;
            break;
        }
    }

    // Create the texture set object

    ntc::TextureSetWrapper textureSet(context);
    textureSetFeatures.separateRefOutData = true;
    ntc::Status ntcStatus = context->CreateTextureSet(textureSetDesc, textureSetFeatures, textureSet.ptr());
    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Failed to create a texture set for %dx%d pixels, %d channels, %d mips, code = %s\n%s\n",
            textureSetDesc.width, textureSetDesc.height, textureSetDesc.channels, textureSetDesc.mips,
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return nullptr;
    }

    ntcStatus = textureSet->SetLatentShape(latentShape);
    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Failed to set the latent shape to %d/%d, code = %s\n%s\n",
            latentShape.gridSizeScale, latentShape.numFeatures,
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return nullptr;
    }
    
    // Upload the image data into the texture set

    int alphaMaskChannel = -1;
    for (std::shared_ptr<SourceImageData> const& image : images)
    {
        size_t const bytesPerComponent = ntc::GetBytesPerPixelComponent(image->channelFormat);
        size_t const pixelStride = 4 * bytesPerComponent;
        ntc::ColorSpace const srcRgbColorSpace = image->isSRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
        ntc::ColorSpace const dstRgbColorSpace = image->channelFormat == ntc::ChannelFormat::FLOAT32 ? ntc::ColorSpace::HLG : srcRgbColorSpace;
        ntc::ColorSpace const srcAlphaColorSpace = ntc::ColorSpace::Linear;
        ntc::ColorSpace const dstAlphaColorSpace = image->channelFormat == ntc::ChannelFormat::FLOAT32 ? ntc::ColorSpace::HLG : srcAlphaColorSpace;
        ntc::ColorSpace const srcColorSpaces[4] = { srcRgbColorSpace, srcRgbColorSpace, srcRgbColorSpace, srcAlphaColorSpace };
        ntc::ColorSpace const dstColorSpaces[4] = { dstRgbColorSpace, dstRgbColorSpace, dstRgbColorSpace, dstAlphaColorSpace };

        for (int mip = 0; mip < textureSetDesc.mips; ++mip)
        {
            if (!image->data[mip])
                continue;

            int mipWidth = std::max(1, image->width >> mip);
            int mipHeight = std::max(1, image->height >> mip);

            ntc::WriteChannelsParameters params;
            params.mipLevel = mip;
            params.addressSpace = ntc::AddressSpace::Host;
            params.width = mipWidth;
            params.height = mipHeight;
            params.pixelStride = pixelStride;
            params.rowPitch = size_t(mipWidth) * pixelStride;
            params.channelFormat = image->channelFormat;
            params.verticalFlip = image->verticalFlip;
            
            if (image->channelSwizzle.empty())
            {
                // No swizzle - write all channels at once
                params.firstChannel = image->firstChannel;
                params.numChannels = image->channels;
                params.pData = image->data[mip];
                params.srcColorSpaces = srcColorSpaces;
                params.dstColorSpaces = dstColorSpaces;

                ntcStatus = textureSet->WriteChannels(params);
                
                if (mip == 0 && image->alphaMaskChannel >= 0)
                    alphaMaskChannel = image->alphaMaskChannel + image->firstChannel;
            }
            else
            {
                int dstChannelOffset = 0;

                // Loop over the swizzled channels and upload each one individually
                for (char ch : image->channelSwizzle)
                {
                    // Decode the channel letter into an offset using a lookup string
                    char const* channelMap = "RGBA";
                    char const* channelPos = strchr(channelMap, ch);
                    if (!channelPos)
                    {
                        // The format of 'channelSwizzle' is validated when the manifest is loaded,
                        // so 'channelPos' should never be NULL here.
                        assert(false);
                        return nullptr;
                    }

                    int const srcChannelOffset = channelPos - channelMap;
                    if (srcChannelOffset >= image->channels)
                    {
                        fprintf(stderr, "Swizzle '%s' for texture '%s' requests the '%c' channel, which does not exist "
                            "in the source texture (it only has %d channels).\n",
                            image->channelSwizzle.c_str(), image->name.c_str(), ch, image->channels);
                        return nullptr;
                    }

                    // Write one channel
                    params.firstChannel = image->firstChannel + dstChannelOffset;
                    params.numChannels = 1;
                    params.pData = image->data[mip] + srcChannelOffset * bytesPerComponent;
                    params.srcColorSpaces = srcColorSpaces + srcChannelOffset;
                    params.dstColorSpaces = dstColorSpaces + dstChannelOffset;
                    
                    ntcStatus = textureSet->WriteChannels(params);

                    // Just check the return code, a failure message will be printed below
                    if (ntcStatus != ntc::Status::Ok)
                        break;

                    // If this channel was the alpha mask in the image before swizzle,
                    // store its index in the texture set after swizzle.
                    if (mip == 0 && srcChannelOffset == image->alphaMaskChannel)
                        alphaMaskChannel = image->firstChannel + dstChannelOffset;

                    ++dstChannelOffset;
                }
            }

            if (ntcStatus != ntc::Status::Ok)
            {
                fprintf(stderr, "Failed to upload texture data to NTC texture set, code = %s\n%s\n",
                    ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                return nullptr;
            }
        }
        
        ntc::ITextureMetadata* texture = textureSet->AddTexture();
        texture->SetName(image->name.c_str());
        texture->SetChannels(image->firstChannel, image->storedChannels);
        texture->SetChannelFormat(image->channelFormat);
        texture->SetBlockCompressedFormat(image->bcFormat);
        texture->SetRgbColorSpace(srcRgbColorSpace);
        texture->SetAlphaColorSpace(srcAlphaColorSpace);

        OverrideTextureBcFormat(texture, &manifest.textures[image->manifestIndex]);

        // Copy the loss function scales for this image's channels into the global array
        assert(image->channels == int(image->lossFunctionScales.size()));
        for (int channel = 0; channel < image->channels; ++channel)
        {
            g_options.compressionSettings.lossFunctionScales[image->firstChannel + channel] =
                image->lossFunctionScales[channel];
        }
    }

    // Pass the alpha mask channel index to NTC

    if (alphaMaskChannel >= 0)
    {
        textureSet->SetMaskChannelIndex(alphaMaskChannel, g_options.discardMaskedOutPixels);
    }

    // Generate the mips if requested

    if (g_options.generateMips)
    {
        ntcStatus = textureSet->GenerateMips();
        if (ntcStatus != ntc::Status::Ok)
        {
            fprintf(stderr, "Failed to generate MIP images, code = %s\n%s\n",
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return nullptr;
        }
    }

    // Done - detach the "smart" pointer and return the raw one

    ntc::ITextureSet* rawTextureSet = textureSet;
    textureSet.Detach();

    return rawTextureSet;
}

bool CompressTextureSet(ntc::IContext* context, ntc::ITextureSet* textureSet, float* outFinalPsnr)
{
    ntc::Status ntcStatus = textureSet->BeginCompression(g_options.compressionSettings);
    CHECK_NTC_RESULT(BeginCompression);

    ntc::CompressionStats stats;
    do
    {
        ntcStatus = textureSet->RunCompressionSteps(&stats);
        if (ntcStatus == ntc::Status::Incomplete || ntcStatus == ntc::Status::Ok)
        {
            printf("Training: %d steps, %.4f ms/step, intermediate PSNR: %.2f dB\r", stats.currentStep,
                stats.millisecondsPerStep, ntc::LossToPSNR(stats.loss));
            fflush(stdout);
        }
    } while (ntcStatus == ntc::Status::Incomplete);
    CHECK_NTC_RESULT(RunCompressionSteps);
    printf("\n");

    ntcStatus = textureSet->FinalizeCompression();
    CHECK_NTC_RESULT(FinalizeCompression);

    if (outFinalPsnr)
        *outFinalPsnr = ntc::LossToPSNR(stats.loss);

    return true;
}

struct AdaptiveSearchResult
{
    std::vector<uint8_t> compressedData;
    ntc::LatentShape latentShape;
    float bitsPerPixel = 0.f;
    float psnr = 0.f;
};

bool CompressTextureSetWithTargetPSNR(ntc::IContext* context, ntc::ITextureSet* textureSet)
{
    ntc::Status ntcStatus;

    ntc::AdaptiveCompressionSessionWrapper session(context);
    ntcStatus = context->CreateAdaptiveCompressionSession(session.ptr());
    CHECK_NTC_RESULT("CreateAdaptiveCompressionSession")

    float const targetPsnr = g_options.targetPsnr;
    float const maxBitsPerPixel = std::isnan(g_options.maxBitsPerPixel) ? 0.f : g_options.maxBitsPerPixel;
    ntcStatus = session->Reset(targetPsnr, maxBitsPerPixel);
    CHECK_NTC_RESULT("Reset")
    
    printf("Starting search for optimal BPP to achieve %.2f dB PSNR.\n", g_options.targetPsnr);
    
    int experimentCount = 0;
    std::vector<AdaptiveSearchResult> results;

    while (!session->Finished())
    {
        float bitsPerPixel;
        ntc::LatentShape latentShape;
        session->GetCurrentPreset(&bitsPerPixel, &latentShape);

        printf("Experiment %d: %.2f bpp...\n", experimentCount + 1, bitsPerPixel);

        ntcStatus = textureSet->SetLatentShape(latentShape);
        CHECK_NTC_RESULT(SetLatentShape)

        float psnr = NAN;
        if (!CompressTextureSet(context, textureSet, &psnr))
            return false;

        // Store the compression result
        AdaptiveSearchResult result;
        result.bitsPerPixel = bitsPerPixel;
        result.latentShape = latentShape;
        result.psnr = psnr;

        // Save the compressed data to an in-memory vector
        size_t bufferSize = textureSet->GetOutputStreamSize();
        result.compressedData.resize(bufferSize);
        ntcStatus = textureSet->SaveToMemory(result.compressedData.data(), &bufferSize);
        CHECK_NTC_RESULT(SaveToMemory)

        // Trim the buffer to the actual size of the saved data
        result.compressedData.resize(bufferSize);

        results.push_back(std::move(result));
        
        session->Next(psnr);
        ++experimentCount;
    }

    // Get and validate the index of the final result
    int finalIndex = session->GetIndexOfFinalRun();
    if (finalIndex < 0 || finalIndex >= int(results.size()))
    {
        fprintf(stderr, "Internal error: GetIndexOfFinalRun() returned %d, which is not a valid index!\n", finalIndex);
        return false;
    }

    // Find the final compresison result
    auto const& result = results[finalIndex];

    printf("Selected compression rate: %.2f bpp, %.2f dB PSNR.\n", result.bitsPerPixel, result.psnr);
    if (result.psnr < targetPsnr)
        printf("WARNING: Target PSNR of %.2f dB was not reached!\n", targetPsnr);

    // If the texture set already has the final shape, do nothing - its data is valid.
    if (result.latentShape == textureSet->GetLatentShape())
        return true;

    // Otherwise, restore the final compression result into the texture set.
    ntcStatus = textureSet->LoadFromMemory(result.compressedData.data(), result.compressedData.size());
    CHECK_NTC_RESULT(LoadFromMemory)

    return true;
}

bool DecompressTextureSet(ntc::IContext* context, ntc::ITextureSet* textureSet, bool useInt8Weights)
{
    ntc::DecompressionStats stats;
    ntc::Status ntcStatus = textureSet->Decompress(&stats, useInt8Weights);
    CHECK_NTC_RESULT(Decompress);

    printf("CUDA decompression time: %.3f ms\n", stats.gpuTimeMilliseconds);

    if (g_options.inputType == ToolInputType::Directory ||
        g_options.inputType == ToolInputType::ManifestFile ||
        g_options.inputType == ToolInputType::ManifestStdin ||
        g_options.inputType == ToolInputType::Images)
    {
        printf("Overall PSNR (%s weights): %.2f dB\n", useInt8Weights ? "INT8" : "FP8", ntc::LossToPSNR(stats.overallLoss));
        
        if (!useInt8Weights)
        {
            size_t maxNameLength = 0;
            for (int i = 0; i < textureSet->GetTextureCount(); ++i)
            {
                maxNameLength = std::max(maxNameLength, strlen(textureSet->GetTexture(i)->GetName()));
            }

            printf("Per-texture PSNR:\n");
            for (int i = 0; i < textureSet->GetTextureCount(); ++i)
            {
                ntc::ITextureMetadata* texture = textureSet->GetTexture(i);
                int firstChannel, numChannels;
                texture->GetChannels(firstChannel, numChannels);

                float textureMSE = 0.f;
                for (int ch = firstChannel; ch < firstChannel + numChannels; ++ch)
                {
                    textureMSE += stats.perChannelLoss[ch];
                }
                textureMSE /= float(numChannels);

                printf("  %-*s : %.2f dB [ ", int(maxNameLength), texture->GetName(), ntc::LossToPSNR(textureMSE));
                for (int ch = firstChannel; ch < firstChannel + numChannels; ++ch)
                {
                    printf("%.2f ", ntc::LossToPSNR(stats.perChannelLoss[ch]));
                }
                printf("]\n");
            }
        }

        if (textureSet->GetDesc().mips > 1)
        {
            for (int mip = 0; mip < textureSet->GetDesc().mips; ++mip)
            {
                printf("MIP %2d  PSNR: %.2f dB\n", mip, ntc::LossToPSNR(stats.perMipLoss[mip]));
            }
        }
    }

    return true;
}

size_t GetTextureSetPixelCount(ntc::ITextureSet* textureSet)
{
    size_t texturePixels = 0;
    ntc::TextureSetDesc const& desc = textureSet->GetDesc();
    for (int mip = 0; mip < desc.mips; ++mip)
    {
        int mipWidth = std::max(1, desc.width >> mip);
        int mipHeight = std::max(1, desc.height >> mip);
        texturePixels += size_t(mipWidth) * size_t(mipHeight);
    }
    return texturePixels;
}

bool SaveCompressedTextureSet(ntc::IContext* context, ntc::ITextureSet* textureSet)
{
    ntc::FileStreamWrapper outputStream(context);
    
    ntc::Status ntcStatus = context->OpenFile(g_options.saveCompressedFileName, true, outputStream.ptr());
    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Cannot open output file '%s', code = %s\n%s\n",
            g_options.saveCompressedFileName, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    ntcStatus = textureSet->ConfigureLosslessCompression(g_options.losslessCompression);
    CHECK_NTC_RESULT(ConfigureLosslessCompression);

    ntc::LosslessCompressionStats losslessStats;
    ntcStatus = textureSet->SaveToStream(outputStream, &losslessStats);
    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Failed to save compressed texture to output file '%s', code = %s\n%s\n",
            g_options.saveCompressedFileName, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    uint64_t const fileSize = outputStream->Tell();
    size_t const texturePixels = GetTextureSetPixelCount(textureSet);
    float const bpp = 8.f * float(fileSize) / float(texturePixels);

    printf("Saved '%s'\n", g_options.saveCompressedFileName);
    printf("File size: %" PRIu64 " bytes, %.2f bits per pixel.\n", fileSize, bpp);
    if (g_options.losslessCompression.compressBCModeBuffers || g_options.losslessCompression.compressLatents)
    {
        double const compressedBufferRatioPercents = (losslessStats.originalSizeOfCompressedBuffers == 0) ? 0.0 :
            100.0 * double(losslessStats.sizeOfCompressedBuffers) / double(losslessStats.originalSizeOfCompressedBuffers);

        double const overallCmpressionRatioPercents = 100.0 *
            double(losslessStats.sizeOfCompressedBuffers + losslessStats.sizeOfUncompressedBuffers) /
            double(losslessStats.originalSizeOfCompressedBuffers + losslessStats.sizeOfUncompressedBuffers);

        printf("%s compressed %d buffers out of %d, compressed ratio %.1f%%, overall ratio %.1f%%, time %.2f ms\n",
            ntc::CompressionTypeToString(g_options.losslessCompression.algorithm),
            losslessStats.compressedBuffers, losslessStats.totalBuffers,
            compressedBufferRatioPercents, overallCmpressionRatioPercents, losslessStats.compressionTimeMs);
    }

    return true;
}

ntc::ITextureSet* LoadCompressedTextureSet(ntc::IContext* context)
{
    ntc::ITextureSet* textureSet = nullptr;
    ntc::TextureSetFeatures textureSetFeatures;
    textureSetFeatures.enableCompression = false;
    textureSetFeatures.stagingBytesPerPixel = 16;
    
    ntc::Status ntcStatus = context->CreateCompressedTextureSetFromFile(
        g_options.loadCompressedFileName, textureSetFeatures, &textureSet);

    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Failed to load compressed texture from file '%s', code = %s\n%s\n",
            g_options.loadCompressedFileName, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return nullptr;
    }
    
    return textureSet;
}

donut::app::DeviceCreationParameters GetGraphicsDeviceParameters(nvrhi::GraphicsAPI graphicsApi)
{
    donut::app::DeviceCreationParameters deviceParams;
    deviceParams.infoLogSeverity = donut::log::Severity::None;
    deviceParams.adapterIndex = g_options.adapterIndex;
    deviceParams.enableDebugRuntime = g_options.debug;
    deviceParams.enableNvrhiValidationLayer = g_options.debug;

    SetNtcGraphicsDeviceParameters(deviceParams, graphicsApi, true, g_options.enableCoopVec, nullptr);

    return deviceParams;
}

void DescribeTextureSet(ntc::ITextureSetMetadata* textureSet)
{
    ntc::TextureSetDesc const& desc = textureSet->GetDesc();
    printf("Dimensions: %dx%d, %d channels, %d mip level(s)\n", desc.width, desc.height, desc.channels, desc.mips);
    
    ntc::LatentShape const& latentShape = textureSet->GetLatentShape();
    printf("Base compression rate: --bitsPerPixel %.3f\n", ntc::GetLatentShapeBitsPerPixel(latentShape));
    printf("Latent shape: --gridSizeScale %d --numFeatures %d\n",
        latentShape.gridSizeScale, latentShape.numFeatures);
    printf("Inference weights: Int8 [%c], FP8 [%c]\n",
        textureSet->IsInferenceWeightTypeSupported(ntc::InferenceWeightType::GenericInt8) ? 'Y' : 'N',
        textureSet->IsInferenceWeightTypeSupported(ntc::InferenceWeightType::GenericFP8) ? 'Y' : 'N');
    
    ntc::CompressionType latentCompression = ntc::CompressionType::None;
    ntc::LatentTextureDesc const latentTextureDesc = textureSet->GetLatentTextureDesc();
    int totalLatentBuffers = 0;
    int compressedLatentBuffers = 0;
    for (int mipLevel = 0; mipLevel < latentTextureDesc.mipLevels; ++mipLevel)
    {
        for (int layerIndex = 0; layerIndex < latentTextureDesc.arraySize; ++layerIndex)
        {
            ntc::LatentTextureFootprint footprint;
            ntc::Status ntcStatus = textureSet->GetLatentTextureFootprint(mipLevel, layerIndex, footprint);

            if (ntcStatus == ntc::Status::Ok)
            {
                ++totalLatentBuffers;
                if (footprint.buffer.compressionType != ntc::CompressionType::None)
                {
                    latentCompression = footprint.buffer.compressionType;
                    ++compressedLatentBuffers;
                }
            }
        }
    }
    printf("Latent compression: %s", ntc::CompressionTypeToString(latentCompression));
    if (latentCompression != ntc::CompressionType::None)
    {
        printf(" (%d/%d slices)", compressedLatentBuffers, totalLatentBuffers);
    }
    printf("\n");
        
    printf("Textures:\n");
    for (int i = 0; i < textureSet->GetTextureCount(); ++i)
    {
        ntc::ITextureMetadata* texture = textureSet->GetTexture(i);
        int firstChannel, numChannels;
        texture->GetChannels(firstChannel, numChannels);
        printf("%d: %s\n", i, texture->GetName());
        printf("   Channels: %d-%d\n", firstChannel, firstChannel + numChannels - 1);
        printf("   Channel format: %s\n", ntc::ChannelFormatToString(texture->GetChannelFormat()));
        printf("   BCn format: %s\n", ntc::BlockCompressedFormatToString(texture->GetBlockCompressedFormat()));
        printf("   RGB color space: %s\n", ntc::ColorSpaceToString(texture->GetRgbColorSpace()));
        if (numChannels > 3)
            printf("   Alpha color space: %s\n", ntc::ColorSpaceToString(texture->GetAlphaColorSpace()));

        bool colorSpacesMatch = true;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            ntc::ColorSpace const dstColorSpace = (ch < 3) ? texture->GetRgbColorSpace() : texture->GetAlphaColorSpace();
            if (textureSet->GetChannelStorageColorSpace(firstChannel + ch) != dstColorSpace)
            {
                colorSpacesMatch = false;
                break;
            }
        }

        if (!colorSpacesMatch)
        {
            printf("   Storage color spaces: ");
            for (int ch = 0; ch < numChannels; ++ch)
            {
                if (ch > 0) printf(", ");
                printf("%s", ntc::ColorSpaceToString(textureSet->GetChannelStorageColorSpace(firstChannel + ch)));
            }
            printf("\n");
        }

        if (texture->GetBlockCompressedFormat() == ntc::BlockCompressedFormat::BC7)
        {
            ntc::CompressionType compression = ntc::CompressionType::None;
            int totalModeBuffers = 0;
            int compressedModeBuffers = 0;

            for (int mipLevel = 0; mipLevel < desc.mips; ++mipLevel)
            {
                ntc::BufferFootprint modeBufferFootprint = texture->GetBC7ModeBufferFootprint(mipLevel);

                if (modeBufferFootprint.rangeInStream.size != 0)
                {
                    ++totalModeBuffers;

                    if (modeBufferFootprint.compressionType != ntc::CompressionType::None)
                    {
                        compression = modeBufferFootprint.compressionType;
                        ++compressedModeBuffers;
                    }
                }
            }

            char const* modeBufferInfo = (totalModeBuffers == 0) ? "None"
                : (totalModeBuffers == desc.mips) ? "All MIPs"
                : "Partial";

            printf("   BC7 acceleration data: %s", modeBufferInfo);
            if (totalModeBuffers > 0)
            {
                printf(", Compression: %s", ntc::CompressionTypeToString(compression));
                if (compression != ntc::CompressionType::None)
                    printf(" (%d/%d buffers)", compressedModeBuffers, totalModeBuffers);
            }
            printf("\n");
        }
    }
}

static bool ListCudaDevices()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Call to cudaGetDeviceCount failed, error code = %s.\n", cudaGetErrorName(err));
        return false;
    }

    if (count == 0)
    {
        printf("No CUDA devices available.\n");
        return true;
    }

    for (int device = 0; device < count; ++device)
    {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Call to cudaGetDeviceProperties(%d) failed, error code = %s.\n",
                device, cudaGetErrorName(err));
            return false;
        }

        printf("Device %d: %s (compute capability %d.%d, %zu MB VRAM)\n", device, prop.name,
            prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    }

    return true;
}

void OverrideBcFormats(ntc::ITextureSetMetadata* textureSetMetadata)
{
    for (int textureIndex = 0; textureIndex < textureSetMetadata->GetTextureCount(); ++textureIndex)
    {
        OverrideTextureBcFormat(textureSetMetadata->GetTexture(textureIndex), nullptr);
    }
}

bool AnyBlockCompressedTextures(ntc::ITextureSetMetadata* textureSetMetadata)
{
    for (int textureIndex = 0; textureIndex < textureSetMetadata->GetTextureCount(); ++textureIndex)
    {
        ntc::ITextureMetadata* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        if (textureMetadata->GetBlockCompressedFormat() != ntc::BlockCompressedFormat::None)
        {
            return true;
        }
    }
    return false;
}

class CustomAllocator : public ntc::IAllocator
{
public:
    void* Allocate(size_t size) override
    {
        void* ptr = malloc(size);
        // printf("Allocating %zu bytes at %p.\n", size, ptr);
        m_bytesAllocated += size;
        return ptr;
    }

    void Deallocate(void* ptr, size_t size) override
    {
        if (!ptr)
            return;
        // printf("Deallocating %zu bytes at %p.\n", size, ptr);
        m_bytesAllocated -= size;
        free(ptr);
    }

    int64_t GetBytesAllocated() const
    {
        return m_bytesAllocated;
    }

private:
    int64_t m_bytesAllocated = 0;
};


int main(int argc, const char** argv)
{
    donut::log::ConsoleApplicationMode();
    donut::log::SetMinSeverity(donut::log::Severity::Warning);

    if (!ProcessCommandLine(argc, argv))
        return 1;
    
    if (g_options.printVersion)
    {
        ntc::VersionInfo libVersion = ntc::GetLibraryVersion();
        printf("LibNTC version: %d.%d.%d %s-%s\n", libVersion.major, libVersion.minor, libVersion.point,
            libVersion.branch, libVersion.commitHash);

        ntc::VersionInfo sdkVersion = GetNtcSdkVersion();
        printf("Tools version:  %s-%s\n", sdkVersion.branch, sdkVersion.commitHash);

        return 0;
    }


    if (g_options.listCudaDevices)
    {
        if (ListCudaDevices())
            return 0;
        else
            return 1;
    }

    bool const useGapi = g_options.useVulkan || g_options.useDX12;

    bool const graphicsDecompressMode = g_options.inputType == ToolInputType::CompressedTextureSet && useGapi 
        && g_options.decompress && !g_options.optimizeBC;

    bool const describeMode = g_options.inputType == ToolInputType::CompressedTextureSet && g_options.describe
        && !g_options.decompress && !g_options.saveCompressedFileName;

    bool const useCuda = !describeMode && !graphicsDecompressMode;

    cudaDeviceProp cudaDeviceProperties{};
    if (g_options.cudaDevice >= 0 && useCuda)
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaSuccess && count > 0)
        {
            err = cudaGetDeviceProperties(&cudaDeviceProperties, g_options.cudaDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Call to cudaGetDeviceProperties(%d) failed, error code = %s.\n",
                    g_options.cudaDevice, cudaGetErrorName(err));
            }
        }
    }

    CustomAllocator customAllocator;

    typedef std::unique_ptr<donut::app::DeviceManager, void(*)(donut::app::DeviceManager*)> DeviceManagerPtr;
    DeviceManagerPtr deviceManager = DeviceManagerPtr(nullptr, nullptr);
    nvrhi::DeviceHandle device;
    nvrhi::CommandListHandle commandList;
    nvrhi::TimerQueryHandle timerQuery;
    std::unique_ptr<GDeflateFeatures> gdeflateFeatures;

    if (useGapi)
    {
        using namespace donut::app;

        nvrhi::GraphicsAPI const graphicsApi = g_options.useVulkan
            ? nvrhi::GraphicsAPI::VULKAN
            : nvrhi::GraphicsAPI::D3D12;
        
        // Create a device manager, wrap it with unique_ptr and a custom deleter that calls Shutdown.
        deviceManager = std::unique_ptr<DeviceManager, void(*)(DeviceManager*)>(DeviceManager::Create(graphicsApi), 
            [](DeviceManager* dm) {
                dm->Shutdown();
                delete dm;
            }
        );

        DeviceCreationParameters deviceParams = GetGraphicsDeviceParameters(graphicsApi);

        if (!deviceManager->CreateInstance(deviceParams))
        {
            fprintf(stderr, "Cannot initialize a %s subsystem.", nvrhi::utils::GraphicsAPIToString(graphicsApi));
            return 1;
        }

        std::vector<donut::app::AdapterInfo> adapters;
        if (!deviceManager->EnumerateAdapters(adapters))
        {
            fprintf(stderr, "Cannot enumerate graphics adapters.");
            return 1;
        }

        if (g_options.listAdapters)
        {
            for (int adapterIndex = 0; adapterIndex < int(adapters.size()); ++adapterIndex)
            {
                auto const& info = adapters[adapterIndex];
                int deviceMemoryMB = int(info.dedicatedVideoMemory / (1024 * 1024));
                printf("Adapter %d: %s (%d MB VRAM)\n", adapterIndex, info.name.c_str(), deviceMemoryMB);
            }

            return 0;
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
                printf("Warning: Couldn't find a matching %s adapter for the selected CUDA device %d (%s).\n",
                    nvrhi::utils::GraphicsAPIToString(graphicsApi), g_options.cudaDevice, cudaDeviceProperties.name);
            }
        }

        if (!deviceManager->CreateHeadlessDevice(deviceParams))
        {
            fprintf(stderr, "Cannot initialize a %s device.\n", nvrhi::utils::GraphicsAPIToString(graphicsApi));
            return 1;
        }

        device = deviceManager->GetDevice();
        commandList = device->createCommandList();
        timerQuery = device->createTimerQuery();
        if (g_options.enableGpuDeflate)
        {
            gdeflateFeatures = InitGDeflate(device, g_options.debug);
        }
    }

    // Initialize the NTC context with or without the graphics device
    ntc::ContextParameters contextParams;
    contextParams.pAllocator = &customAllocator;
    contextParams.cudaDevice = useCuda ? g_options.cudaDevice : ntc::DisableCudaDevice;
    
    if (deviceManager)
    {
        ntc::GraphicsAPI const ntcGapi = deviceManager->GetGraphicsAPI() == nvrhi::GraphicsAPI::D3D12
            ? ntc::GraphicsAPI::D3D12
            : ntc::GraphicsAPI::Vulkan;

        bool const osSupportsCoopVec = (ntcGapi == ntc::GraphicsAPI::D3D12) ? IsDX12DeveloperModeEnabled() : true;

        contextParams.graphicsApi = ntcGapi;
        contextParams.d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        contextParams.vkInstance = device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
        contextParams.vkPhysicalDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
        contextParams.vkDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
        contextParams.enableCooperativeVector = osSupportsCoopVec && g_options.enableCoopVec;
    }

    ntc::ContextWrapper context;
    ntc::Status ntcStatus = ntc::CreateContext(context.ptr(), contextParams);
    if (ntcStatus != ntc::Status::Ok && !(ntcStatus == ntc::Status::CudaUnavailable && !useCuda))
    {
        fprintf(stderr, "Failed to create an NTC context, code = %s: %s\n",
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        if (ntcStatus == ntc::Status::CudaUnavailable)
        {
            fprintf(stderr, "\n"
                "For decompression of NTC texture sets on GPUs that do not support CUDA, "
                "please use --vk or --dx12 (where available).\n"
                "All other image processing operations require CUDA.\n");
        }
        return 1;
    }

    if (cudaDeviceProperties.major > 0 && ntcStatus != ntc::Status::CudaUnavailable)
    {
        printf("Using %s with CUDA API. Compute capability %d.%d\n",
            cudaDeviceProperties.name, cudaDeviceProperties.major, cudaDeviceProperties.minor);
    }

    if (useGapi)
    {
        printf("Using %s with %s API. CoopVec [%c], GDeflate [%c]\n",
            deviceManager->GetRendererString(),
            nvrhi::utils::GraphicsAPIToString(deviceManager->GetGraphicsAPI()),
            context->IsCooperativeVectorSupported() ? 'Y' : 'N',
            gdeflateFeatures && gdeflateFeatures->gpuDecompressionSupported ? 'Y' : 'N');
    }

    if (graphicsDecompressMode || describeMode)
    {
        assert(g_options.loadCompressedFileName); // parseCommandLine checks this condition, but let's be sure...

        ntc::FileStreamWrapper inputFile(context);
        ntcStatus = context->OpenFile(g_options.loadCompressedFileName, false, inputFile.ptr());
        if (ntcStatus != ntc::Status::Ok)
        {
            fprintf(stderr, "Failed to open input file '%s', code = %s: %s\n", g_options.loadCompressedFileName,
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return 1;
        }

        ntc::TextureSetMetadataWrapper metadata(context);
        ntcStatus = context->CreateTextureSetMetadataFromStream(inputFile, metadata.ptr());
        if (ntcStatus != ntc::Status::Ok)
        {
            fprintf(stderr, "Failed to load texture set metadata from '%s', code = %s: %s\n", g_options.loadCompressedFileName,
                ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return 1;
        }

        OverrideBcFormats(metadata);

        if (g_options.describe)
        {
            DescribeTextureSet(metadata);
        }

        if (describeMode)
            return 0;

        int const mipLevels = g_options.saveMips ? metadata->GetDesc().mips : 1;

        GraphicsResourcesForTextureSet graphicsResources;
        if (!CreateGraphicsResourcesFromMetadata(context, device, metadata, mipLevels, false, graphicsResources))
            return 1;

        GraphicsDecompressionPass gdp(device, NTC_MAX_CHANNELS * NTC_MAX_MIPS);

        if (!gdp.Init())
        {
            fprintf(stderr, "GraphicsDecompressionPass::Init failed.\n");
            return 1;
        }
        
        std::vector<float> iterationTimes;
        iterationTimes.resize(g_options.benchmarkIterations);

        for (int iteration = 0; iteration < g_options.benchmarkIterations; ++iteration)
        {
            bool const decompressSucceeded = DecompressTextureSetWithGraphicsAPI(device, commandList,
                timerQuery, gdp, gdeflateFeatures.get(),
                context, metadata, iteration == 0 ? inputFile.Get() : nullptr, mipLevels, graphicsResources);

            if (!decompressSucceeded)
                return 1;

            device->waitForIdle();
            device->runGarbageCollection();

            float const decompressTimeSeconds = device->getTimerQueryTime(timerQuery);
            iterationTimes[iteration] = decompressTimeSeconds;
        }
        
        if (g_options.benchmarkIterations > 1)
        {
            float const medianDecompressionTime = Median(iterationTimes);
            printf("Median decompression time over %d iterations: %.3f ms\n", g_options.benchmarkIterations,
                medianDecompressionTime * 1e3f);
        }

        bool const anyBCTextures = AnyBlockCompressedTextures(metadata);

        if (g_options.saveImagesPath)
        {
            if (anyBCTextures)
            {
                if (!BlockCompressAndSaveGraphicsTextures(context, metadata, inputFile.Get(),
                    device, commandList, timerQuery, gdeflateFeatures.get(),
                    g_options.saveImagesPath, g_options.benchmarkIterations, graphicsResources))
                    return 1;
            }

            if (!SaveGraphicsStagingTextures(metadata, device, g_options.saveImagesPath, g_options.imageFormat,
                g_options.saveMips, graphicsResources))
                return 1;
        }
    }
    else
    {
        ntc::TextureSetWrapper textureSet(context);

        Manifest manifest;
        switch (g_options.inputType)
        {
            case ToolInputType::Directory: {
                assert(g_options.loadImagesPath);

                GenerateManifestFromDirectory(g_options.loadImagesPath, g_options.loadMips, manifest);
                *textureSet.ptr() = LoadImages(context, manifest, true);
                break;
            }
            case ToolInputType::Images: {
                assert(!g_options.loadImagesList.empty());

                GenerateManifestFromFileList(g_options.loadImagesList, manifest);
                *textureSet.ptr() = LoadImages(context, manifest, true);
                break;
            }
            case ToolInputType::ManifestFile: {
                assert(g_options.loadManifestFileName);

                std::string manifestError;
                if (!ReadManifestFromFile(g_options.loadManifestFileName, manifest, manifestError))
                {
                    fprintf(stderr, "%s\n", manifestError.c_str());
                    return 1;
                }

                *textureSet.ptr() = LoadImages(context, manifest, false);
                break;
            }
            case ToolInputType::ManifestStdin: {
                std::string manifestError;
                if (!ReadManifestFromStdin(manifest, manifestError))
                {
                    fprintf(stderr, "%s\n", manifestError.c_str());
                    return 1;
                }

                *textureSet.ptr() = LoadImages(context, manifest, false);
                break;
            }
            case ToolInputType::CompressedTextureSet: {
                assert(g_options.loadCompressedFileName);

                *textureSet.ptr() = LoadCompressedTextureSet(context);
                
                if (textureSet)
                {
                    // LoadImages already applies overrides to the texture set and the manifest,
                    // so process the overrides here for the case of loading a compressed texture set.
                    OverrideBcFormats(textureSet);
                }

                break;
            }
            default:
                assert(!"Unsupported input type!");
                return 1;
        }

        if (!textureSet)
            return 1;

        if (g_options.describe)
        {
            DescribeTextureSet(textureSet);
        }

        if (g_options.saveManifestFileName)
        {
            std::string manifestError;
            if (!WriteManifestToFile(g_options.saveManifestFileName, manifest, manifestError))
            {
                fprintf(stderr, "%s\n", manifestError.c_str());
                return 1;
            }
            else
            {
                printf("Saved manifest to '%s'\n", g_options.saveManifestFileName);
            }
        }

        textureSet->SetExperimentalKnob(g_options.experimentalKnob);

        bool const anyBCTextures = AnyBlockCompressedTextures(textureSet);

        if (g_options.matchBcPsnr && !anyBCTextures)
        {
            fprintf(stderr, "--matchBcPsnr requires that at least one texture in the set is compressed to a BCn format.\n");
            return 1;
        }

        GraphicsResourcesForTextureSet graphicsResources;
        if (g_options.matchBcPsnr || g_options.optimizeBC || g_options.saveImagesPath && anyBCTextures)
        {
            // Verify that we have a graphics device - cannot do that in ProcessCommandLine
            // because we don't know if there are any BCn textures at that point...
            if (!device)
            {
                fprintf(stderr, "BCn encoding requires either --vk or --dx12 (where available).\n"
                    "To save images in a non-BC format, use --bcFormat none.\n");
                return 1;
            }

            int const mipLevels = textureSet->GetDesc().mips;

            if (!CreateGraphicsResourcesFromMetadata(context, device, textureSet,
                mipLevels, /* enableCudaSharing = */ true, graphicsResources))
                return 1;
        }

        if (g_options.matchBcPsnr)
        {
            if (!CopyTextureSetDataIntoGraphicsTextures(context, textureSet, ntc::TextureDataPage::Reference,
                /* allMipLevels = */ false, /* onlyBlockCompressedFormats = */ true, graphicsResources))
                return 1;

            if (!ComputePsnrForBlockCompressedTextureSet(context, textureSet, device,
                commandList, graphicsResources, g_options.targetPsnr))
                return 1;

            // Apply the user-specified offset and limits
            g_options.targetPsnr = std::min(g_options.maxBcPsnr, std::max(g_options.minBcPsnr,
                g_options.targetPsnr + g_options.bcPsnrOffset));

            printf("Selected target PSNR: %.2f dB.\n", g_options.targetPsnr);
        }
        
        if (g_options.compress)
        {
            if (std::isnan(g_options.targetPsnr))
            {
                if (!CompressTextureSet(context, textureSet, nullptr))
                    return 1;
            }
            else
            {
                if (!CompressTextureSetWithTargetPSNR(context, textureSet))
                    return 1;
            }
        }

        if (g_options.decompress)
        {
            if (g_options.compress)
            {
                if (!DecompressTextureSet(context, textureSet, /* useInt8Weights = */ true))
                    return 1;
            }

            if (!DecompressTextureSet(context, textureSet, /* useInt8Weights = */ false))
                return 1;
        }

        if (g_options.optimizeBC || g_options.saveImagesPath && anyBCTextures)
        {
            ntc::TextureDataPage const sourcePage = g_options.decompress
                ? ntc::TextureDataPage::Output
                : ntc::TextureDataPage::Reference;
                
            if (!CopyTextureSetDataIntoGraphicsTextures(context, textureSet, sourcePage,
                /* allMipLevels = */ true, /* onlyBlockCompressedFormats = */ true, graphicsResources))
                return 1;
        }

        if (g_options.optimizeBC)
        {
            if (!OptimizeBlockCompression(context, textureSet, device,
                commandList, g_options.bcPsnrThreshold, graphicsResources))
                return 1;
        }

        if (g_options.saveImagesPath)
        {
            if (anyBCTextures)
            {
                if (!BlockCompressAndSaveGraphicsTextures(context, textureSet, nullptr,
                    device, commandList, timerQuery, gdeflateFeatures.get(),
                    g_options.saveImagesPath, g_options.benchmarkIterations, graphicsResources))
                    return 1;
            }
                
            if (!SaveImagesFromTextureSet(context, textureSet))
                return 1;
        }

        if (g_options.saveCompressedFileName)
        {
            if (!SaveCompressedTextureSet(context, textureSet))
                return 1;
        }
        else if (g_options.compress)
        {
            size_t estimatedSize;
            if (ntc::EstimateCompressedTextureSetSize(textureSet->GetDesc(), textureSet->GetLatentShape(),
                estimatedSize) == ntc::Status::Ok)
            {
                size_t const texturePixels = GetTextureSetPixelCount(textureSet);
                float const bpp = 8.f * float(estimatedSize) / float(texturePixels);
                printf("Estimated file size: %zu bytes, %.2f bits per pixel.\n", estimatedSize, bpp);
            }
        }
    }

    context.Release();

    if (customAllocator.GetBytesAllocated() != 0)
        fprintf(stderr, "Library leaked %" PRIi64 " bytes!\n", customAllocator.GetBytesAllocated());

    return 0;
}