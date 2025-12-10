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

#if NTC_WITH_DX12
#include <directx/d3d12.h>
extern "C"
{
    _declspec(dllexport) extern const unsigned int D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;
    _declspec(dllexport) extern const char* D3D12SDKPath = ".\\d3d12\\";
}
#endif

#include <ntc-utils/DeviceUtils.h>

#include <libntc/ntc.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

#if NTC_WITH_DX12
static bool g_dx12DeveloperModeEnabled = false;
#endif
#if NTC_WITH_VULKAN
static bool g_vkMemoryDecompressionSupported = false;
#endif

constexpr bool DirectStorageForceCPUDecompression = false;

bool IsDX12DeveloperModeEnabled()
{
#if NTC_WITH_DX12
    return g_dx12DeveloperModeEnabled;
#else
    return false;
#endif
}

void SetNtcGraphicsDeviceParameters(
    donut::app::DeviceCreationParameters& deviceParams,
    nvrhi::GraphicsAPI graphicsApi,
    bool enableSharedMemory,
    bool enableDX12ExperimentalFeatures,
    char const* windowTitle)
{
#if NTC_WITH_VULKAN
    if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
    {
        if (enableSharedMemory)
        {
#ifdef _WIN32
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);

        // Add feature structures querying for cooperative vector support and DP4a support
        static VkPhysicalDeviceCooperativeVectorFeaturesNV cooperativeVectorFeatures{};
        cooperativeVectorFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
        static VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures{};
        replicatedCompositesFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT;
        replicatedCompositesFeatures.pNext = &cooperativeVectorFeatures;
        static VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.pNext = &replicatedCompositesFeatures;
        static VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan11Features;
        static VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        static VkPhysicalDeviceMemoryDecompressionFeaturesNV memoryDecompressionFeatures{};
        memoryDecompressionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_NV;
        memoryDecompressionFeatures.pNext = &vulkan13Features;
        deviceParams.physicalDeviceFeatures2Extensions = &memoryDecompressionFeatures;
        
        // Set the callback to modify some bits in VkDeviceCreateInfo before creating the device
        deviceParams.deviceCreateInfoCallback = [](VkDeviceCreateInfo& info)
        {
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->shaderInt16 = true;
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->fragmentStoresAndAtomics = true;

            // Iterate through the structure chain and find the structures to patch
            VkBaseOutStructure* pCurrent = reinterpret_cast<VkBaseOutStructure*>(&info);
            VkBaseOutStructure* pLast = nullptr;
            while (pCurrent)
            {
                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
                {
                    reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(pCurrent)->storageBuffer16BitAccess = 
                        vulkan11Features.storageBuffer16BitAccess;
                }

                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                {
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->shaderFloat16 = 
                        vulkan12Features.shaderFloat16;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModel = true;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModelDeviceScope = true;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->storageBuffer8BitAccess = 
                        vulkan12Features.storageBuffer8BitAccess;
                }

                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                {
                    reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(pCurrent)->shaderIntegerDotProduct =
                        vulkan13Features.shaderIntegerDotProduct;
                    reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(pCurrent)->shaderDemoteToHelperInvocation =
                        vulkan13Features.shaderDemoteToHelperInvocation;
                }

                pLast = pCurrent;
                pCurrent = pCurrent->pNext;
            }

            // If cooperative vector is supported, add a feature structure enabling it on the device
            if (pLast && cooperativeVectorFeatures.cooperativeVector)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&cooperativeVectorFeatures);
                cooperativeVectorFeatures.pNext = nullptr;
                pLast = pLast->pNext;
            }

            // If replicated composites are supported, add a feature structure enabling it on the device
            if (pLast && replicatedCompositesFeatures.shaderReplicatedComposites)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&replicatedCompositesFeatures);
                replicatedCompositesFeatures.pNext = nullptr;
                pLast = pLast->pNext;
            }

            // If memory decompression is supported, add a feature structure enabling it on the device
            if (pLast && memoryDecompressionFeatures.memoryDecompression)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&memoryDecompressionFeatures);
                memoryDecompressionFeatures.pNext = nullptr;
                pLast = pLast->pNext;
                g_vkMemoryDecompressionSupported = true;
            }
        };
    }
#endif

#if NTC_WITH_DX12
    g_dx12DeveloperModeEnabled = false;
    if (graphicsApi == nvrhi::GraphicsAPI::D3D12 && enableDX12ExperimentalFeatures)
    {
        UUID Features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
        HRESULT hr = D3D12EnableExperimentalFeatures(_countof(Features), Features, nullptr, nullptr);

        if (FAILED(hr))
        {
            char const* messageText = 
                "Couldn't enable D3D12 experimental shader models. Cooperative Vector features will not be available.\n"
                "Please make sure that Developer Mode is enabled in the Windows system settings.";

            if (windowTitle)
            {
                MessageBoxA(NULL, messageText, windowTitle, MB_ICONWARNING);
            }
            else
            {
                donut::log::warning("%s", messageText);
            }
        }
        else
        {
            g_dx12DeveloperModeEnabled = true;
        }
    }
#endif
}

#if NTC_WITH_DX12
nvrhi::RefCountPtr<IDStorageQueue2> CreateDStorageQueue(ID3D12Device* d3dDevice, bool debugMode)
{
    DSTORAGE_CONFIGURATION config{};
    config.DisableTelemetry = TRUE; // No, Microsoft, telemetry by default is not a good thing.
    config.NumSubmitThreads = 1;
    config.DisableGpuDecompression = DirectStorageForceCPUDecompression;
    if (FAILED(DStorageSetConfiguration(&config)))
        return nullptr;

    nvrhi::RefCountPtr<IDStorageFactory> factory;
    if (FAILED(DStorageGetFactory(IID_PPV_ARGS(&factory))))
        return nullptr;
    
    if (debugMode)
    {
        factory->SetDebugFlags(DSTORAGE_DEBUG_SHOW_ERRORS | DSTORAGE_DEBUG_BREAK_ON_ERROR);
    }
    
    DSTORAGE_QUEUE_DESC queueDesc{};
    queueDesc.Device = d3dDevice;
    queueDesc.Capacity = 1024;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queueDesc.Name = "NTC Decompression Queue";
    nvrhi::RefCountPtr<IDStorageQueue2> queue;
    if (FAILED(factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue))))
        return nullptr;

    return queue;
}
#endif

std::unique_ptr<GDeflateFeatures> InitGDeflate(nvrhi::IDevice* device, bool debugMode)
{
    auto features = std::make_unique<GDeflateFeatures>();

#if NTC_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        if (d3dDevice)
        {
            features->dstorageQueue = CreateDStorageQueue(d3dDevice, debugMode);
            
            if (features->dstorageQueue)
            {
                DSTORAGE_COMPRESSION_SUPPORT compressionSupport = features->dstorageQueue->GetCompressionSupport(
                    DSTORAGE_COMPRESSION_FORMAT_GDEFLATE);
                
                DSTORAGE_COMPRESSION_SUPPORT minimalSupport = DirectStorageForceCPUDecompression
                    ? DSTORAGE_COMPRESSION_SUPPORT_CPU_FALLBACK
                    : (DSTORAGE_COMPRESSION_SUPPORT_GPU_FALLBACK |
                       DSTORAGE_COMPRESSION_SUPPORT_GPU_OPTIMIZED);
                if ((compressionSupport & minimalSupport) != 0)
                {
                    features->gpuDecompressionSupported = true;
                }
                
                features->dstorageEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
            }

            if (!features->gpuDecompressionSupported)
            {
                // Something failed above, release the resources
                features->dstorageQueue = nullptr;
            }
        }
    }
#endif
#if NTC_WITH_VULKAN
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        features->gpuDecompressionSupported = g_vkMemoryDecompressionSupported;
    }
#endif

    return features;
}

GDeflateFeatures::~GDeflateFeatures()
{
#if NTC_WITH_DX12
    if (dstorageEvent)
    {
        CloseHandle(dstorageEvent);
        dstorageEvent = NULL;
    }
#endif
}
