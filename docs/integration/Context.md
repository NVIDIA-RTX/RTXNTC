# Initializing LibNTC Context and Graphics Device

A context is required for all LibNTC operations. The same context type supports de/compression operations with CUDA and various operations with graphics APIs.

To create a basic CUDA context for compression:
```c++
ntc::IContext* context = nullptr;
ntc::ContextParameters contextParams;
ntc::Status ntcStatus = ntc::CreateContext(&context, contextParams);
if (ntcStatus != ntc::Status::Ok)
{ /* Handle the error */ }
```

To create a context that supports graphics operations:
```c++
ntc::IContext* context = nullptr;
ntc::ContextParameters contextParams;
if (using_Vulkan)
{
    contextParams.graphicsApi = ntc::GraphicsAPI::Vulkan;
    contextParams.vkInstance = vkInstance;
    contextParams.vkPhysicalDevice = vkPhysicalDevice;
    contextParams.vkDevice = vkDevice;
}
else if (using_DX12)
{
    contextParams.graphicsApi = ntc::GraphicsAPI::D3D12;
    contextParams.d3d12Device = d3d12Device;
}

// Optionally, use the 'enableCooperativeVector' to disable CoopVec support if needed.
contextParams.enableCooperativeVector = ...;

ntc::Status ntcStatus = ntc::CreateContext(&context, contextParams);
if (ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable)
{
    // Handle the error.
    // Note that the CudaUnavailable status exists and is returned when there is no CUDA capable device,
    // in which case the context will not support compression operations or creating ITextureSet objects.
}
```

To release the context, use `ntc::DestroyContext(context)`. Alternatively, use the `ntc::ContextWrapper` class for RAII-style resource management.

All methods of the `IContext` are thread-safe. Other interfaces like `ITextureSet` make no such promises.

## Error Reporting

All LibNTC functions that return `ntc::Status` also update the internal error message buffer. The contents of this buffer can be obtained by calling `ntc::GetLastErrorMessage()` immediately after the failed call. The buffer is stored in a `thread_local` variable, which means the error message must be obtained on the same thread that called the previous function that failed.

For more basic error reporting, use `ntc::StatusToString(status)`.

## Vulkan Extensions and Features

When NTC is used to decompress texture sets on Vulkan, the following feature flags should be set if supported:

```
VkPhysicalDeviceFeatures::shaderInt16 = true

VkPhysicalDeviceVulkan11Features::storageBuffer16BitAccess = true

VkPhysicalDeviceVulkan12Features::shaderFloat16 = true
VkPhysicalDeviceVulkan12Features::storageBuffer8BitAccess = true

VkPhysicalDeviceVulkan13Features::shaderDemoteToHelperInvocation = true
VkPhysicalDeviceVulkan13Features::shaderIntegerDotProduct = true
```

Optionally, to use Cooperative Vector based inference, the following extensions and feature flags should be enabled:

```
VK_NV_cooperative_vector extension

VkPhysicalDeviceCooperativeVectorFeaturesNV::cooperativeVector = true
```

## DX12 Extensions

When using Cooperative Vector based inference on DX12, the following things should be done:

- Agility SDK linked and enabled in the executable;
- `D3D12EnableExperimentalFeatures` called with the `D3D12ExperimentalShaderModels` feature before device creation.
