# Implementing Inference on Sample with LibNTC

NTC is designed to support decompression of individual texels in the texture set, which means it can be efficiently used to decompress only the texels needed to render a specific view. In this case, the decompression logic is executed directly in the pixel or ray tracing shader where material textures would normally be sampled. This mode is called Inference on Sample.

Compared to regular texture sampling, decompressing texels from NTC is a relatively expensive operation in terms of computation, and it only returns one unfiltered texel with all material channels at a time. This has two important consequences: 

1. Inference on Sample should only be used on high-performance GPUs that support Cooperative Vector extensions. We provide a fallback implementation that uses DP4a for decompression instead of CoopVec, but that is significantly slower and should only be used for functional validation.
2. Simulating regular trilinear or anisotropic texture filtering with NTC would be prohibitively expensive (although functionally possible), so Inference on Sample should be used in combination with Stochastic Texture Filtering (STF) instead, and filtered by a denoiser or DLSS after shading. See [Filtering After Shading with Stochastic Texture Filtering](https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/) for more information.

Implementing Inference on Sample in a renderer is relatively straightforward and generally consists of three phases: parsing the texture set file, uploading data to the GPU, and running inference in the shader.

### 1. Parsing the texture set file

The first part is exactly the same as with graphics API decompression, or Inference on Load. Open the input file or create a custom stream object, and use it to construct the `ITextureSetMetadata` object. That object can be used later to query information about textures in the set, dimensions, and so on.

```c++
ntc::FileStreamWrapper inputFile(context);
ntcStatus = context->OpenFile(fileName, /* write = */ false, inputFile.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::TextureSetMetadataWrapper metadata(context);
ntcStatus = context->CreateTextureSetMetadataFromStream(inputFile, metadata.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Once you have the texture set metadata object, use the `ITextureSetMetadata::GetLatentTextureDesc(...)`, `ITextureSetMetadata::GetLatentTextureFootprint(...)`, `ITextureSetMetadata::GetInferenceWeights(...)`, `ITextureSetMetadata::ConvertInferenceWeights(...)` in the same way that you would do for Infernece on Load, and upload the latents and weights to the GPU. Then use the `IContext::MakeInferenceData(...)` method to obtain the data needed to run inference in the shader. Note the `weightType` parameter; it indicates which version of inference function the application will use, which affects the layout of the weights. When the library determines that the graphics device supports CoopVec, the `MakeInferenceData` method will be able to provide weights for the CoopVec decompression functions; when there is no such support, a call to `GetInferenceWeights` or `MakeInferenceData` with those weight types will return `Status::Unsupported`.

```c++
// Select the weight type that matches the version of the inference shader that's being used.
auto const weightType = ntc::InferenceWeightType::GenericInt8;

void const* pWeightData = nullptr;
size_t uploadSize = 0;
size_t convertedSize = 0;
ntcStatus = metadata->GetInferenceWeights(weightType, &pWeightData, &uploadSize, &convertedSize);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::InferenceData inferenceData;
ntcStatus = m_ntcContext->MakeInferenceData(metadata, weightType, &inferenceData);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

### 2. Uploading data to the GPU

Inference on Sample needs four inputs, obtained through different ways:

1. The constants describing texture set geometry and other parameters. They are represented by the `NtcTextureSetConstants` structure and populated by `IContext::MakeInferenceData(...)`.
2. The network weights. These weights are provided by `ITextureSetMetadata::GetInferenceWeights(...)` and should be uploaded into a ByteAddressBuffer (nonzero offsets are supported).
3. The latent texture. The latents comprise the bulk of the NTC texture set data and can be read directly from the NTC file into a Texture2DArray, without going through the library. To create the latent texture array, use `ITextureSetMetadata::GetLatentTextureDesc(...)`. To find out how to read and upload data into that texture array, use `ITextureSetMetadata::GetLatentTextureFootprint(...)`. See the [Inference on Load guide](InferenceOnLoad.md) for more information.
4. The latent sampler. It's a simple texture sampler with bilinear filtering and wrap addressing.

The code for uploading these buffers depends on the engine and its graphics API abstraction layer. For an example using NVRHI, see [NtcMaterialLoader.cpp](/samples/renderer/NtcMaterialLoader.cpp) in the Renderer sample.

### 3. Running inference in the shader

The shader functions necessary to perform Inference on Sample are provided with LibNTC, in the [`libntc/shaders/Inference.hlsli`](/libraries/RTXNTC-Library/include/libntc/shaders/Inference.hlsli) and [`libntc/shaders/InferenceCoopVec.hlsli`](/libraries/RTXNTC-Library/include/libntc/shaders/InferenceCoopVec.hlsli) files. These headers are compatible with the latest DXC and Slang compilers.

There are a few main function that perform inference, i.e. compute texture colors for a given texel position. They are called `NtcSampleTextureSet` or `NtcSampleTextureSet_CoopVec_FP8`, differing only in the set of instructions that they use and the weights that they require. They have the same signature:

```c++
bool NtcSampleTextureSet[_CoopVec_FP8](
    NtcTextureSetConstants desc,
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    ByteAddressBuffer weightsBuffer,
    uint weightsOffset,
    int2 texel,
    int mipLevel,
    bool convertToLinearColorSpace,
    out float outputs[NTC_MLP_OUTPUT_CHANNELS = 16]);
```

The `desc` and the `weightsBuffer` buffers are provided earlier by the `MakeInferenceData` and `GetInferenceWeights` functions, and `latentsTexture`  contains the latents, as described above. The `texel` and `mipLevel` parameters point at the specific texel that needs to be decoded. The `convertToLinearColorSpace` parameter tells if the outputs should be converted to linear color space using texture set metadata, or returned in their storage color spaces. The results are placed into the `outputs` array, in the same order that was used when placing textures in the texture set during compression. There is no shader-side API to distribute the output channels into per-texture vectors; that is up to the application. The simplest solution is to use a fixed mapping from texture semantics to the channel indices, like it's done in the Renderer sample app.

Note that the `NtcSampleTextureSet` function takes an integer texel position and not normalized UV coordinates. Applications should calculate the texel position before calling the NTC function, and obtain the texture set dimensions and mip level count using the `NtcGetTextureDimensions` and `NtcGetTextureMipLevels` functions, respectively.

For a complete example running Inference on Sample combined with Stochastic Texture Filtering, see [`renderer/NtcForwardShadingPass.hlsl`](/samples/renderer/NtcForwardShadingPass.hlsl).
