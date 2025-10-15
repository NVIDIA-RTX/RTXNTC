# Compressing and Decompressing Texture Sets with LibNTC

The CUDA-based compression and decompression API is centered around the `ntc::ITextureSet` interface.

## Compression

First, create a texture set:

```c++
// Use the RAII wrapper to make sure that the texture set is destroyed later.
ntc::TextureSetWrapper textureSet(context);

ntc::TextureSetDesc desc;
desc.width = ...;
desc.height = ...;
desc.channels = ...; // Total number of channels in all textures, up to 16
desc.mips = ...;

ntc::TextureSetFeatures features; // Default values are OK for most use cases

ntcStatus = context->CreateTextureSet(desc, features, textureSet.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error;
```

Texture sets are created with storage for color data, but without storage for any compressed data or any specific compression settings. Before compression can run, the target compression parameters (latent shape) must be defined:

```c++
// Pick the compression settings automatically from a target BPP value.
// You can adjust the latent shape directly, but the parameter space is huge with many similar options.
float requestedBitsPerPixel = 5.0f; // Or something else
float actualBpp;
ntc::LatentShape latentShape;
ntc::Status ntcStatus = ntc::PickLatentShape(requestedBitsPerPixel, actualBpp, latentShape);
if (ntcStatus != ntc::Status::Ok)
    // Handle the error;

ntcStatus = textureSet->SetLatentShape(latentShape);
if (ntcStatus != ntc::Status::Ok)
    // Handle the error;
```

If not using the `TextureSetWrapper` class, the texture set must be destroyed explicitly later:
```c++
context->DestroyTextureSet(textureSet);
```

The texture set objects holds a copy of all texture data. So, the texture data and metadata needs to be uploaded to the texture set first before compression starts:

```c++
// Sort-of-pseudocode, assuming that 'texture' has all those members...
int firstChannel = 0;
for (auto& texture : textures)
{
    // Determine the color spaces for the RGB and A channels of the texture
    ntc::ColorSpace rgbColorSpace = texture->isSRGB ? ntc::ColorSpace::sRGB : ntc::ColorSpace::Linear;
    ntc::ColorSpace alphaColorSpace = ntc::ColorSpace::Linear;
    ntc::ColorSpace colorSpaces[4] = { rgbColorSpace, rgbColorSpace, rgbColorSpace, alphaColorSpace };

    ntc::ITextureMetadata* textureMetadata = textureSet->AddTexture();
    textureMetadata->SetName(texture->name); // Some string identifier, not necessarily a file name
    textureMetadata->SetChannels(firstChannel, texture->numChannels);
    textureMetadata->SetPixelFormat(ntc::PixelFormat::BC7); // This is the desired output pixel format, not input
    textureMetadata->SetRgbColorSpace(rgbColorSpace);
    textureMetadata->SetAlphaColorSpace(alphaColorSpace);

    // Fill out the WriteChannels parameters structure
    ntc::WriteChannelsParameters params;
    params.mipLevel = 0;
    params.firstChannel = firstChannel;
    params.numChannels = texture->numCchannels;
    params.pData = texture->pixelData; // This is where the source data is located.
    params.addressSpace = ntc::AddressSpace::Host; // ...or Device if it's on the same CUDA device
    params.width = texture->width;
    params.height = texture->height;
    params.pixelStride = size_t(texture->numChannels);
    params.rowPitch = size_t(image.width * texture->numChannels);
    params.channelFormat = ntc::PixelFormat::UNORM8; // This is the input pixel format.
    params.srcColorSpaces = colorSpaces;
    params.dstColorSpaces = colorSpaces;

    ntc::Status ntcStatus = textureSet->WriteChannels(params);
    
    if (ntcStatus != ntc::Status::Ok)
    {
        // Handle the error. Use ntc::GetLastErrorMessage() for details.
    }

    // Allocate sequential channels of the texture set to each texture.
    // You can use other allocation schemes, such as using fixed channel indices
    // for specific PBR semantics.
    firstChannel += texture->numChannels;
}
```

Once all texture data has been uploaded, compression can begin.

```c++
// Default values are OK; these affect training speed and quality.
ntc::CompressionSettings settings;
ntc::Status ntcStatus = textureSet->BeginCompression(settings);
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::CompressionStats stats;
do
{
    ntcStatus = textureSet->RunCompressionSteps(&stats);
    if (ntcStatus == ntc::Status::Incomplete || ntcStatus == ntc::Status::Ok)
        // Provide a progress report to the user.
} while (ntcStatus == ntc::Status::Incomplete);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntcStatus = textureSet->FinalizeCompression();

if (ntcStatus != ntc::Status::Ok)
    // Handle the error here as well.
```

Once compression is completed, the compressed texture set can be written out into a file or into a memory buffer. Here's how to write it into a file:

```c++
ntc::FileStreamWrapper outputStream(context);
ntc::Status ntcStatus = context->OpenFile(fileName, /* write = */ true, outputStream.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntcStatus = textureSet->SaveToStream(outputStream);
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

// outputStream will be destroyed and file will be closed at the end of the code block.
```

Alternatively, the convenience functions `ITextureSet::SaveToMemory` and `ITextureSet::SaveToFile` may be used instead of creating the stream object and using `SaveToStream`.

## Adaptive Compression

The term "adaptive compression" refers to an algorithm that finds the optimal Bits per Pixel (BPP) value that results in at least the requested PSNR metric for a given texture set. This algorithm is implemented in the NTC library through the `IAdaptiveCompressionSession` sessuion interface. The interface provides functions that will guide a compression loop; the loop itself must be implemented on the application side.

First, create the session object:

```c++
AdaptiveCompressionSessionWrapper session(context);
ntcStatus = context->CreateAdaptiveCompressionSession(session.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Then, initialize the session with search parameters:

```c++
session->Reset(targetPsnr, maxBitsPerPixel);
```

After that, perform the compression loop, serializing every compression result into a vector in memory and storing those results. This is necessary because the optimal result will not necessarily be achieved on the last compression run, and storing intermediate results is much faster than re-running the compression again with optimal parameters.

```c++
std::vector<std::vector<uint8_t>> compressedVersions;
while (!session->Finished())
{
    // Obtain the current compression settings from the session
    float bpp;
    LatentShape latentShape;
    session->GetCurrentPreset(&bpp, &latentShape);
 
    // Resize the latents in the texture set.
    // The texture set needs some latent shape when it is created,
    // but when doing adaptive compression, that shape doesn't matter.
    ntcStatus = textureSet->SetLatentShape(latentShape);
    if (ntcStatus != ntc::Status::Ok)
        // Handle the error.

    // Perform compression on textureSet and obtain PSNR.
    // Implementation details omitted here, see the Compression section above.
    float psnr;
    CompressTextureSet(textureSet, &psnr);

    // Serialize the texture set into 'compressedData'
    std::vector<uint8_t> compressedData;
    size_t bufferSize = textureSet->GetOutputStreamSize();
    compressedData.resize(bufferSize);
    ntcStatus = textureSet->SaveToMemory(compressedData.data(), &bufferSize);
    if (ntcStatus != ntc::Status::Ok)
        // Handle the error.

    // Trim the buffer to the actual size of the saved data
    compressedData.resize(bufferSize);

    // Save the compressed data for later use
    compressedVersions.push_back(serializedData);

    // Pass the PSNR from the current compression run to the session and let it decide the next step
    session->Next(psnr);
}
```

Once the loop is finished, obtain the index of the optimal run from the session and save the data to a file or elsewhere.

```c++
int const finalIndex = session->GetIndexOfFinalRun();

std::vector<uint8_t> const& finalCompressedData = compressedVersions[finalIndex];
```

## Decompression with CUDA

The same `ITextureSet` interface can be used to load a compressed texture set and unpack it into the internal storage as raw pixel data.

First, create the texture set from a stream:

```c++
ntc::FileStreamWrapper inputStream(context);
ntc::Status ntcStatus = context->OpenFile(fileName, /* write = */ false, inputStream.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::TextureSetWrapper textureSet(context);
ntc::TextureSetFeatures features;
ntcStatus = context->CreateCompressedTextureSetFromStream(inputStream, features, textureSet.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Alternatively, the convenience functions `IContext::CreateCompressedTextureSetFromMemory` and `ITextureSet::CreateCompressedTextureSetFromFile` may be used instead of creating the stream object and using `CreateCompressedTextureSetFromStream`.

Then, decompress the texture set contents:

```c++
// Use 'stats' to find the decompression quality metrics if reference images were provided
// and the decompression GPU time measurement.
ntc::DecompressionStats stats;
ntcStatus = textureSet->Decompress(&stats, /* useFP8Weights = */ false);
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Finally, enumerate the textures and read the texture set contents:

```c++
// Get the descriptor of the texture set to find its dimensions later
ntc::TextureSetDesc const& textureSetDesc = textureSet->GetDesc();

// Iterate over all textures in the set...
int const textureCount = textureSet->GetTextureCount();
for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
{
    ntc::ITextureMetadata* textureMetadata = textureSet->GetTexture(textureIndex);
    assert(textureMetadata); // When used like this, GetTexture should always return a valid texture object.

    // Find out where this texture is stored in the set and how many channels it has
    int firstChannel, numChannels;
    textureMetadata->GetChannels(firstChannel, numChannels);

    // Read the color spaces from the texture metadata
    ntc::ColorSpace rgbColorSpace = texture->GetRgbColorSpace();
    ntc::ColorSpace alphaColorSpace = texture->GetAlphaColorSpace();
    ntc::ColorSpace colorSpaces[4] = { rgbColorSpace, rgbColorSpace, rgbColorSpace, alphaColorSpace };

    // Allocate storage for this texture
    uint8_t* textureData = malloc(numChannels * textureSetDesc.width * textureSetDesc.height);

    // Fill out the ReadChannels parameters structure
    ntc::ReadChannelsParameters params;
    params.page = ntc::TextureDataPage::Output;
    params.mipLevel = 0;
    params.firstChannel = firstChannel;
    params.numChannels = numChannels;
    params.pOutData = textureData;
    params.addressSpace = ntc::AddressSpace::Host;
    params.width = textureSetDesc.width;
    params.height = textureSetDesc.height;
    params.pixelStride = size_t(numChannels) * bytesPerComponent;
    params.rowPitch = size_t(numChannels * textureSetDesc.width);
    params.channelFormat = ntc::PixelFormat::UNORM8;
    params.dstColorSpaces = colorSpaces;
    params.useDithering = true;

    ntc::Status ntcStatus = textureSet->ReadChannels(params);

    if (ntcStatus != ntc::Status::Ok)
        // Handle the error.
    
    // ... Do something with textureData ...

    free(textureData);
}
```

Note that the texture data can be read into a buffer with more channels in each pixel than `numChannels` passed to `ReadChannels`. In this case, correct values for `pixelStride` and `rowPitch` must be provided. The first `numChannels` in each pixel will contain the read values, and the rest will be set to default values (1.0 for alpha, 0.0 for others).
