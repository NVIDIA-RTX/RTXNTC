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

#include <ntc-utils/BufferLoading.h>
#include <ntc-utils/DeviceUtils.h>
#include <donut/core/log.h>
#if NTC_WITH_VULKAN
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#endif

using namespace donut;

static size_t RoundUp4(size_t size)
{
    return (size + 3) & ~3ull;
}

static void AppendBufferRange(nvrhi::BufferRange& inoutRange, size_t& totalSize, size_t appendSize)
{
    inoutRange.byteOffset = totalSize;
    inoutRange.byteSize = appendSize;
    totalSize += RoundUp4(appendSize);
}

void FillBufferLoadingTasksForBC(
    ntc::TextureSetDesc const& textureSetDesc,
    ntc::ITextureMetadata* textureMetadata,
    std::vector<BufferLoadingTask>& tasks,
    bool gpuDecompressionSupported,
    nvrhi::GraphicsAPI graphicsAPI,
    size_t& stagingBufferSize,
    size_t& tempBufferSize,
    size_t& finalBufferSize)
{
    tasks.clear();
    tasks.resize(textureSetDesc.mips);

    stagingBufferSize = 0;
    tempBufferSize = 0;
    finalBufferSize = 0;
    
    bool const enableVkDecompression = gpuDecompressionSupported && graphicsAPI == nvrhi::GraphicsAPI::VULKAN;
    bool const enableDStorage = gpuDecompressionSupported && graphicsAPI == nvrhi::GraphicsAPI::D3D12;

    for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
    {
        BufferLoadingTask& task = tasks[mipLevel];

        textureMetadata->GetBC7ModeBuffer(mipLevel, &task.directCopySource, &task.directCopySize);
        if (task.directCopySize)
        {
            task.pipeline = BufferLoadingPipeline::DirectCopy;
            AppendBufferRange(task.stagingBufferRange, stagingBufferSize, task.directCopySize);
            AppendBufferRange(task.finalBufferRange, finalBufferSize, task.directCopySize);
            continue;
        }

        task.footprint = textureMetadata->GetBC7ModeBufferFootprint(mipLevel);
        if (task.footprint.uncompressedSize == 0)
            continue;
        
        if (task.footprint.compressionType == ntc::CompressionType::None)
        {
            task.pipeline = BufferLoadingPipeline::ReadUncompressed;
            AppendBufferRange(task.stagingBufferRange, stagingBufferSize, task.footprint.rangeInStream.size);
            task.readIntoStagingBuffer = true;
        }
        else if (task.footprint.compressionType == ntc::CompressionType::GDeflate)
        {
            if (enableVkDecompression)
            {
                task.pipeline = BufferLoadingPipeline::DecompressWithVk;
                
                // GDeflate header is read into a CPU buffer
                size_t const headerSize = ntc::GetGDeflateHeaderSize(task.footprint.uncompressedSize);
                task.compressedData.resize(headerSize);
                task.readIntoCpuBuffer = true;

                // Actual compressed data is read directly into the staging buffer
                assert(task.footprint.rangeInStream.size > headerSize);
                size_t const uploadSize = task.footprint.rangeInStream.size - headerSize;
                AppendBufferRange(task.stagingBufferRange, stagingBufferSize, uploadSize);
                AppendBufferRange(task.tempBufferRange, tempBufferSize, uploadSize);
                task.readIntoStagingBuffer = true;
            }
            else if (enableDStorage)
            {
                task.pipeline = BufferLoadingPipeline::DecompressWithDStorage;
                task.compressedData.resize(task.footprint.rangeInStream.size);
                task.readIntoCpuBuffer = true;
            }
            else
            {
                task.pipeline = BufferLoadingPipeline::DecompressOnCPU;
                task.compressedData.resize(task.footprint.rangeInStream.size);
                task.uncompressedData.resize(task.footprint.uncompressedSize);
                AppendBufferRange(task.stagingBufferRange, stagingBufferSize, task.footprint.uncompressedSize);
                task.readIntoCpuBuffer = true;
            }
        }
        else
        {
            assert(!"Unsupported compression type!");
        }

        AppendBufferRange(task.finalBufferRange, finalBufferSize, task.footprint.uncompressedSize);
    }
}

#if NTC_WITH_DX12
void UploadAndDecompressBufferWithDStorage(IDStorageQueue* dstorageQueue,
    std::vector<uint8_t>& compressedData,
    nvrhi::IBuffer* decompressedBuffer,
    nvrhi::BufferRange decompressedRange)
{
    assert(dstorageQueue); // This function is not called otherwise
    
    // Fill out and submit the request.
    DSTORAGE_REQUEST request{};
    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    request.Source.Memory.Source = compressedData.data();
    request.Source.Memory.Size = compressedData.size();
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Destination.Buffer.Resource = decompressedBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
    request.Destination.Buffer.Offset = decompressedRange.byteOffset;
    request.Destination.Buffer.Size = decompressedRange.byteSize;
    request.UncompressedSize = decompressedRange.byteSize;
    dstorageQueue->EnqueueRequest(&request);
}

void UploadAndDecompressTextureWithDStorage(IDStorageQueue* dstorageQueue,
    std::vector<uint8_t>& compressedData,
    nvrhi::ITexture* destinationTexture,
    int mipLevel,
    int layerIndex,
    size_t uncompressedSize)
{
    assert(dstorageQueue); // This function is not called otherwise

    nvrhi::TextureDesc const& textureDesc = destinationTexture->getDesc();
    uint32_t const mipWidth = std::max(textureDesc.width >> mipLevel, 1u);
    uint32_t const mipHeight = std::max(textureDesc.height >> mipLevel, 1u);
    
    // Fill out and submit the request.
    DSTORAGE_REQUEST request{};
    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    request.Source.Memory.Source = compressedData.data();
    request.Source.Memory.Size = compressedData.size();
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION;
    request.Destination.Texture.Resource = destinationTexture->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
    request.Destination.Texture.SubresourceIndex = uint32_t(mipLevel) + uint32_t(layerIndex) * textureDesc.mipLevels;
    request.Destination.Texture.Region.right = mipWidth;
    request.Destination.Texture.Region.bottom = mipHeight;
    request.Destination.Texture.Region.back = 1;
    request.UncompressedSize = uncompressedSize;
    dstorageQueue->EnqueueRequest(&request);
}
#endif

bool DecompressWithVulkanExtension(nvrhi::IDevice* device, nvrhi::ICommandList* commandList,
    ntc::IContext* context,
    void const* compressedHeader,
    size_t const compressedHeaderSize,
    size_t const uncompressedSize,
    nvrhi::IBuffer* compressedBuffer,
    size_t const compressedOffset,
    nvrhi::IBuffer* decompressedBuffer,
    size_t const decompressedOffset)
{
#if NTC_WITH_VULKAN
    void* vkCommandList = commandList->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
    if (!vkCommandList)
        return false;
    
    commandList->setBufferState(compressedBuffer, nvrhi::ResourceStates::ShaderResource);
    commandList->setBufferState(decompressedBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();

    ntc::Status ntcStatus = context->DecompressGDeflateOnVulkanGPU(vkCommandList,
        compressedHeader, compressedHeaderSize,
        compressedBuffer->getGpuVirtualAddress() + compressedOffset,
        decompressedBuffer->getGpuVirtualAddress() + decompressedOffset);

    if (ntcStatus != ntc::Status::Ok)
    {
        fprintf(stderr, "Call to DecompressGDeflateOnVulkanGPU failed, error code = %s: %s\n",
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }
    
    return true;
#else
    return false;
#endif
}

bool CopyBufferToTextureVulkan(nvrhi::ICommandList* commandList,
    nvrhi::IBuffer* srcBuffer,
    size_t srcOffset,
    ntc::LatentTextureFootprint const& footprint,
    nvrhi::ITexture* dstTexture,
    int mipLevel,
    int arrayLayer)
{
#if NTC_WITH_VULKAN
    // Use a buffer-to-image copy with raw Vulkan API beacuse there is no way to do that through NVRHI.
    // The closest feature a copyTexture function taking a StagingTexture, but it requires an actual staging texture,
    // which we don't need here (no CPU access necessary), and NVRHI's StagingTexture doesn't provide access
    // to the underlying buffer and its subresource placement.

    vk::BufferImageCopy region = vk::BufferImageCopy()
        .setBufferOffset(srcOffset)
        .setImageExtent(vk::Extent3D()
            .setWidth(footprint.width)
            .setHeight(footprint.height)
            .setDepth(1))
        .setImageSubresource(vk::ImageSubresourceLayers()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(arrayLayer)
            .setMipLevel(mipLevel)
            .setLayerCount(1));

    vk::CommandBuffer vkCmdBuf = vk::CommandBuffer(commandList->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer));
    assert(vkCmdBuf);
    vk::Buffer vkSrcBuffer = vk::Buffer(srcBuffer->getNativeObject(nvrhi::ObjectTypes::VK_Buffer));
    assert(vkSrcBuffer);
    vk::Image vkDstImage = vk::Image(dstTexture->getNativeObject(nvrhi::ObjectTypes::VK_Image));
    assert(vkDstImage);

    // Note: the image is already in the TransferDstOptimal layout, because we manage latent texture states
    // explicitly, and they're transitioned to the CopyDest state in ExecuteTextureLoadingTasks(...)
    vkCmdBuf.copyBufferToImage(vkSrcBuffer, vkDstImage, vk::ImageLayout::eTransferDstOptimal, 1, &region);
    
    return true;
#else
    return false;
#endif
}

bool ExecuteBufferLoadingTasks(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    ntc::IContext* context,
    ntc::IStream* inputFile,
    GDeflateFeatures* gdeflateFeatures,
    std::vector<BufferLoadingTask>& tasks,
    nvrhi::BufferHandle& finalBuffer,
    size_t stagingBufferSize,
    size_t tempBufferSize,
    size_t finalBufferSize)
{
    nvrhi::BufferHandle stagingBuffer;
    MappedBuffer<uint8_t> mappedStagingBuffer(device);

    nvrhi::BufferHandle tempBuffer;

    if (stagingBufferSize)
    {
        nvrhi::BufferDesc stagingBufferDesc = nvrhi::BufferDesc()
            .setByteSize(stagingBufferSize)
            .setDebugName("BC7 Mode Staging Buffer")
            .setCpuAccess(nvrhi::CpuAccessMode::Write);
        stagingBuffer = device->createBuffer(stagingBufferDesc);
        if (!stagingBuffer)
            return false;
        if (!mappedStagingBuffer.Map(stagingBuffer, nvrhi::CpuAccessMode::Write))
            return false;
    }

    if (tempBufferSize)
    {
        nvrhi::BufferDesc tempBufferDesc = nvrhi::BufferDesc()
            .setByteSize(tempBufferSize)
            .setDebugName("BC7 Mode Temp Buffer")
            .setCanHaveRawViews(true)
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest);
        tempBuffer = device->createBuffer(tempBufferDesc);
        if (!tempBuffer)
            return false;
    }

    if (finalBufferSize)
    {
        nvrhi::BufferDesc finalBufferDesc = nvrhi::BufferDesc()
            .setByteSize(finalBufferSize)
            .setDebugName("BC7 Mode Buffer")
            .setCanHaveRawViews(true)
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
        finalBuffer = device->createBuffer(finalBufferDesc);
        if (!finalBuffer)
            return false;
    }

    bool anyDStorageTasks = false;
    commandList->open();
    for (BufferLoadingTask& task : tasks)
    {
        if (task.pipeline == BufferLoadingPipeline::None)
            continue; // Nothing to do

        if (task.pipeline == BufferLoadingPipeline::DirectCopy)
        {
            assert(task.directCopySource);
            memcpy(mappedStagingBuffer.Get() + task.stagingBufferRange.byteOffset,
                task.directCopySource, task.directCopySize);
            commandList->copyBuffer(finalBuffer, task.finalBufferRange.byteOffset,
                stagingBuffer, task.stagingBufferRange.byteOffset, task.directCopySize);
            continue; // Task completed
        }
        
        assert(inputFile);
        bool readSuccessful = inputFile->Seek(task.footprint.rangeInStream.offset);
        
        // Read the data into the CPU buffer (task.compressedData), the staging buffer,
        // or both in a split mode: headers go into the CPU buffer, payload goes into the staging buffer.
        uint64_t totalBytesRead = 0;
        if (readSuccessful && task.readIntoCpuBuffer)
        {
            assert(task.compressedData.size() != 0);
            readSuccessful = inputFile->Read(task.compressedData.data(), task.compressedData.size());
            totalBytesRead += task.compressedData.size();
        }
        if (readSuccessful && task.readIntoStagingBuffer)
        {
            assert(task.stagingBufferRange.byteSize != 0);
            readSuccessful = inputFile->Read(mappedStagingBuffer.Get() + task.stagingBufferRange.byteOffset,
                task.stagingBufferRange.byteSize);
            totalBytesRead += task.stagingBufferRange.byteSize;
        }
        assert(totalBytesRead == task.footprint.rangeInStream.size);

        if (!readSuccessful)
        {
            fprintf(stderr, "Failed to read BC7 data from file (%zu bytes at offset %zu)\n",
                task.footprint.rangeInStream.size, task.footprint.rangeInStream.offset);
            task.pipeline = BufferLoadingPipeline::None;
            continue; // Task failed
        }

        switch(task.pipeline)
        {
            case BufferLoadingPipeline::ReadUncompressed:
                commandList->copyBuffer(finalBuffer, task.finalBufferRange.byteOffset,
                    stagingBuffer, task.stagingBufferRange.byteOffset, task.footprint.rangeInStream.size);
                break;

            case BufferLoadingPipeline::DecompressOnCPU: {
                ntc::Status ntcStatus = context->DecompressBuffer(
                    ntc::CompressionType::GDeflate,
                    task.compressedData.data(),
                    task.footprint.rangeInStream.size,
                    task.uncompressedData.data(),
                    task.stagingBufferRange.byteSize,
                    task.footprint.uncompressedCrc32);

                if (ntcStatus != ntc::Status::Ok)
                {
                    fprintf(stderr, "Failed to decompress BC7 data, error code = %s: %s\n",
                        ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                    task.pipeline = BufferLoadingPipeline::None;
                    break; // Task failed
                }

                assert(task.uncompressedData.size() == task.stagingBufferRange.byteSize);
                memcpy(mappedStagingBuffer.Get() + task.stagingBufferRange.byteOffset,
                    task.uncompressedData.data(), task.uncompressedData.size());

                commandList->copyBuffer(finalBuffer, task.finalBufferRange.byteOffset,
                    stagingBuffer, task.stagingBufferRange.byteOffset, task.footprint.uncompressedSize);
                break;
            }

            case BufferLoadingPipeline::DecompressWithVk:
                assert(task.tempBufferRange.byteSize == task.stagingBufferRange.byteSize);
                commandList->copyBuffer(tempBuffer, task.tempBufferRange.byteOffset, stagingBuffer,
                    task.stagingBufferRange.byteOffset, task.stagingBufferRange.byteSize);

                DecompressWithVulkanExtension(device, commandList,
                    context,
                    task.compressedData.data(),
                    task.compressedData.size(),
                    task.footprint.uncompressedSize,
                    tempBuffer,
                    task.tempBufferRange.byteOffset,
                    finalBuffer,
                    task.finalBufferRange.byteOffset);
                break;

            case BufferLoadingPipeline::DecompressWithDStorage: {
                // DStorage decompression is done later, after this cmdlist is executed
                anyDStorageTasks = true;
                break;
            }
            default:
                assert(!"Unknown BufferLoadingPipeline value!");
        }
    }
    commandList->close();
    device->executeCommandList(commandList);

#if NTC_WITH_DX12
    if (anyDStorageTasks)
    {
        device->waitForIdle();
        
        assert(gdeflateFeatures);

        for (BufferLoadingTask& task : tasks)
        {
            if (task.pipeline == BufferLoadingPipeline::DecompressWithDStorage)
            {
                UploadAndDecompressBufferWithDStorage(gdeflateFeatures->dstorageQueue, task.compressedData, finalBuffer,
                    task.finalBufferRange);
            }
        }
        
        // Do a complete sync with the DStorage queue on the CPU.
        // Normally, apps should synchronize the DStorage queue with the DX12 queues, but here we don't have
        // realtime constraints. Also, we would have to extend the lifetime of the DStorage input buffers until
        // they're completely consumed, which requires additional tracking...
        assert(gdeflateFeatures->dstorageEvent);
        gdeflateFeatures->dstorageQueue->EnqueueSetEvent(gdeflateFeatures->dstorageEvent);
        gdeflateFeatures->dstorageQueue->Submit();
        
        WaitForSingleObject(gdeflateFeatures->dstorageEvent, INFINITE);
    }
#endif
    return true;
}

void FillTextureLoadingTasksForLatents(
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::ITexture* destinationTexture,
    int firstLatentMipLevel,
    std::vector<TextureSubresourceLoadingTask>& tasks,
    bool gpuDecompressionSupported,
    nvrhi::GraphicsAPI graphicsAPI,
    size_t& compressedBufferSize,
    size_t& decompressedBufferSize)
{
    ntc::LatentTextureDesc const latentTextureDesc = textureSetMetadata->GetLatentTextureDesc();

    compressedBufferSize = 0;
    decompressedBufferSize = 0;

    bool const enableVkDecompression = gpuDecompressionSupported && graphicsAPI == nvrhi::GraphicsAPI::VULKAN;
    bool const enableDStorage = gpuDecompressionSupported && graphicsAPI == nvrhi::GraphicsAPI::D3D12;

    // Make sure that the destination texture does *not* use automatic state tracking.
    // Reason: DirectStorage uploads require that the texture is in the CopyDest state between the command lists.
    // When automatic state tracking is used (keepInitialState == true), NVRHI will always transition the resource
    // to the initial state at the end of the command list. So, in order to make DirectStorage work with auto tracking,
    // we'd have to use the CopyDest as the initial and default state for all latent textures. But that's not optimal
    // during render time, when these textures need to be used as shader resources.
    // So, we disable automatic state tracking, manually manage the CopyDest state during uploads, and then
    // do a permanent transition to ShaderResource once the uploads are finished.
    assert(destinationTexture->getDesc().keepInitialState == false);

    for (int mipLevel = firstLatentMipLevel; mipLevel < latentTextureDesc.mipLevels; ++mipLevel)
    {
        for (int layerIndex = 0; layerIndex < latentTextureDesc.arraySize; ++layerIndex)
        {
            TextureSubresourceLoadingTask& task = tasks.emplace_back();
            task.mipLevel = mipLevel;
            task.layerIndex = layerIndex;
            task.destinationTexture = destinationTexture;

            ntc::Status ntcStatus = textureSetMetadata->GetLatentTextureFootprint(mipLevel, layerIndex, task.footprint);
            if (ntcStatus != ntc::Status::Ok)
                continue;

            if (task.footprint.buffer.compressionType == ntc::CompressionType::None)
            {
                task.pipeline = BufferLoadingPipeline::ReadUncompressed;
                task.uncompressedData.resize(task.footprint.buffer.rangeInStream.size);
                task.readUncompressedIntoCpuBuffer = true;
            }
            else if (task.footprint.buffer.compressionType == ntc::CompressionType::GDeflate)
            {
                if (enableVkDecompression)
                {
                    task.pipeline = BufferLoadingPipeline::DecompressWithVk;
                    task.gdeflateHeaderSize = ntc::GetGDeflateHeaderSize(task.footprint.buffer.uncompressedSize);
                    task.compressedData.resize(task.footprint.buffer.rangeInStream.size);
                    AppendBufferRange(task.compressedBufferRange, compressedBufferSize,
                        task.footprint.buffer.rangeInStream.size - task.gdeflateHeaderSize);
                    AppendBufferRange(task.decompressedBufferRange, decompressedBufferSize,
                        task.footprint.buffer.uncompressedSize);
                    task.readCompressedIntoCpuBuffer = true;
                }
                else if (enableDStorage)
                {
                    task.pipeline = BufferLoadingPipeline::DecompressWithDStorage;
                    task.compressedData.resize(task.footprint.buffer.rangeInStream.size);
                    task.readCompressedIntoCpuBuffer = true;
                }
                else
                {
                    task.pipeline = BufferLoadingPipeline::DecompressOnCPU;
                    task.compressedData.resize(task.footprint.buffer.rangeInStream.size);
                    task.uncompressedData.resize(task.footprint.buffer.uncompressedSize);
                    task.readCompressedIntoCpuBuffer = true;
                }
            }
        }
    }
}

bool ExecuteTextureLoadingTasks(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    ntc::IContext* context,
    ntc::IStream* inputFile,
    GDeflateFeatures* gdeflateFeatures,
    std::vector<TextureSubresourceLoadingTask>& tasks,
    size_t compressedBufferSize,
    size_t decompressedBufferSize)
{
    nvrhi::BufferHandle compressedBuffer;
    nvrhi::BufferHandle decompressedBuffer;

    if (compressedBufferSize)
    {
        nvrhi::BufferDesc compressedBufferDesc = nvrhi::BufferDesc()
            .setByteSize(compressedBufferSize)
            .setDebugName("Compressed Latents Buffer")
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest);
        compressedBuffer = device->createBuffer(compressedBufferDesc);
        if (!compressedBuffer)
            return false;
    }

    if (decompressedBufferSize)
    {
        nvrhi::BufferDesc decompressedBufferDesc = nvrhi::BufferDesc()
            .setByteSize(decompressedBufferSize)
            .setDebugName("Decompressed Latents Buffer")
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest);
        decompressedBuffer = device->createBuffer(decompressedBufferDesc);
        if (!decompressedBuffer)
            return false;
    }

    bool anyDStorageTasks = false;
    nvrhi::ITexture* lastTexture = nullptr;

    commandList->open();
    for (TextureSubresourceLoadingTask& task : tasks)
    {
        if (task.pipeline == BufferLoadingPipeline::None)
            continue; // Nothing to do
            
        assert(inputFile);
        bool readSuccessful = inputFile->Seek(task.footprint.buffer.rangeInStream.offset);
        
        // Read the data into one of the CPU buffers (task.compressedData or task.uncompressedBuffer)
        uint64_t totalBytesRead = 0;
        if (readSuccessful && task.readCompressedIntoCpuBuffer)
        {
            assert(!task.compressedData.empty());
            readSuccessful = inputFile->Read(task.compressedData.data(), task.compressedData.size());
            totalBytesRead += task.compressedData.size();
        }
        if (readSuccessful && task.readUncompressedIntoCpuBuffer)
        {
            assert(!task.uncompressedData.empty());
            readSuccessful = inputFile->Read(task.uncompressedData.data(), task.uncompressedData.size());
            totalBytesRead += task.uncompressedData.size();
        }
        assert(totalBytesRead == task.footprint.buffer.rangeInStream.size);
        
        if (!readSuccessful)
        {
            log::warning("Failed to read latent data from file (%zu bytes at offset %zu)\n",
                task.footprint.buffer.rangeInStream.size, task.footprint.buffer.rangeInStream.offset);
            task.pipeline = BufferLoadingPipeline::None;
            continue; // Task failed
        }

        // Transition all texture subresources to the CopyDest state ahead of time.
        // Assume that tasks are grouped by texture, and work on all subresources of each texture once.
        if (lastTexture != task.destinationTexture)
        {
            commandList->beginTrackingTextureState(task.destinationTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::Common);
            commandList->setTextureState(task.destinationTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
            commandList->commitBarriers();
            lastTexture = task.destinationTexture;
        }

        switch(task.pipeline)
        {
            case BufferLoadingPipeline::ReadUncompressed:
                commandList->writeTexture(task.destinationTexture, task.layerIndex, task.mipLevel,
                    task.uncompressedData.data(), task.footprint.rowPitch);
                break;

            case BufferLoadingPipeline::DecompressOnCPU: {
                ntc::Status ntcStatus = context->DecompressBuffer(
                    task.footprint.buffer.compressionType,
                    task.compressedData.data(),
                    task.compressedData.size(),
                    task.uncompressedData.data(),
                    task.uncompressedData.size(),
                    task.footprint.buffer.uncompressedCrc32);

                if (ntcStatus != ntc::Status::Ok)
                {
                    fprintf(stderr, "Failed to decompress latent data, error code = %s: %s\n",
                        ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                    task.pipeline = BufferLoadingPipeline::None;
                    break; // Task failed
                }

                commandList->writeTexture(task.destinationTexture, task.layerIndex, task.mipLevel,
                    task.uncompressedData.data(), task.footprint.rowPitch);
                break;
            }

            case BufferLoadingPipeline::DecompressWithVk:
                assert(task.compressedData.size() == task.gdeflateHeaderSize + task.compressedBufferRange.byteSize);
                commandList->writeBuffer(compressedBuffer,
                    task.compressedData.data() + task.gdeflateHeaderSize,
                    task.compressedBufferRange.byteSize,
                    task.compressedBufferRange.byteOffset);
                
                DecompressWithVulkanExtension(device, commandList,
                    context,
                    task.compressedData.data(),
                    task.gdeflateHeaderSize,
                    task.footprint.buffer.uncompressedSize,
                    compressedBuffer,
                    task.compressedBufferRange.byteOffset,
                    decompressedBuffer,
                    task.decompressedBufferRange.byteOffset);

                CopyBufferToTextureVulkan(commandList,
                    decompressedBuffer, task.decompressedBufferRange.byteOffset,
                    task.footprint, task.destinationTexture, task.mipLevel, task.layerIndex);
                break;

            case BufferLoadingPipeline::DecompressWithDStorage: {
                // DStorage decompression is done later, after this cmdlist is executed
                anyDStorageTasks = true;
                break;
            }
            default:
                assert(!"Unknown BufferLoadingPipeline value!");
        }
        
    }
    
    commandList->close();
    device->executeCommandList(commandList);

#if NTC_WITH_DX12
    if (anyDStorageTasks)
    {
        device->waitForIdle();
        
        assert(gdeflateFeatures);

        for (TextureSubresourceLoadingTask& task : tasks)
        {
            if (task.pipeline == BufferLoadingPipeline::DecompressWithDStorage)
            {
                UploadAndDecompressTextureWithDStorage(gdeflateFeatures->dstorageQueue, task.compressedData,
                    task.destinationTexture, task.mipLevel, task.layerIndex, task.footprint.buffer.uncompressedSize);
            }
        }
        
        // Do a complete sync with the DStorage queue on the CPU.
        assert(gdeflateFeatures->dstorageEvent);
        gdeflateFeatures->dstorageQueue->EnqueueSetEvent(gdeflateFeatures->dstorageEvent);
        gdeflateFeatures->dstorageQueue->Submit();
        
        WaitForSingleObject(gdeflateFeatures->dstorageEvent, INFINITE);
    }
#endif

    // Transition all latent textures to the ShaderResource state, permanently.
    // This has to be done after the DirectStorage queue completes.
    lastTexture = nullptr;
    commandList->open();
    for (TextureSubresourceLoadingTask& task : tasks)
    {
        if (lastTexture != task.destinationTexture)
        {
            commandList->beginTrackingTextureState(task.destinationTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
            commandList->setPermanentTextureState(task.destinationTexture, nvrhi::ResourceStates::ShaderResource);
            lastTexture = task.destinationTexture;
        }
    }
    commandList->commitBarriers();
    commandList->close();
    device->executeCommandList(commandList);

    return true;
}
