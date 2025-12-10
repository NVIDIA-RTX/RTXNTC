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

#pragma once

#include <nvrhi/nvrhi.h>
#include <libntc/ntc.h>
#include <vector>

// This enum lists all versions of the uploading and decompression pipelines implemented in this subsystem.
enum class BufferLoadingPipeline
{
    None,

    // For buffers (only used for BC mode data provided right after CUDA compression):
    // - CPU: Copy uncompressed data from a CPU buffer into stagingBuffer
    // - GPU: Copy into finalBuffer
    DirectCopy,

    // For buffers:
    // - CPU: Read uncompressed data from file into stagingBuffer
    // - GPU: Copy into finalBuffer
    // For textures:
    // - CPU: Read uncompressed data from file into the uncompressedData vector
    // - CPU+GPU: Use writeTexture to upload data to the GPU, let NVRHI handle the staging
    ReadUncompressed,

    // For buffers:
    // - CPU: Read compressed data from file into the compressedData vector
    // - CPU: Decompress into stagingBuffer
    // - GPU: Copy into finalBuffer
    // For textures:
    // - CPU: Read compressed data from file into the compressedData vector
    // - CPU: Decompress into the uncompressedData vector
    // - CPU+GPU: Use writeTexture to upload data to the GPU, let NVRHI handle the staging
    DecompressOnCPU,

    // For buffers:
    // - CPU: Read headers from file into the compressedData vector
    // - CPU: Read compressed data (no headers) into stagingBuffer
    // - GPU: Copy compressed data into tempBuffer
    // - GPU: Decompress into the finalBuffer
    // For textures:
    // - CPU: Read compressed data from file into the compressedData vector
    // - CPU+GPU: Use writeBuffer to upload compressed data (no headers) into compressedBuffer
    // - GPU: Decompress into decompressedBuffer
    // - GPU: Copy into the final texture
    DecompressWithVk,

    // For both buffers and textures:
    // - CPU: Read compressed data into the compressedData vector
    //       (Note: DirectStorage can handle reading from files, too, but here it might be a memory buffer, not a file)
    // - CPU+GPU: Upload and decompress, let DirectStorage handle everything
    DecompressWithDStorage
};

struct GDeflateFeatures;

struct BufferLoadingTask
{
    BufferLoadingPipeline pipeline = BufferLoadingPipeline::None;
    void const* directCopySource = nullptr;
    size_t directCopySize = 0;
    ntc::BufferFootprint footprint;
    nvrhi::BufferRange stagingBufferRange;
    nvrhi::BufferRange tempBufferRange;
    nvrhi::BufferRange finalBufferRange;
    std::vector<uint8_t> compressedData;
    std::vector<uint8_t> uncompressedData;
    bool readIntoCpuBuffer = false;
    bool readIntoStagingBuffer = false;
};

struct TextureSubresourceLoadingTask
{
    BufferLoadingPipeline pipeline = BufferLoadingPipeline::None;
    ntc::LatentTextureFootprint footprint;
    nvrhi::TextureHandle destinationTexture;
    int mipLevel = 0;
    int layerIndex = 0;
    size_t gdeflateHeaderSize = 0;
    nvrhi::BufferRange compressedBufferRange;
    nvrhi::BufferRange decompressedBufferRange;
    std::vector<uint8_t> compressedData;
    std::vector<uint8_t> uncompressedData;
    bool readCompressedIntoCpuBuffer = false;
    bool readUncompressedIntoCpuBuffer = false;
};

void FillBufferLoadingTasksForBC(
    ntc::TextureSetDesc const& textureSetDesc,
    ntc::ITextureMetadata* textureMetadata,
    std::vector<BufferLoadingTask>& tasks,
    bool gpuDecompressionSupported,
    nvrhi::GraphicsAPI graphicsAPI,
    size_t& stagingBufferSize,
    size_t& tempBufferSize,
    size_t& finalBufferSize);

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
    size_t finalBufferSize);

void FillTextureLoadingTasksForLatents(
    ntc::ITextureSetMetadata* textureSetMetadata,
    nvrhi::ITexture* destinationTexture,
    int firstLatentMipLevel,
    std::vector<TextureSubresourceLoadingTask>& tasks,
    bool gpuDecompressionSupported,
    nvrhi::GraphicsAPI graphicsAPI,
    size_t& compressedBufferSize,
    size_t& decompressedBufferSize);

bool ExecuteTextureLoadingTasks(
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList,
    ntc::IContext* context,
    ntc::IStream* inputFile,
    GDeflateFeatures* gdeflateFeatures,
    std::vector<TextureSubresourceLoadingTask>& tasks,
    size_t compressedBufferSize,
    size_t decompressedBufferSize);

template<typename T>
class MappedBuffer
{
public:
    MappedBuffer(nvrhi::IDevice* device)
        : m_device(device)
    { }

    bool Map(nvrhi::IBuffer* buffer, nvrhi::CpuAccessMode mode)
    {
        Unmap();
        m_buffer = buffer;
        m_ptr = static_cast<T*>(m_device->mapBuffer(buffer, mode));
        return m_ptr != nullptr;
    }

    void Unmap()
    {
        if (m_ptr)
        {
            m_device->unmapBuffer(m_buffer);
            m_ptr = nullptr;
        }
        m_buffer = nullptr;
    }

    ~MappedBuffer()
    {
        Unmap();
    }

    T* Get() const
    {
        return m_ptr;
    }

private:
    T* m_ptr = nullptr;
    nvrhi::DeviceHandle m_device;
    nvrhi::BufferHandle m_buffer;
};
