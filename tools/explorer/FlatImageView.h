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

#pragma once

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>
#include <memory>

namespace donut::engine
{
class BindingCache;
class CommonRenderPasses;
class ShaderFactory;
}

class FlatImageView
{
public:
    FlatImageView(
        std::shared_ptr<donut::engine::BindingCache> bindingCache,
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        nvrhi::IDevice* device);

    bool Init(nvrhi::FramebufferInfo const& framebufferInfo);
    void Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer);
    void ReadPixel();

    bool MousePosUpdate(double xpos, double ypos);
    bool MouseButtonUpdate(int button, int action, int mods);
    bool MouseScrollUpdate(double xoffset, double yoffset);

    void Reset();
    void SetTextureSize(int width, int height, int mips);
    void SetTextures(nvrhi::ITexture* leftTexture, nvrhi::ITexture* rightTexture, int channels, bool sRGB);
    void SetViewport(dm::float2 origin, dm::float2 size);
    void SetImageName(bool right, const std::string& name);

    void BuildControlDialog();
    bool IsRequestingRestore(int& outRunOrdinal, bool& outRightTexture);

private:
    std::shared_ptr<donut::engine::BindingCache> m_bindingCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    nvrhi::TextureHandle m_leftTexture;
    nvrhi::TextureHandle m_rightTexture;
    int m_textureChannels = 0;
    bool m_textureSRGB = false;
    dm::float2 m_textureSize = 0.f;
    dm::float2 m_viewOrigin = 0.f;
    dm::float2 m_viewSize = 0.f;
    int m_numMips = 0;
    int m_mipLevel = 0;
    std::string m_leftImageName;
    std::string m_rightImageName;
    int m_requestingRestore = -1;
    bool m_requestingRestoreRight = false;

    dm::int2 m_textureCenterOffset = 0;
    float m_displayScale = 1.f;
    bool m_drag = false;
    bool m_moveSplit = false;
    int m_splitPosition = 0;
    dm::int2 m_dragStart = 0;
    dm::int2 m_mousePos = 0;
    uint32_t m_displayMode = 0; // see DisplayMode in FlatImageViewConstants.h
    uint32_t m_channelMask = 0xf;
    float m_colorScale = 1.f;
    bool m_applyToneMapping = false;

    bool m_enablePixelInspector = true;
    bool m_pixelValuesValid = false;
    dm::float4 m_leftPixelValue;
    dm::float4 m_rightPixelValue;

    nvrhi::DeviceHandle m_device;
    nvrhi::ShaderHandle m_pixelShader;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::GraphicsPipelineHandle m_graphicsPipeline;
    nvrhi::BufferHandle m_pixelBuffer;
    nvrhi::BufferHandle m_pixelStagingBuffer1;
    nvrhi::BufferHandle m_pixelStagingBuffer2;

    dm::float2 WindowPosToUv(dm::int2 windowPos) const;
    dm::int2 UvToWindowPos(dm::float2 uv) const;
    dm::ibox2 GetTexelBounds(dm::int2 windowPos) const;
    void FitImageToView();

    // Zoom in or out, maintaining the same location on the image that stable point (e.g. mouse cursor) is at
    void SetDisplayScaleStable(float newScale, dm::int2 stablePoint);
};

