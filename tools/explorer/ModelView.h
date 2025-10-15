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
#include <donut/engine/View.h>
#include <nvrhi/nvrhi.h>
#include <memory>
#include <vector>

#include <ntc-utils/Semantics.h>

namespace donut::engine
{
class CommonRenderPasses;
class DirectionalLight;
class SceneGraph;
class ShaderFactory;
class FramebufferFactory;
}

namespace donut::render
{
class SkyPass;
}

namespace donut::app
{
class ThirdPersonCamera;
}

class ModelView
{
public:
    ModelView(
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        nvrhi::IDevice* device);

    bool Init(nvrhi::FramebufferInfoEx const& framebufferInfo);
    void Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer);

    bool MousePosUpdate(double xpos, double ypos);
    bool MouseButtonUpdate(int button, int action, int mods);
    bool MouseScrollUpdate(double xoffset, double yoffset);

    void SetTexture(nvrhi::ITexture* texture, bool isSRGB, int slot, bool right);
    void SetDecompressedImagesAvailable(bool available);
    void SetSemanticBindings(const SemanticBinding* bindings, int count);
    void SetViewport(dm::float2 origin, dm::float2 size);
    void SetImageName(bool right, const std::string& name);
    void SetNumTextureMips(int mips);

    void Animate(float elapsedTimeSeconds);
    void BuildControlDialog();
    bool IsRequestingRestore(int& outRunOrdinal, bool& outRightTexture);

private:
    enum class DisplayMode
    {
        LeftTexture,
        RightTexture,
        SplitScreen
    };

    static constexpr uint32_t c_MaxTextures = 16;

    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::SceneGraph> m_sceneGraph;
    std::shared_ptr<donut::engine::DirectionalLight> m_light;
    std::shared_ptr<donut::app::ThirdPersonCamera> m_camera;
    std::shared_ptr<donut::render::SkyPass> m_skyPass;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;

    std::vector<SemanticBinding> m_semanticBindings;
    std::array<nvrhi::BindingSetItem, c_MaxTextures * 2> m_descriptors;
    std::array<bool, c_MaxTextures> m_convertFromSRGB{};
    bool m_descriptorsValid = false;
    donut::engine::PlanarView m_view;
    
    nvrhi::DeviceHandle m_device;
    nvrhi::ShaderHandle m_vertexShader;
    nvrhi::ShaderHandle m_pixelShader;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingLayoutHandle m_bindlessLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::GraphicsPipelineHandle m_graphicsPipeline;
    nvrhi::DescriptorTableHandle m_descriptorTable;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::SamplerHandle m_sampler;

    nvrhi::ShaderHandle m_overlayPixelShader;
    nvrhi::BindingLayoutHandle m_overlayBindingLayout;
    nvrhi::BindingSetHandle m_overlayBindingSet;
    nvrhi::GraphicsPipelineHandle m_overlayPipeline;
    
    nvrhi::TextureHandle m_depthBuffer;
    nvrhi::TextureHandle m_colorBuffer;

    dm::float4x4 m_projectionMatrix = dm::float4x4::identity();

    DisplayMode m_displayMode = DisplayMode::LeftTexture;
    int m_splitPosition = -1;
    bool m_decompressedImagesAvailable = false;
    int m_textureMips = 0;
    float m_showMipLevel = 0.f;
    bool m_moveSplit = false;
    dm::int2 m_mousePos = 0;
    bool m_moveLight = false;
    dm::int2 m_dragStart = 0;
    std::string m_leftImageName;
    std::string m_rightImageName;
    int m_requestingRestore = -1;
    bool m_requestingRestoreRight = false;
};

