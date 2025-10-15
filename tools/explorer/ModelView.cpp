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

#include "ModelView.h"
#include "ImGuiExtensions.h"
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/app/Camera.h>
#include <donut/render/SkyPass.h>
#include <nvrhi/utils.h>

#include <utility>

#if NTC_WITH_DX12
#include "compiled_shaders/ModelView_MainPS.dxil.h"
#include "compiled_shaders/ModelView_MainVS.dxil.h"
#include "compiled_shaders/ModelView_OverlayPS.dxil.h"
#endif
#if NTC_WITH_VULKAN
#include "compiled_shaders/ModelView_MainPS.spirv.h"
#include "compiled_shaders/ModelView_MainVS.spirv.h"
#include "compiled_shaders/ModelView_OverlayPS.spirv.h"
#endif

using namespace donut;
using namespace donut::math;

#include "ModelViewConstants.h"

constexpr float c_verticalFov = 60.f;

ModelView::ModelView(
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
    nvrhi::IDevice* device)
    : m_commonPasses(std::move(commonPasses))
    , m_shaderFactory(std::move(shaderFactory))
    , m_device(device)
{
    m_camera = std::make_shared<donut::app::ThirdPersonCamera>();
    m_camera->SetTargetPosition(0.f);
    m_camera->SetDistance(3.f);
    m_camera->SetRotation(radians(135.f), radians(30.f));
    m_camera->Animate(0.f);
    
    m_sceneGraph = std::make_shared<engine::SceneGraph>();
    auto rootNode = std::make_shared<engine::SceneGraphNode>();
    m_light = std::make_shared<engine::DirectionalLight>();
    rootNode->SetLeaf(m_light);
    m_sceneGraph->SetRootNode(rootNode);

    m_light->SetDirection(double3(-1.0, -1.0, -1.0));
    m_light->angularSize = 1.f;
    m_light->irradiance = 3.f;

    m_descriptors.fill(nvrhi::BindingSetItem::None());
}

bool ModelView::Init(nvrhi::FramebufferInfoEx const& framebufferInfo)
{
    if (!m_vertexShader)
    {        
        auto vertexShaderDesc = nvrhi::ShaderDesc()
            .setShaderType(nvrhi::ShaderType::Vertex)
            .setEntryName("MainVS");
        m_vertexShader = m_shaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_ModelView_MainVS), nullptr, vertexShaderDesc);
    }

    if (!m_pixelShader)
    {
        auto pixelShaderDesc = nvrhi::ShaderDesc()
            .setShaderType(nvrhi::ShaderType::Pixel)
            .setEntryName("MainPS");
        m_pixelShader = m_shaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_ModelView_MainPS), nullptr, pixelShaderDesc);
    }

    if (!m_overlayPixelShader)
    {
        auto pixelShaderDesc = nvrhi::ShaderDesc()
            .setShaderType(nvrhi::ShaderType::Pixel)
            .setEntryName("OverlayPS");
        m_overlayPixelShader = m_shaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_ModelView_OverlayPS), nullptr, pixelShaderDesc);
    }

    if (!m_vertexShader || !m_pixelShader || !m_overlayPixelShader)
        return false;

    if (m_depthBuffer)
    {
        const auto depthBufferDesc = m_depthBuffer->getDesc();
        if (depthBufferDesc.width != framebufferInfo.width || depthBufferDesc.height != framebufferInfo.height)
        {
            m_depthBuffer = nullptr;
            m_colorBuffer = nullptr;
        }
    }

    if (!m_depthBuffer || !m_colorBuffer)
    {
        auto textureDesc = nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(framebufferInfo.width)
            .setHeight(framebufferInfo.height)
            .setFormat(nvrhi::Format::D24S8)
            .setIsRenderTarget(true)
            .setDebugName("DepthBuffer")
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true);

        m_depthBuffer = m_device->createTexture(textureDesc);
        textureDesc
            .setFormat(nvrhi::Format::SRGBA8_UNORM)
            .setDebugName("ColorBuffer")
            .setInitialState(nvrhi::ResourceStates::RenderTarget);
        m_colorBuffer = m_device->createTexture(textureDesc);

        if (!m_depthBuffer || !m_colorBuffer)
            return false;

        m_framebufferFactory = std::make_shared<engine::FramebufferFactory>(m_device);
        m_framebufferFactory->RenderTargets = {m_colorBuffer};
        m_framebufferFactory->DepthTarget = m_depthBuffer;
        
        m_skyPass = std::make_shared<donut::render::SkyPass>(m_device, m_shaderFactory, m_commonPasses, m_framebufferFactory, m_view);
    }

    if (!m_overlayPipeline)
    {
        auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(OverlayConstants)));

        m_overlayBindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

        auto bindingSetDesc = nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(OverlayConstants)));

        m_overlayBindingSet = m_device->createBindingSet(bindingSetDesc, m_overlayBindingLayout);

        auto pipelineDesc = nvrhi::GraphicsPipelineDesc()
            .setPrimType(nvrhi::PrimitiveType::TriangleStrip)
            .setVertexShader(m_commonPasses->m_FullscreenVS)
            .setPixelShader(m_overlayPixelShader)
            .addBindingLayout(m_overlayBindingLayout)
            .setRenderState(nvrhi::RenderState()
                .setDepthStencilState(nvrhi::DepthStencilState()
                    .disableDepthTest()
                    .disableDepthWrite()));

        m_overlayPipeline = m_device->createGraphicsPipeline(pipelineDesc, framebufferInfo);
    }

    if (!m_overlayPipeline)
        return false;

    if (m_graphicsPipeline)
        return true;

    const uint32_t descriptorTableSize = c_MaxTextures * 2;
    auto bindlessLayoutDesc = nvrhi::BindlessLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Pixel)
        .setMaxCapacity(descriptorTableSize)
        .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(1));

    m_bindlessLayout = m_device->createBindlessLayout(bindlessLayoutDesc);

    m_descriptorTable = m_device->createDescriptorTable(m_bindlessLayout);
    m_device->resizeDescriptorTable(m_descriptorTable, descriptorTableSize, false);

    m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ModelViewConstants), "ModelViewConstants", engine::c_MaxRenderPassConstantBufferVersions));

    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
        .addItem(nvrhi::BindingLayoutItem::Sampler(0));

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

    auto samplerDesc = nvrhi::SamplerDesc()
        .setAllFilters(true)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
        .setMaxAnisotropy(16);
    m_sampler = m_device->createSampler(samplerDesc);

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_sampler));

    m_bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);
    
    auto renderState = nvrhi::RenderState()
        .setRasterState(nvrhi::RasterState()
            .setCullNone())
        .setDepthStencilState(nvrhi::DepthStencilState()
            .disableDepthTest()
            .disableDepthWrite())
        .setBlendState(nvrhi::BlendState()
            .setRenderTarget(0, nvrhi::BlendState::RenderTarget()
                .enableBlend()
                .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
                .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)));

    auto graphicsPipelineDesc = nvrhi::GraphicsPipelineDesc()
        .setPrimType(nvrhi::PrimitiveType::TriangleStrip)
        .setVertexShader(m_vertexShader)
        .setPixelShader(m_pixelShader)
        .addBindingLayout(m_bindingLayout)
        .addBindingLayout(m_bindlessLayout)
        .setRenderState(renderState);
    
    m_graphicsPipeline = m_device->createGraphicsPipeline(graphicsPipelineDesc, framebufferInfo);

    return true;
}

void ModelView::Animate(float elapsedTimeSeconds)
{
    m_camera->Animate(elapsedTimeSeconds);

    m_view.SetMatrices(m_camera->GetWorldToViewMatrix(), m_projectionMatrix);
    m_view.UpdateCache();

    m_camera->SetView(m_view);

    static uint32_t frameIndex = 0;
    m_sceneGraph->Refresh(++frameIndex);
}

void ModelView::Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer)
{
    if (!m_descriptorsValid)
    {
        // Need to make sure nothing is being executed while we update the descriptor table.
        // This is a rare event, so a WFI is fine.
        m_device->waitForIdle();

        // Write all descriptors, don't bother tracking individual ones - there's only a few.
        for (const auto& bindingSetItem : m_descriptors)
        {
            if (bindingSetItem.type != nvrhi::ResourceType::None)
               m_device->writeDescriptorTable(m_descriptorTable, bindingSetItem);
        }

        m_descriptorsValid = true;
    }

    // Draw the sky in the background
    commandList->clearDepthStencilTexture(m_depthBuffer, nvrhi::AllSubresources, true, 1.f, true, 0);
    render::SkyParameters skyParameters{};
    skyParameters.brightness = 0.5f;
    m_skyPass->Render(commandList, m_view, *m_light, skyParameters);
    m_commonPasses->BlitTexture(commandList, framebuffer, m_colorBuffer);

    // Fill the model rendering constants
    ModelViewConstants constants{};
    m_view.FillPlanarViewConstants(constants.view);
    m_light->FillLightConstants(constants.light);
    constants.mipLevel = m_showMipLevel;
    constants.skyColor = skyParameters.skyColor * skyParameters.brightness;
    constants.groundColor = skyParameters.groundColor * skyParameters.brightness;
    constants.decompressedTextureOffset = c_MaxTextures;
    // Setup the split-screen variables, reuse that logic to display the decompressed texture only by setting splitPosition to -1.
    constants.enableSplitScreen = m_displayMode == DisplayMode::SplitScreen || m_displayMode == DisplayMode::RightTexture;
    constants.splitPosition = m_displayMode == DisplayMode::SplitScreen ? m_splitPosition : -1;
    constants.convertFromSrgbMask = 0;
    for (uint32_t i = 0; i < c_MaxTextures; ++i)
    {
        if (m_convertFromSRGB[i])
            constants.convertFromSrgbMask |= 1 << i;
    }
    
    const int imageOffset = 0;
    constants.albedoTexture = -1;
    constants.alphaTexture = -1;
    constants.emissiveTexture = -1;
    constants.metalnessTexture = -1;
    constants.normalTexture = -1;
    constants.occlusionTexture = -1;
    constants.roughnessTexture = -1;

    for (const auto& semantic : m_semanticBindings)
    {
        switch(semantic.label)
        {
            case SemanticLabel::Albedo:
                constants.albedoTexture = semantic.imageIndex + imageOffset;
                constants.albedoChannel = semantic.firstChannel;
                break;
            case SemanticLabel::AlphaMask:
                constants.alphaTexture = semantic.imageIndex + imageOffset;
                constants.alphaChannel = semantic.firstChannel;
                break;
            case SemanticLabel::Displacement:
                // Can't really use it here...
                break;
            case SemanticLabel::Emissive:
                constants.emissiveTexture = semantic.imageIndex + imageOffset;
                constants.emissiveChannel = semantic.firstChannel;
                break;
            case SemanticLabel::Metalness:
                constants.metalnessTexture = semantic.imageIndex + imageOffset;
                constants.metalnessChannel = semantic.firstChannel;
                break;
            case SemanticLabel::Normal:
                constants.normalTexture = semantic.imageIndex + imageOffset;
                constants.normalChannel = semantic.firstChannel;
                break;
            case SemanticLabel::Occlusion:
                constants.occlusionTexture = semantic.imageIndex + imageOffset;
                constants.occlusionChannel = semantic.firstChannel; 
                break;
            case SemanticLabel::Roughness:
                constants.roughnessTexture = semantic.imageIndex + imageOffset;
                constants.roughnessChannel = semantic.firstChannel;
                break;
            default: ;
        }
    }

    // Draw the model
    commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

    auto state = nvrhi::GraphicsState()
        .setPipeline(m_graphicsPipeline)
        .addBindingSet(m_bindingSet)
        .addBindingSet(m_descriptorTable)
        .setFramebuffer(framebuffer)
        .setViewport(m_view.GetViewportState());

    commandList->setGraphicsState(state);

    commandList->draw(nvrhi::DrawArguments().setVertexCount(4));

    // Draw the overlay
    auto viewExtent = m_view.GetViewExtent();
    if (m_displayMode == DisplayMode::SplitScreen && m_splitPosition > viewExtent.minX && m_splitPosition < viewExtent.maxX)
    {
        auto overlayState = nvrhi::GraphicsState()
            .setPipeline(m_overlayPipeline)
            .addBindingSet(m_overlayBindingSet)
            .setFramebuffer(framebuffer)
            .setViewport(m_view.GetViewportState());

        // Currently, the overlay pass only draws a vertical bar - limit its scope of rendering to just that bar
        overlayState.viewport.scissorRects[0].minX = m_splitPosition - 1;
        overlayState.viewport.scissorRects[0].maxX = m_splitPosition + 1;

        commandList->setGraphicsState(overlayState);

        OverlayConstants overlayConstants{};
        overlayConstants.splitPosition = m_splitPosition;
        commandList->setPushConstants(&overlayConstants, sizeof(overlayConstants));

        commandList->draw(nvrhi::DrawArguments().setVertexCount(4));
    }
}

bool ModelView::MousePosUpdate(double xpos, double ypos)
{
    m_camera->MousePosUpdate(xpos, ypos);

    m_mousePos = int2(int(xpos), int(ypos));

    if (m_moveSplit)
    {
        m_splitPosition = m_mousePos.x;
    }
    else if (m_moveLight)
    {
        float3 direction = float3(m_light->GetDirection());

        float azimuth, elevation, distance;
        cartesianToSphericalDegrees(direction, azimuth, elevation, distance);

        // Drag the sun around, use the view FOV to derive the degrees per pixel ratio.
        // The 1.5 factor is used because without it, movement feels too slow.
        const float degreesPerPixel = 1.5f * c_verticalFov / float(m_view.GetViewExtent().height());
        azimuth += float(m_mousePos.x - m_dragStart.x) * degreesPerPixel;
        elevation += float(m_mousePos.y - m_dragStart.y) * degreesPerPixel;

        // Clamp the eleveation to avoid unstable azimuth values at the polar singularities.
        elevation = dm::clamp(elevation, -89.f, 89.f);

        direction = sphericalDegreesToCartesian(azimuth, elevation, distance);

        m_light->SetDirection(double3(direction));

        m_dragStart = m_mousePos;
    }

    return true;
}

bool ModelView::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_displayMode == DisplayMode::SplitScreen)
    {
        if (action == GLFW_PRESS && (button == GLFW_MOUSE_BUTTON_LEFT && mods == GLFW_MOD_SHIFT ||
                                     button == GLFW_MOUSE_BUTTON_RIGHT && mods == 0))
        {
            m_moveSplit = true;
            m_splitPosition = m_mousePos.x;
            return true;
        }

        if (action == GLFW_RELEASE && m_moveSplit)
        {
            m_moveSplit = false;
            return true;
        }
    }

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT && mods == GLFW_MOD_CONTROL)
    {
        m_moveLight = true;
        m_dragStart = m_mousePos;
        return true;
    }
    
    if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT && m_moveLight)
    {
        m_moveLight = false;
        return true;
    }

    m_camera->MouseButtonUpdate(button, action, mods);
    return true;
}

bool ModelView::MouseScrollUpdate(double xoffset, double yoffset)
{
    m_camera->MouseScrollUpdate(xoffset, yoffset);
    return true;
}

void ModelView::SetTexture(nvrhi::ITexture* texture, bool isSRGB, int slot, bool right)
{
    if (slot < 0 || slot >= int(c_MaxTextures))
    {
        assert(false);
        return;
    }

    nvrhi::Format viewFormat = texture->getDesc().format;
    m_convertFromSRGB[slot] = false;
    if (isSRGB)
    {
        if (viewFormat == nvrhi::Format::RGBA8_UNORM)
            viewFormat = nvrhi::Format::SRGBA8_UNORM;
        else
            m_convertFromSRGB[slot] = true;
    }

    uint32_t page = right ? c_MaxTextures : 0;
    auto bindingSetItem = nvrhi::BindingSetItem::Texture_SRV(uint32_t(slot) + page, texture, viewFormat);
    if (m_descriptors[bindingSetItem.slot] != bindingSetItem)
    {
        m_descriptorsValid = false;
        m_descriptors[bindingSetItem.slot] = bindingSetItem;
    }
}

void ModelView::SetNumTextureMips(int mips)
{
    m_textureMips = mips;
    m_showMipLevel = std::min(m_showMipLevel, float(m_textureMips - 1));
}

void ModelView::SetDecompressedImagesAvailable(bool available)
{
    if (available && !m_decompressedImagesAvailable)
        m_displayMode = DisplayMode::RightTexture;
    m_decompressedImagesAvailable = available;
}

void ModelView::SetSemanticBindings(const SemanticBinding* bindings, int count)
{
    m_semanticBindings.resize(count);
    if (count > 0)
        memcpy(m_semanticBindings.data(), bindings, count * sizeof(SemanticBinding));
}

void ModelView::SetViewport(dm::float2 origin, dm::float2 size)
{
    if (m_view.GetViewport().width() == 0)
        m_splitPosition = int(origin.x + size.x * 0.5f);

    m_view.SetViewport(nvrhi::Viewport(
        origin.x, origin.x + size.x,
        origin.y, origin.y + size.y,
        0.f, 1.f));

    m_view.UpdateCache();

    m_projectionMatrix = perspProjD3DStyle(radians(c_verticalFov), size.x / size.y, 0.01f, 100.f);
}

void ModelView::SetImageName(bool right, const std::string& name)
{
    if (right)
        m_rightImageName = name;
    else
        m_leftImageName = name;
}

void ModelView::BuildControlDialog()
{
    ImGuiIO const& io = ImGui::GetIO();
    float const fontSize = ImGui::GetFontSize();

    auto viewExtent = m_view.GetViewExtent();
    ImGui::SetNextWindowPos(ImVec2(
        float(viewExtent.minX + viewExtent.maxX) * 0.5f, 
        float(viewExtent.maxY) - fontSize * 0.6f),
        ImGuiCond_Always, ImVec2(0.5f, 1.0f));

    ImGui::Begin("Model View", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

    // Mip level slider
    if (m_textureMips > 1)
    {
        ImGui::PushItemWidth(120.f);
        ImGui::SliderFloat("##MipLevel", &m_showMipLevel, 0.f, float(m_textureMips) - 1.f, "Mip %.1f");
        ImGui::PopItemWidth();
    }
    else
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("(No Mips)");
    }
    
    // Display mode selection
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.f, 3.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 1.f));

    if (!m_decompressedImagesAvailable)
        m_displayMode = DisplayMode::LeftTexture;

    std::tuple<DisplayMode, const char*> modes[] = {
        { DisplayMode::LeftTexture, m_leftImageName.c_str() },
        { DisplayMode::RightTexture, m_rightImageName.c_str() },
        { DisplayMode::SplitScreen, "Split-Screen" }
    };

    bool first = true;
    for (const auto& [mode, label] : modes)
    {
        ImGui::SameLine(first ? fontSize * 8.5f : 0.f);
        first = false;

        // Use an ID string with ### to make ImGui element ID independent from the button label which is volatile
        const std::string id = std::string(label) + "###" + std::to_string(int(mode));

        bool active = m_displayMode == mode;
        ImGui::BeginDisabled(!m_decompressedImagesAvailable);
        ImGui::ToggleButton(id.c_str(), &active, ImVec2(fontSize * 6.5f, 0));
        ImGui::EndDisabled();
        if (active) m_displayMode = mode;

        if ((mode == DisplayMode::LeftTexture || mode == DisplayMode::RightTexture) && ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CompressionRun"))
            {
                if (payload->Data && payload->DataSize == sizeof(int))
                {
                    m_requestingRestore = *static_cast<int*>(payload->Data);
                    m_requestingRestoreRight = mode == DisplayMode::RightTexture;
                }
            }

            ImGui::EndDragDropTarget();
        }
    }
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    ImGui::End();
}

bool ModelView::IsRequestingRestore(int& outRunOrdinal, bool& outRightTexture)
{
    outRunOrdinal = m_requestingRestore;
    outRightTexture = m_requestingRestoreRight;
    m_requestingRestore = -1;
    return outRunOrdinal >= 0;
}

