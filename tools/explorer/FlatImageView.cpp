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

#include "FlatImageView.h"
#include "ImGuiExtensions.h"
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/BindingCache.h>
#include <utility>

#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>

#if NTC_WITH_DX12
#include "compiled_shaders/FlatImageView_MainPS.dxil.h"
#endif
#if NTC_WITH_VULKAN
#include "compiled_shaders/FlatImageView_MainPS.spirv.h"
#endif

using namespace donut;
using namespace donut::math;

#include "FlatImageViewConstants.h"

FlatImageView::FlatImageView(
    std::shared_ptr<donut::engine::BindingCache> bindingCache,
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
    nvrhi::IDevice* device)
    : m_bindingCache(std::move(bindingCache))
    , m_commonPasses(std::move(commonPasses))
    , m_shaderFactory(std::move(shaderFactory))
    , m_device(device)
{
}

bool FlatImageView::Init(nvrhi::FramebufferInfo const& framebufferInfo)
{
    if (m_graphicsPipeline)
        return true;

    auto pixelShaderDesc = nvrhi::ShaderDesc()
        .setShaderType(nvrhi::ShaderType::Pixel)
        .setEntryName("MainPS");

    m_pixelShader = m_shaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_FlatImageView_MainPS),
        nullptr, pixelShaderDesc);

    if (!m_pixelShader)
        return false;

    auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Pixel)
        .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FlatImageViewConstants)))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
        .addItem(nvrhi::BindingLayoutItem::TypedBuffer_UAV(0))
        .addItem(nvrhi::BindingLayoutItem::Sampler(0));

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

    auto renderState = nvrhi::RenderState()
        .setDepthStencilState(nvrhi::DepthStencilState()
            .disableDepthTest()
            .disableDepthWrite());

    auto graphicsPipelineDesc = nvrhi::GraphicsPipelineDesc()
        .setPrimType(nvrhi::PrimitiveType::TriangleStrip)
        .setVertexShader(m_commonPasses->m_FullscreenVS)
        .setPixelShader(m_pixelShader)
        .addBindingLayout(m_bindingLayout)
        .setRenderState(renderState);

    m_graphicsPipeline = m_device->createGraphicsPipeline(graphicsPipelineDesc, framebufferInfo);

    if (!m_graphicsPipeline)
        return false;

    auto bufferDesc = nvrhi::BufferDesc()
        .setDebugName("Pixel Buffer")
        .setByteSize(8 * sizeof(float))
        .setFormat(nvrhi::Format::RGBA32_FLOAT)
        .setCanHaveTypedViews(true)
        .setCanHaveUAVs(true)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
        .setKeepInitialState(true);
    m_pixelBuffer = m_device->createBuffer(bufferDesc);

    if (!m_pixelBuffer)
        return false;

    auto stagingBufferDesc = nvrhi::BufferDesc()
        .setDebugName("Pixel Buffer Staging")
        .setByteSize(bufferDesc.byteSize)
        .setCpuAccess(nvrhi::CpuAccessMode::Read)
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);
    m_pixelStagingBuffer1 = m_device->createBuffer(stagingBufferDesc);
    m_pixelStagingBuffer2 = m_device->createBuffer(stagingBufferDesc);

    if (!m_pixelStagingBuffer1 || !m_pixelStagingBuffer2)
        return false;

    return true;
}

void FlatImageView::Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer)
{
    if (!m_leftTexture)
        return;

    const nvrhi::TextureDesc& textureDesc = m_leftTexture->getDesc();
    uint32_t sourceMip = std::min(uint32_t(m_mipLevel), textureDesc.mipLevels - 1);

    auto textureSubresourceSet = nvrhi::TextureSubresourceSet(sourceMip, 1, 0, 1);
    
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FlatImageViewConstants)))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_leftTexture, nvrhi::Format::UNKNOWN, textureSubresourceSet))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_rightTexture, nvrhi::Format::UNKNOWN, textureSubresourceSet))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_pixelBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_PointClampSampler));

    auto bindingSet = m_bindingCache->GetOrCreateBindingSet(bindingSetDesc, m_bindingLayout);

    auto viewport = nvrhi::Viewport(
        m_viewOrigin.x, m_viewOrigin.x + m_viewSize.x,
        m_viewOrigin.y, m_viewOrigin.y + m_viewSize.y,
        0.f, 1.f);

    auto state = nvrhi::GraphicsState()
        .setPipeline(m_graphicsPipeline)
        .addBindingSet(bindingSet)
        .setFramebuffer(framebuffer)
        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(viewport));

    commandList->setGraphicsState(state);

    FlatImageViewConstants constants{};
    constants.viewCenter = m_viewOrigin + m_viewSize * 0.5f;
    constants.textureCenterOffset = float2(m_textureCenterOffset);
    constants.displayScale = m_displayScale;
    constants.textureSize = m_textureSize;
    constants.pixelPickPosition = m_enablePixelInspector ? m_mousePos : int2(-1);
    constants.channelMask = m_channelMask & ((1 << m_textureChannels) - 1);
    constants.displayMode = DisplayMode(m_displayMode);
    constants.splitPosition = m_splitPosition;
    constants.colorScale = m_colorScale;
    constants.applyToneMapping = m_applyToneMapping;
    constants.isSRGB = m_textureSRGB;
    constants.pixelHighlightTopLeft = 0;
    constants.pixelHighlightBottomRight = 0;
    if (m_enablePixelInspector)
    {
        dm::ibox2 pickPixelBounds = GetTexelBounds(m_mousePos);
        if (pickPixelBounds.diagonal().x > 4)
        {
            constants.pixelHighlightTopLeft = pickPixelBounds.m_mins - 1;
            constants.pixelHighlightBottomRight = pickPixelBounds.m_maxs + 1;
        }
    }
    
    commandList->setPushConstants(&constants, sizeof(constants));

    commandList->draw(nvrhi::DrawArguments().setVertexCount(4));

    commandList->copyBuffer(m_pixelStagingBuffer1, 0, m_pixelBuffer, 0, m_pixelBuffer->getDesc().byteSize);
}

void FlatImageView::ReadPixel()
{
    if (!m_enablePixelInspector)
        return;

    float4* readbackData = static_cast<float4*>(m_device->mapBuffer(m_pixelStagingBuffer2, nvrhi::CpuAccessMode::Read));
    if (!readbackData)
        return;

    m_leftPixelValue = readbackData[0];
    m_rightPixelValue = readbackData[1];
    m_device->unmapBuffer(m_pixelStagingBuffer2);
    m_pixelValuesValid = true;

    std::swap(m_pixelStagingBuffer1, m_pixelStagingBuffer2);
}

bool FlatImageView::MousePosUpdate(double xpos, double ypos)
{
    m_mousePos = int2(int(xpos), int(ypos));

    if (m_drag)
    {
        m_textureCenterOffset += m_mousePos - m_dragStart;
        m_splitPosition += m_mousePos.x - m_dragStart.x;
        m_dragStart = m_mousePos;
    }
    else if (m_moveSplit)
    {
        m_splitPosition = m_mousePos.x;
    }

    return true;
}

bool FlatImageView::MouseButtonUpdate(int button, int action, int mods)
{
    if (!m_leftTexture)
        return false;
    
    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT && mods == 0)
    {
        m_drag = true;
        m_dragStart = m_mousePos;
        return true;
    }
    
    if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT && m_drag)
    {
        m_drag = false;
        return true;
    }

    if (m_displayMode == uint32_t(DisplayMode::SplitScreen))
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

    return true;
}

void FlatImageView::SetDisplayScaleStable(float newScale, dm::int2 stablePoint)
{
    const float2 stableUv = WindowPosToUv(stablePoint);
    const float2 splitUv = WindowPosToUv(int2(m_splitPosition, 0));
    
    m_displayScale = newScale;

    // Move the center using the difference between where the mouse cursor actually is 
    // and where it is predicted to be using the new scale
    m_textureCenterOffset += stablePoint - UvToWindowPos(stableUv);

    // Move the split to maintain its position relative to the image
    m_splitPosition = UvToWindowPos(splitUv).x;
}

bool FlatImageView::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (floor(xoffset) == xoffset && floor(yoffset) == yoffset)
    {
        // Integer offsets: this is either a real mouse wheel, or a touchpad zoom gesture - zoom the image
        float multiplier = powf(2.f, yoffset);
        float newScale = std::max(1.f / 16.f, std::min(16.f, m_displayScale * multiplier));

        SetDisplayScaleStable(newScale, m_mousePos);
    }
    else
    {
        // Fractional offsets: it's a touchpad 2-finger pan gesture - pan the image
        double const dragSpeed = 100.f;
        m_textureCenterOffset.x += int(xoffset * dragSpeed);
        m_textureCenterOffset.y += int(yoffset * dragSpeed);
    }

    return true;
}

void FlatImageView::Reset()
{
    m_leftTexture = nullptr;
    m_rightTexture = nullptr;
}

void FlatImageView::SetTextureSize(int width, int height, int mips)
{
    m_textureSize = float2(float(width), float(height));
    m_numMips = mips;
    m_mipLevel = std::min(m_mipLevel, m_numMips - 1);
}

void FlatImageView::SetTextures(nvrhi::ITexture* leftTexture, nvrhi::ITexture* rightTexture, int channels, bool sRGB)
{
    if (m_rightTexture == m_leftTexture && rightTexture != leftTexture)
    {
        // If the right texture just became available, show it
        m_displayMode = uint32_t(DisplayMode::RightTexture);
    }

    // When we first get the image, fit it to the view, but that needs to happen at the end of this function
    const bool fitImageToView = !m_leftTexture && leftTexture;

    m_leftTexture = leftTexture;
    m_rightTexture = rightTexture;
    m_textureChannels = channels;
    m_textureSRGB = sRGB;

    if (!m_leftTexture)
        return;
    
    if (fitImageToView)
        FitImageToView();
}

void FlatImageView::SetViewport(dm::float2 origin, dm::float2 size)
{
    if (m_viewSize.x == 0)
        m_splitPosition = int(size.x) / 2;

    m_viewOrigin = origin;
    m_viewSize = size;
}

void FlatImageView::SetImageName(bool right, const std::string& name)
{
    if (right)
        m_rightImageName = name;
    else
        m_leftImageName = name;
}

void FlatImageView::BuildControlDialog()
{
    ImGuiIO const& io = ImGui::GetIO();
    float const fontSize = ImGui::GetFontSize();

    ImGui::SetNextWindowPos(ImVec2(
        m_viewOrigin.x + m_viewSize.x * 0.5f, 
        m_viewOrigin.y + m_viewSize.y - fontSize * 0.6f),
        ImGuiCond_Always, ImVec2(0.5f, 1.0f));

    ImGui::Begin("Flat Image View", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

    // Mip Level slider
    if (m_numMips > 1)
    {
        ImGui::PushItemWidth(fontSize * 5.f);
        ImGui::SliderInt("##MipLevel", &m_mipLevel, 0, m_numMips - 1, "Mip %d");
        ImGui::PopItemWidth();
    }
    else
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("(No Mips)");
    }
    
    // Channel toggle buttons
    uint32_t availableChannelMask = (1 << m_textureChannels) - 1;
    uint32_t effectiveChannelMask = m_channelMask & availableChannelMask;

    ImGui::SameLine(fontSize * 6.25f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.f, 3.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.f, 0.1f, 0.1f, 1.f));
    ImGui::ToggleButtonFlags("R", &effectiveChannelMask, 1);
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 1.f, 0.1f, 1.f));
    ImGui::BeginDisabled(m_textureChannels < 2);
    ImGui::ToggleButtonFlags("G", &effectiveChannelMask, 2, ImVec2(0.f, 0.f));
    ImGui::EndDisabled();
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.1f, 1.f, 1.f));
    ImGui::BeginDisabled(m_textureChannels < 2);
    ImGui::ToggleButtonFlags("B", &effectiveChannelMask, 4, ImVec2(0.f, 0.f));
    ImGui::EndDisabled();
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 1.f));
    ImGui::BeginDisabled(m_textureChannels < 2);
    ImGui::ToggleButtonFlags("A", &effectiveChannelMask, 8, ImVec2(0.f, 0.f));
    ImGui::EndDisabled();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    m_channelMask = (m_channelMask & ~availableChannelMask) | (effectiveChannelMask & availableChannelMask);

    // Color scale slider and button
    ImGui::SameLine(0.f, fontSize * 1.25f);
    ImGui::PushItemWidth(fontSize * 6.25f);
    ImGui::DragFloat("##ColorScale", &m_colorScale, 0.1f, 0.001f, 100.f, "Color %.3fx", ImGuiSliderFlags_Logarithmic);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("1x"))
        m_colorScale = 1.f;

    // Tone mapping toggle button
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.f, 3.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 1.f));
    ImGui::ToggleButton("ToneMap", &m_applyToneMapping);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    // Image scale slider and buttons
    ImGui::SameLine(0.f, fontSize * 1.25f);
    ImGui::TextUnformatted("Scale:");
    ImGui::SameLine();
    if (ImGui::Button("Fit"))
        FitImageToView();
    ImGui::SameLine();
    if (ImGui::Button("1:1"))
    {
        SetDisplayScaleStable(1.f, int2(m_viewOrigin + m_viewSize * 0.5f));
        m_displayScale = 1.f;
    }


    if (!m_rightTexture)
        m_displayMode = uint32_t(DisplayMode::LeftTexture);
    
    // Second row: display mode selection
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 8, 6 });
    ImGui::Separator();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.f, 3.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 1.f));

    std::tuple<DisplayMode, const char*> modes[] = {
        { DisplayMode::LeftTexture, m_leftImageName.c_str() },
        { DisplayMode::RightTexture, m_rightImageName.c_str() },
        { DisplayMode::Difference, "Abs Diff" },
        { DisplayMode::RelativeDifference, "Rel Diff" },
        { DisplayMode::SplitScreen, "Split-Screen" }
    };

    bool first = true;
    for (const auto& [mode, label] : modes)
    {
        if (!first)
            ImGui::SameLine();
        else
            first = false;

        // Use an ID string with ### to make ImGui element ID independent from the button label which is volatile
        const std::string id = std::string(label) + "###" + std::to_string(int(mode));

        bool active = m_displayMode == uint32_t(mode);
        ImGui::BeginDisabled(m_leftTexture == m_rightTexture);
        ImGui::ToggleButton(id.c_str(), &active, ImVec2(fontSize * 6.45f, 0));
        ImGui::EndDisabled();
        if (active) m_displayMode = uint32_t(mode);

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
    ImGui::PopStyleVar();

    ImGui::End();

    // Pixel Inspector window

    ImGui::SetNextWindowPos(
        ImVec2((m_viewOrigin.x + m_viewSize.x) / io.DisplayFramebufferScale.x - fontSize * 0.6f, fontSize * 2.f),
        ImGuiCond_Always, ImVec2(1.0f, 0.f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(fontSize * 9.5f, -1.f), ImVec2(fontSize * 9.5f, -1.f));
    if (ImGui::Begin("Pixel Inspector", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Checkbox("Enable", &m_enablePixelInspector);
        
        if (m_enablePixelInspector && m_leftTexture)
        {
            bool const useDecimal = m_leftTexture->getDesc().format == nvrhi::Format::RGBA8_UNORM;
            int4 leftColor = int4(round(m_leftPixelValue * 255.f));
            int4 rightColor = int4(round(m_rightPixelValue * 255.f));
            
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.1f, 0.1f, 1.f));
            if (useDecimal)
                ImGui::Text("R: %3d | %3d", leftColor.x, rightColor.x);
            else
                ImGui::Text("R: %.2f | %.2f", m_leftPixelValue.x, m_rightPixelValue.x);
            ImGui::PopStyleColor();
            
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 1.f, 0.1f, 1.f));
            if (useDecimal)
                ImGui::Text("G: %3d | %3d", leftColor.y, rightColor.y);
            else
                ImGui::Text("G: %.2f | %.2f", m_leftPixelValue.y, m_rightPixelValue.y);
            ImGui::PopStyleColor();
            
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 1.f, 1.f));
            if (useDecimal)
                ImGui::Text("B: %3d | %3d", leftColor.z, rightColor.z);
            else
                ImGui::Text("B: %.2f | %.2f", m_leftPixelValue.z, m_rightPixelValue.z);
            ImGui::PopStyleColor();
            
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.f));
            if (useDecimal)
                ImGui::Text("A: %3d | %3d", leftColor.w, rightColor.w);
            else
                ImGui::Text("A: %.2f | %.2f", m_leftPixelValue.w, m_rightPixelValue.w);
            ImGui::PopStyleColor();
        }
        else
            m_pixelValuesValid = false;
    }
    ImGui::End();
}

bool FlatImageView::IsRequestingRestore(int& outRunOrdinal, bool& outRightTexture)
{
    outRunOrdinal = m_requestingRestore;
    outRightTexture = m_requestingRestoreRight;
    m_requestingRestore = -1;
    return outRunOrdinal >= 0;
}

dm::float2 FlatImageView::WindowPosToUv(dm::int2 windowPos) const
{
    const float2 viewCenter = m_viewOrigin + m_viewSize * 0.5f;
    const float2 relativePos = float2(windowPos) - viewCenter - float2(m_textureCenterOffset);
    return 0.5f + relativePos / (m_textureSize * m_displayScale);
}

dm::int2 FlatImageView::UvToWindowPos(dm::float2 uv) const
{
    const float2 viewCenter = m_viewOrigin + m_viewSize * 0.5f;
    const float2 relativePos = (uv - 0.5f) * (m_textureSize * m_displayScale);
    return int2(relativePos + viewCenter) + m_textureCenterOffset;
}

dm::ibox2 FlatImageView::GetTexelBounds(dm::int2 windowPos) const
{
    const float2 viewCenter = m_viewOrigin + m_viewSize * 0.5f;
    const float mipScale = powf(2.f, float(m_mipLevel));
    float2 mipSize;
    mipSize.x = std::max(1.f, floorf(m_textureSize.x / mipScale));
    mipSize.y = std::max(1.f, floorf(m_textureSize.y / mipScale));
    const float2 realMipScale = m_textureSize / mipSize;
    const float2 textureCenter = mipSize * 0.5f;

    const float2 relativePos = float2(windowPos) - viewCenter - float2(m_textureCenterOffset);
    float2 texelPos = textureCenter + relativePos / (m_displayScale * realMipScale);
    texelPos.x = floorf(texelPos.x);
    texelPos.y = floorf(texelPos.y);
    
    int2 windowPosTL = int2((texelPos - textureCenter) * m_displayScale * realMipScale + viewCenter + float2(m_textureCenterOffset));
    texelPos += 1.f;
    int2 windowPosBR = int2((texelPos - textureCenter) * m_displayScale * realMipScale + viewCenter + float2(m_textureCenterOffset) - 1.f);

    return dm::ibox2(windowPosTL, windowPosBR);
}

void FlatImageView::FitImageToView()
{
    float2 scales = m_viewSize / m_textureSize;
    m_displayScale = std::min(scales.x, scales.y);
    m_textureCenterOffset = 0;
}
