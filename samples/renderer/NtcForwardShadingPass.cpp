/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "NtcForwardShadingPass.h"
#include "NtcMaterial.h"
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/MaterialBindingCache.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/SceneTypes.h>
#include <nvrhi/utils.h>
#include <libntc/ntc.h>

#if NTC_WITH_DX12
    #include "compiled_shaders/NtcForwardShadingPass_CoopVec.dxil.h"
    #include "compiled_shaders/NtcForwardShadingPass.dxil.h"
    #include "compiled_shaders/LegacyForwardShadingPass.dxil.h"
    #include "compiled_shaders/ForwardShadingPassFeedback.dxil.h"
    // This shader comes from Donut - see CMakeLists.txt that adds an include path to .../donut/shaders
    #include "compiled_shaders/passes/forward_vs_buffer_loads.dxil.h"
#endif

#if NTC_WITH_VULKAN
    #include "compiled_shaders/NtcForwardShadingPass_CoopVec.spirv.h"
    #include "compiled_shaders/NtcForwardShadingPass.spirv.h"
    #include "compiled_shaders/LegacyForwardShadingPass.spirv.h"
    // Comes from Donut, same as forward_vs_buffer_loads.dxil.h above
    #include "compiled_shaders/passes/forward_vs_buffer_loads.spirv.h"
#endif

using namespace donut::math;
#include <donut/shaders/forward_cb.h>
#include "NtcForwardShadingPassConstants.h"

std::shared_ptr<donut::engine::Material> NtcSceneTypeFactory::CreateMaterial()
{
    return std::make_shared<NtcMaterial>();
}


nvrhi::ShaderHandle NtcForwardShadingPass::GetOrCreatePixelShader(PipelineKey key)
{
    // These key fields don't affect pixel shaders, zero them
    key.cullMode = nvrhi::RasterCullMode::None;
    key.frontCounterClockwise = false;
    key.reverseDepth = false;

    // See if there already is a pixel shader with that key
    auto it = m_pixelShaders.find(key);
    if (it != m_pixelShaders.end())
        return it->second;

    // Create a new shader
    bool const transmissiveMaterial = 
        key.domain == donut::engine::MaterialDomain::Transmissive ||
        key.domain == donut::engine::MaterialDomain::TransmissiveAlphaTested ||
        key.domain == donut::engine::MaterialDomain::TransmissiveAlphaBlended;
    bool const alphaTestedMaterial = 
        key.domain == donut::engine::MaterialDomain::AlphaTested && !key.hasDepthPrepass ||
        key.domain == donut::engine::MaterialDomain::TransmissiveAlphaTested;
    
    ntc::InferenceWeightType weightType = ntc::InferenceWeightType(key.weightType);
    bool const useCoopVec = weightType == ntc::InferenceWeightType::CoopVecFP8;


    std::vector<donut::engine::ShaderMacro> defines;
    defines.push_back({ "TRANSMISSIVE_MATERIAL", transmissiveMaterial ? "1": "0" });
    defines.push_back({ "ENABLE_ALPHA_TEST", alphaTestedMaterial ? "1" : "0" });

    nvrhi::ShaderHandle pixelShader;
    switch (key.ntcMode)
    {
        case NtcMode::InferenceOnSample:
            defines.push_back({ "USE_NTC_MATERIAL", key.useNtcMaterial ? "1" : "0" });

            if (useCoopVec)
            {
                pixelShader = m_shaderFactory->CreateStaticPlatformShader(
                    DONUT_MAKE_PLATFORM_SHADER(g_NtcForwardShadingPass_CoopVec),
                    &defines, nvrhi::ShaderType::Pixel);
            }
            else
            {
                pixelShader = m_shaderFactory->CreateStaticPlatformShader(
                    DONUT_MAKE_PLATFORM_SHADER(g_NtcForwardShadingPass),
                    &defines, nvrhi::ShaderType::Pixel);
            }
            break;

        case NtcMode::InferenceOnLoad:
            defines.push_back({ "USE_STF", key.useSTF ? "1" : "0" });
            pixelShader = m_shaderFactory->CreateStaticPlatformShader(
                DONUT_MAKE_PLATFORM_SHADER(g_LegacyForwardShadingPass),
                &defines, nvrhi::ShaderType::Pixel);
            break;

        case NtcMode::InferenceOnFeedback:
            defines.push_back({ "USE_STF", key.useSTF ? "1" : "0" });
            pixelShader = m_shaderFactory->CreateStaticPlatformShader(
                donut::engine::StaticShader(), // dxbc - irrelevant
                DONUT_MAKE_DXIL_SHADER(g_ForwardShadingPassFeedback_dxil),
                donut::engine::StaticShader(), // spirv - feedback not supported
                &defines, nvrhi::ShaderType::Pixel);
            break;

        default:
            assert(!"Unknown ntcMode");
    }

    m_pixelShaders[key] = pixelShader;
    return pixelShader;
}

nvrhi::GraphicsPipelineHandle NtcForwardShadingPass::GetOrCreatePipeline(PipelineKey key, nvrhi::IFramebuffer* framebuffer)
{
    if (key.ntcMode != NtcMode::InferenceOnSample)
    {
        key.weightType = 0;
        key.useNtcMaterial = false;
    }
    else
    {
        key.useSTF = true;
    }

    // See if there already is a pipeline with that key
    auto it = m_pipelines.find(key);
    if (it != m_pipelines.end())
        return it->second;

    // Create a new pipeline
    nvrhi::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.inputLayout = m_inputLayout;
    pipelineDesc.VS = m_vertexShader;
    pipelineDesc.renderState.rasterState.frontCounterClockwise = key.frontCounterClockwise;
    pipelineDesc.renderState.rasterState.setCullMode(key.cullMode);
    nvrhi::IBindingLayout* materialBindingLayout = nullptr;
    switch(key.ntcMode)
    {
        case NtcMode::InferenceOnSample:
            materialBindingLayout = key.useNtcMaterial
                ? (nvrhi::IBindingLayout*)m_materialBindingLayout
                : (nvrhi::IBindingLayout*)m_emptyMaterialBindingLayout;
            break;

        case NtcMode::InferenceOnLoad:
            materialBindingLayout = m_legacyMaterialBindingCache->GetLayout();
            break;

        case NtcMode::InferenceOnFeedback:
            materialBindingLayout = m_materialBindingLayoutFeedback;
            break;

        default:
            assert(!"Unknown ntcMode");
    }

    pipelineDesc.bindingLayouts = {
        materialBindingLayout,
        m_inputBindingLayout,
        m_viewBindingLayout,
        m_shadingBindingLayout };

    pipelineDesc.renderState.depthStencilState
        .setDepthFunc(key.reverseDepth
            ? nvrhi::ComparisonFunc::GreaterOrEqual
            : nvrhi::ComparisonFunc::LessOrEqual);
    
    pipelineDesc.PS = GetOrCreatePixelShader(key);

    switch (key.domain)
    {
    case donut::engine::MaterialDomain::Opaque:
    case donut::engine::MaterialDomain::AlphaTested:
        if (key.hasDepthPrepass)
        {
            pipelineDesc.renderState.depthStencilState
                .disableDepthWrite()
                .setDepthFunc(nvrhi::ComparisonFunc::Equal);
        }
        break;

    case donut::engine::MaterialDomain::AlphaBlended: {
        pipelineDesc.renderState.blendState.targets[0]
            .enableBlend()
            .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
            .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
            .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
            .setDestBlendAlpha(nvrhi::BlendFactor::One);
        
        pipelineDesc.renderState.depthStencilState.disableDepthWrite();
        break;
    }

    case donut::engine::MaterialDomain::Transmissive:
    case donut::engine::MaterialDomain::TransmissiveAlphaTested:
    case donut::engine::MaterialDomain::TransmissiveAlphaBlended: {
        pipelineDesc.renderState.blendState.targets[0]
            .enableBlend()
            .setSrcBlend(nvrhi::BlendFactor::One)
            .setDestBlend(nvrhi::BlendFactor::Src1Color)
            .setSrcBlendAlpha(nvrhi::BlendFactor::Zero)
            .setDestBlendAlpha(nvrhi::BlendFactor::One);

        pipelineDesc.renderState.depthStencilState.disableDepthWrite();
        break;
    }
    default:
        return nullptr;
    }

    nvrhi::GraphicsPipelineHandle pipeline = m_device->createGraphicsPipeline(pipelineDesc,
        framebuffer->getFramebufferInfo());
    m_pipelines[key] = pipeline;
    return pipeline;
}

nvrhi::BindingSetHandle NtcForwardShadingPass::GetOrCreateMaterialBindingSet(NtcMaterial const* material)
{
    auto found = m_materialBindingSets.find(material);
    if (found != m_materialBindingSets.end())
        return found->second;

    nvrhi::BindingSetDesc bindingSetDesc;
    nvrhi::BindingSetHandle bindingSet;
    bindingSetDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_MATERIAL_CONSTANTS, material->materialConstants));
    if (material->ntcConstantBuffer)
    {
        bindingSetDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_NTC_MATERIAL_CONSTANTS, material->ntcConstantBuffer));
        bindingSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(FORWARD_BINDING_NTC_LATENTS_TEXTURE, material->ntcLatentsTexture));
        bindingSetDesc.addItem(nvrhi::BindingSetItem::RawBuffer_SRV(FORWARD_BINDING_NTC_WEIGHTS_BUFFER, material->ntcWeightsBuffer));
        bindingSet = m_device->createBindingSet(bindingSetDesc, m_materialBindingLayout);
    }
    else
    {
        bindingSet = m_device->createBindingSet(bindingSetDesc, m_emptyMaterialBindingLayout);
    }

    m_materialBindingSets[material] = bindingSet;
    return bindingSet;
}

static nvrhi::BindingSetItem GetReservedBindingSetItem(uint32_t slot,
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> const& texture, nvrhi::ITexture* fallback)
{
    if (texture)
        return nvrhi::BindingSetItem::Texture_SRV(slot, texture->GetReservedTexture().Get());
    return nvrhi::BindingSetItem::Texture_SRV(slot, fallback);
}

static nvrhi::BindingSetItem GetFeedbackBindingSetItem(uint32_t slot,
    nvrhi::RefCountPtr<nvfeedback::FeedbackTexture> const& texture)
{
    if (texture)
        return nvrhi::BindingSetItem::SamplerFeedbackTexture_UAV(slot, texture->GetSamplerFeedbackTexture().Get());
    return nvrhi::BindingSetItem::SamplerFeedbackTexture_UAV(slot, nullptr);
}

nvrhi::BindingSetHandle NtcForwardShadingPass::GetOrCreateMaterialBindingSetFeedback(NtcMaterial const* material)
{
    auto found = m_materialBindingSetsFeedback.find(material);
    if (found != m_materialBindingSetsFeedback.end())
        return found->second;

    auto const fallbackTexture = m_commonPasses->m_GrayTexture;
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_MATERIAL_CONSTANTS, material->materialConstants))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE,      material->baseOrDiffuseTextureFeedback,        fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE,     material->metalRoughOrSpecularTextureFeedback, fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE,       material->normalTextureFeedback,               fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE,     material->emissiveTextureFeedback,             fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE,    material->occlusionTextureFeedback,            fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE, material->transmissionTextureFeedback,         fallbackTexture))
        .addItem(GetReservedBindingSetItem(FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE,      material->opacityTextureFeedback,              fallbackTexture))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE,      material->baseOrDiffuseTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE,     material->metalRoughOrSpecularTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE,       material->normalTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE,     material->emissiveTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE,    material->occlusionTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE, material->transmissionTextureFeedback))
        .addItem(GetFeedbackBindingSetItem(FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE,      material->opacityTextureFeedback))
    ;

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_materialBindingLayoutFeedback);
 
    m_materialBindingSetsFeedback[material] = bindingSet;
    return bindingSet;    
}

std::shared_ptr<donut::engine::MaterialBindingCache> NtcForwardShadingPass::CreateLegacyMaterialBindingCache(
    donut::engine::CommonRenderPasses& commonPasses)
{
    using namespace donut::engine;

    std::vector<MaterialResourceBinding> materialBindings = {
        { MaterialResource::ConstantBuffer,         FORWARD_BINDING_MATERIAL_CONSTANTS },
        { MaterialResource::DiffuseTexture,         FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE },
        { MaterialResource::SpecularTexture,        FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE },
        { MaterialResource::NormalTexture,          FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE },
        { MaterialResource::EmissiveTexture,        FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE },
        { MaterialResource::OcclusionTexture,       FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE },
        { MaterialResource::TransmissionTexture,    FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE },
        { MaterialResource::OpacityTexture,         FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE }
    };

    return std::make_shared<MaterialBindingCache>(
        m_device,
        nvrhi::ShaderType::Pixel,
        /* registerSpace = */ FORWARD_SPACE_MATERIAL,
        /* registerSpaceIsDescriptorSet = */ true,
        materialBindings,
        commonPasses.m_AnisotropicWrapSampler,
        commonPasses.m_GrayTexture,
        commonPasses.m_BlackTexture);
}

bool NtcForwardShadingPass::Init()
{
    auto vertexShaderDesc = nvrhi::ShaderDesc()
        .setShaderType(nvrhi::ShaderType::Vertex)
        .setEntryName("buffer_loads");

    m_vertexShader = m_shaderFactory->CreateStaticPlatformShader(DONUT_MAKE_PLATFORM_SHADER(g_forward_vs_buffer_loads),
        nullptr, vertexShaderDesc);

    using namespace donut::engine;

    auto viewLayoutDecs = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
        .setRegisterSpace(FORWARD_SPACE_VIEW)
        .setRegisterSpaceIsDescriptorSet(true)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(FORWARD_BINDING_VIEW_CONSTANTS));

    m_viewBindingLayout = m_device->createBindingLayout(viewLayoutDecs);

    auto shadingLayoutDecs = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Pixel)
        .setRegisterSpace(FORWARD_SPACE_SHADING)
        .setRegisterSpaceIsDescriptorSet(true)
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(FORWARD_BINDING_LIGHT_CONSTANTS))
        .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(FORWARD_BINDING_NTC_PASS_CONSTANTS))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_MATERIAL_SAMPLER))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_STF_SAMPLER))
        .addItem(nvrhi::BindingLayoutItem::Sampler(FORWARD_BINDING_LATENTS_SAMPLER));

    m_shadingBindingLayout = m_device->createBindingLayout(shadingLayoutDecs);

    auto materialLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Pixel)
        .setRegisterSpace(FORWARD_SPACE_MATERIAL)
        .setRegisterSpaceIsDescriptorSet(true)
        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(FORWARD_BINDING_MATERIAL_CONSTANTS));

    m_emptyMaterialBindingLayout = m_device->createBindingLayout(materialLayoutDesc);

    materialLayoutDesc
        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(FORWARD_BINDING_NTC_MATERIAL_CONSTANTS))
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_NTC_LATENTS_TEXTURE))
        .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(FORWARD_BINDING_NTC_WEIGHTS_BUFFER));

    m_materialBindingLayout = m_device->createBindingLayout(materialLayoutDesc);

    if (m_device->queryFeatureSupport(nvrhi::Feature::SamplerFeedback))
    {
        auto materialLayoutFeedbackDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Pixel)
            .setRegisterSpace(FORWARD_SPACE_MATERIAL)
            .setRegisterSpaceIsDescriptorSet(true)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(FORWARD_BINDING_MATERIAL_CONSTANTS))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_DIFFUSE_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_SPECULAR_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_NORMAL_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_EMISSIVE_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_OCCLUSION_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_TRANSMISSION_FEEDBACK_UAV))
            .addItem(nvrhi::BindingLayoutItem::SamplerFeedbackTexture_UAV(FORWARD_BINDING_MATERIAL_OPACITY_FEEDBACK_UAV));

        m_materialBindingLayoutFeedback = m_device->createBindingLayout(materialLayoutFeedbackDesc);
    }

    auto inputBindingLayoutDesc = nvrhi::BindingLayoutDesc()
        .setVisibility(nvrhi::ShaderType::Vertex)
        .setRegisterSpace(FORWARD_SPACE_INPUT)
        .setRegisterSpaceIsDescriptorSet(true)
        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(FORWARD_BINDING_VERTEX_BUFFER))
        .addItem(nvrhi::BindingLayoutItem::PushConstants(FORWARD_BINDING_PUSH_CONSTANTS, sizeof(ForwardPushConstants)));
        
    m_inputBindingLayout = m_device->createBindingLayout(inputBindingLayoutDesc);

    int const numConstantBufferVersions = 16;
    m_viewConstants = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ForwardShadingViewConstants), "ForwardShadingViewConstants", numConstantBufferVersions));
    m_lightConstants = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ForwardShadingLightConstants), "ForwardShadingLightConstants", numConstantBufferVersions));
    m_passConstants = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(NtcForwardShadingPassConstants), "NtcForwardShadingPassConstants", numConstantBufferVersions));

    auto viewBindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_VIEW_CONSTANTS, m_viewConstants));

    m_viewBindingSet = m_device->createBindingSet(viewBindingSetDesc, m_viewBindingLayout);

    auto samplerDesc = nvrhi::SamplerDesc()
        .setAllFilters(false)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_stfSampler = m_device->createSampler(samplerDesc);

    auto latentsSamplerDesc = nvrhi::SamplerDesc()
        .setMinFilter(true)
        .setMagFilter(true)
        .setMipFilter(false)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_latentSampler = m_device->createSampler(latentsSamplerDesc);

    auto shadingBindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_LIGHT_CONSTANTS, m_lightConstants))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(FORWARD_BINDING_NTC_PASS_CONSTANTS, m_passConstants))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_MATERIAL_SAMPLER, m_commonPasses->m_AnisotropicWrapSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_STF_SAMPLER, m_stfSampler))
        .addItem(nvrhi::BindingSetItem::Sampler(FORWARD_BINDING_LATENTS_SAMPLER, m_latentSampler));

    m_shadingBindingSet = m_device->createBindingSet(shadingBindingSetDesc, m_shadingBindingLayout);

    m_legacyMaterialBindingCache = CreateLegacyMaterialBindingCache(*m_commonPasses);

    return true;
}

void NtcForwardShadingPass::ResetBindingCache()
{
    m_materialBindingSets.clear();
    m_materialBindingSetsFeedback.clear();
    m_legacyMaterialBindingCache->Clear();
}

void NtcForwardShadingPass::PrepareLights(
    nvrhi::ICommandList* commandList,
    const std::vector<std::shared_ptr<donut::engine::Light>>& lights,
    dm::float3 ambientColorTop,
    dm::float3 ambientColorBottom)
{
    ForwardShadingLightConstants constants = {};

    for (int nLight = 0; nLight < std::min(static_cast<int>(lights.size()), FORWARD_MAX_LIGHTS); nLight++)
    {
        const auto& light = lights[nLight];

        LightConstants& lightConstants = constants.lights[constants.numLights];
        light->FillLightConstants(lightConstants);

        ++constants.numLights;
    }

    constants.ambientColorTop = float4(ambientColorTop, 0.f);
    constants.ambientColorBottom = float4(ambientColorBottom, 0.f);

    commandList->writeBuffer(m_lightConstants, &constants, sizeof(constants));
}

donut::engine::ViewType::Enum NtcForwardShadingPass::GetSupportedViewTypes() const
{
    return donut::engine::ViewType::PLANAR;
}

void NtcForwardShadingPass::PreparePass(Context& context, nvrhi::ICommandList* commandList, uint32_t frameIndex,
    bool useSTF, int stfFilterMode, bool hasDepthPrepass, NtcMode ntcMode, float feedbackThreshold)
{
    NtcForwardShadingPassConstants passConstants {};
    passConstants.frameIndex = frameIndex;
    passConstants.stfFilterMode = stfFilterMode;
    passConstants.feedbackThreshold = feedbackThreshold;
    commandList->writeBuffer(m_passConstants, &passConstants, sizeof(passConstants));
    context.keyTemplate.hasDepthPrepass = hasDepthPrepass;
    context.keyTemplate.ntcMode = ntcMode;
    context.keyTemplate.useSTF = useSTF;
}

void NtcForwardShadingPass::SetupView(
    donut::render::GeometryPassContext& abstractContext,
    nvrhi::ICommandList* commandList,
    const donut::engine::IView* view,
    const donut::engine::IView* viewPrev)
{
    auto& context = static_cast<Context&>(abstractContext);
    
    ForwardShadingViewConstants viewConstants = {};
    view->FillPlanarViewConstants(viewConstants.view);
    commandList->writeBuffer(m_viewConstants, &viewConstants, sizeof(viewConstants));

    context.keyTemplate.frontCounterClockwise = view->IsMirrored();
    context.keyTemplate.reverseDepth = view->IsReverseDepth();

}

bool NtcForwardShadingPass::SetupMaterial(
    donut::render::GeometryPassContext& abstractContext,
    const donut::engine::Material* material,
    nvrhi::RasterCullMode cullMode,
    nvrhi::GraphicsState& state)
{
    auto& context = static_cast<Context&>(abstractContext);

    auto ntcMaterial = static_cast<NtcMaterial const*>(material);

    PipelineKey key = context.keyTemplate;
    key.cullMode = cullMode;
    key.domain = material->domain;
    key.weightType = ntcMaterial->weightType;
    key.useNtcMaterial = key.ntcMode == NtcMode::InferenceOnSample && ntcMaterial->ntcConstantBuffer != nullptr;

    nvrhi::IBindingSet* materialBindingSet = nullptr;
    switch(key.ntcMode)
    {
        case NtcMode::InferenceOnSample:
            materialBindingSet = GetOrCreateMaterialBindingSet(ntcMaterial);
            break;

        case NtcMode::InferenceOnLoad:
            materialBindingSet = m_legacyMaterialBindingCache->GetMaterialBindingSet(ntcMaterial);
            break;

        case NtcMode::InferenceOnFeedback:
            materialBindingSet = GetOrCreateMaterialBindingSetFeedback(ntcMaterial);
            break;

        default:
            assert(!"Unknown ntcMode");
    }

    if (!materialBindingSet)
        return false;

    nvrhi::IGraphicsPipeline* pipeline = GetOrCreatePipeline(key, state.framebuffer);
    
    if (!pipeline)
        return false;

    state.pipeline = pipeline;
    state.bindings = { materialBindingSet, context.inputBindingSet, m_viewBindingSet, m_shadingBindingSet };

    return true;
}

void NtcForwardShadingPass::SetupInputBuffers(
    donut::render::GeometryPassContext& abstractContext,
    const donut::engine::BufferGroup* buffers,
    nvrhi::GraphicsState& state)
{
    auto& context = static_cast<Context&>(abstractContext);
    
    context.inputBindingSet = GetOrCreateInputBindingSet(buffers);

    state.indexBuffer = { buffers->indexBuffer, nvrhi::Format::R32_UINT, 0 };

    context.positionOffset = buffers->getVertexBufferRange(donut::engine::VertexAttribute::Position).byteOffset;
    context.texCoordOffset = buffers->getVertexBufferRange(donut::engine::VertexAttribute::TexCoord1).byteOffset;
    context.normalOffset = buffers->getVertexBufferRange(donut::engine::VertexAttribute::Normal).byteOffset;
    context.tangentOffset = buffers->getVertexBufferRange(donut::engine::VertexAttribute::Tangent).byteOffset;
}

nvrhi::BindingSetHandle NtcForwardShadingPass::CreateInputBindingSet(const donut::engine::BufferGroup* bufferGroup)
{
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(FORWARD_BINDING_INSTANCE_BUFFER, bufferGroup->instanceBuffer))
        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(FORWARD_BINDING_VERTEX_BUFFER, bufferGroup->vertexBuffer))
        .addItem(nvrhi::BindingSetItem::PushConstants(FORWARD_BINDING_PUSH_CONSTANTS, sizeof(ForwardPushConstants)));

    return m_device->createBindingSet(bindingSetDesc, m_inputBindingLayout);
}

nvrhi::BindingSetHandle NtcForwardShadingPass::GetOrCreateInputBindingSet(const donut::engine::BufferGroup* bufferGroup)
{
    auto it = m_inputBindingSets.find(bufferGroup);
    if (it == m_inputBindingSets.end())
    {
        auto bindingSet = CreateInputBindingSet(bufferGroup);
        m_inputBindingSets[bufferGroup] = bindingSet;
        return bindingSet;
    }

    return it->second;
}

void NtcForwardShadingPass::SetPushConstants(
    donut::render::GeometryPassContext& abstractContext,
    nvrhi::ICommandList* commandList,
    nvrhi::GraphicsState& state,
    nvrhi::DrawArguments& args)
{
    auto& context = static_cast<Context&>(abstractContext);

    ForwardPushConstants constants;
    constants.startInstanceLocation = args.startInstanceLocation;
    constants.startVertexLocation = args.startVertexLocation;
    constants.positionOffset = context.positionOffset;
    constants.texCoordOffset = context.texCoordOffset;
    constants.normalOffset = context.normalOffset;
    constants.tangentOffset = context.tangentOffset;

    commandList->setPushConstants(&constants, sizeof(constants));

    args.startInstanceLocation = 0;
    args.startVertexLocation = 0;
}