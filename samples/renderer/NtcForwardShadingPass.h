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

#include <donut/render/ForwardShadingPass.h>

struct NtcMaterial;

enum class NtcMode
{
    InferenceOnSample,
    InferenceOnLoad,
    InferenceOnFeedback
};

class NtcForwardShadingPass : public donut::render::IGeometryPass
{
protected:
    struct PipelineKey
    {
        int weightType = 0;
        donut::engine::MaterialDomain domain = donut::engine::MaterialDomain::Opaque;
        nvrhi::RasterCullMode cullMode = nvrhi::RasterCullMode::Back;
        bool frontCounterClockwise = false;
        bool reverseDepth = false;
        bool hasDepthPrepass = false;
        NtcMode ntcMode = NtcMode::InferenceOnSample;
        bool useSTF = false;
        bool useNtcMaterial = false;

        bool operator==(PipelineKey const& other) const
        {
            return weightType == other.weightType &&
                   domain == other.domain &&
                   cullMode == other.cullMode &&
                   frontCounterClockwise == other.frontCounterClockwise &&
                   reverseDepth == other.reverseDepth &&
                   hasDepthPrepass == other.hasDepthPrepass &&
                   ntcMode == other.ntcMode &&
                   useSTF == other.useSTF &&
                   useNtcMaterial == other.useNtcMaterial;
        }

        bool operator!=(PipelineKey const& other) const
        {
            return !(*this == other);
        }
    };

    struct PipelineKeyHash
    {
        std::size_t operator()(PipelineKey const& s) const noexcept
        {
            size_t hash = 0;
            nvrhi::hash_combine(hash, s.weightType);
            nvrhi::hash_combine(hash, s.domain);
            nvrhi::hash_combine(hash, s.cullMode);
            nvrhi::hash_combine(hash, s.frontCounterClockwise);
            nvrhi::hash_combine(hash, s.reverseDepth);
            nvrhi::hash_combine(hash, s.hasDepthPrepass);
            nvrhi::hash_combine(hash, uint32_t(s.ntcMode));
            nvrhi::hash_combine(hash, s.useSTF);
            return hash;
        }
    };

    nvrhi::DeviceHandle m_device;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::MaterialBindingCache> m_legacyMaterialBindingCache;
    
    nvrhi::BindingLayoutHandle m_viewBindingLayout;
    nvrhi::BindingLayoutHandle m_shadingBindingLayout;
    nvrhi::BindingSetHandle m_viewBindingSet;
    nvrhi::BindingSetHandle m_shadingBindingSet;
    nvrhi::BufferHandle m_viewConstants;
    nvrhi::BufferHandle m_lightConstants;
    nvrhi::BufferHandle m_passConstants;
    nvrhi::SamplerHandle m_stfSampler;
    nvrhi::SamplerHandle m_latentSampler;
    
    nvrhi::BindingLayoutHandle m_materialBindingLayout;
    nvrhi::BindingLayoutHandle m_emptyMaterialBindingLayout;
    nvrhi::BindingLayoutHandle m_materialBindingLayoutFeedback;
    nvrhi::BindingLayoutHandle m_inputBindingLayout;
    std::unordered_map<NtcMaterial const*, nvrhi::BindingSetHandle> m_materialBindingSets;
    std::unordered_map<NtcMaterial const*, nvrhi::BindingSetHandle> m_materialBindingSetsFeedback;
    std::unordered_map<const donut::engine::BufferGroup*, nvrhi::BindingSetHandle> m_inputBindingSets;

    nvrhi::InputLayoutHandle m_inputLayout;
    nvrhi::ShaderHandle m_vertexShader;
    std::unordered_map<PipelineKey, nvrhi::ShaderHandle, PipelineKeyHash> m_pixelShaders;
    std::unordered_map<PipelineKey, nvrhi::GraphicsPipelineHandle, PipelineKeyHash> m_pipelines;

    nvrhi::ShaderHandle GetOrCreatePixelShader(PipelineKey key);
    nvrhi::GraphicsPipelineHandle GetOrCreatePipeline(PipelineKey key, nvrhi::IFramebuffer* framebuffer);
    nvrhi::BindingSetHandle GetOrCreateMaterialBindingSet(NtcMaterial const* material);
    nvrhi::BindingSetHandle GetOrCreateMaterialBindingSetFeedback(NtcMaterial const* material);
    nvrhi::BindingSetHandle CreateInputBindingSet(const donut::engine::BufferGroup* bufferGroup);
    nvrhi::BindingSetHandle GetOrCreateInputBindingSet(const donut::engine::BufferGroup* bufferGroup);
    std::shared_ptr<donut::engine::MaterialBindingCache> CreateLegacyMaterialBindingCache(donut::engine::CommonRenderPasses& commonPasses);
    
public:

    class Context : public donut::render::GeometryPassContext
    {
    public:
        PipelineKey keyTemplate;
        nvrhi::BindingSetHandle inputBindingSet;
        
        uint32_t positionOffset = 0;
        uint32_t texCoordOffset = 0;
        uint32_t normalOffset = 0;
        uint32_t tangentOffset = 0;
    };

    NtcForwardShadingPass(nvrhi::IDevice* device,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses)
        : m_device(device)
        , m_commonPasses(commonPasses)
        , m_shaderFactory(shaderFactory)
    { }

    bool Init();
    void ResetBindingCache();

    void PrepareLights(
        nvrhi::ICommandList* commandList,
        const std::vector<std::shared_ptr<donut::engine::Light>>& lights,
        dm::float3 ambientColorTop,
        dm::float3 ambientColorBottom);

    void PreparePass(Context& context, nvrhi::ICommandList* commandList, uint32_t frameIndex,
        bool useSTF, int stfFilterMode, bool hasDepthPrepass, NtcMode ntcMode, float feedbackThreshold);

    // IGeometryPass implementation

    [[nodiscard]] donut::engine::ViewType::Enum GetSupportedViewTypes() const override;
    void SetupView(donut::render::GeometryPassContext& context, nvrhi::ICommandList* commandList,
        const donut::engine::IView* view, const donut::engine::IView* viewPrev) override;
    bool SetupMaterial(donut::render::GeometryPassContext& context, const donut::engine::Material* material,
        nvrhi::RasterCullMode cullMode, nvrhi::GraphicsState& state) override;
    void SetupInputBuffers(donut::render::GeometryPassContext& context, const donut::engine::BufferGroup* buffers,
        nvrhi::GraphicsState& state) override;
    void SetPushConstants(donut::render::GeometryPassContext& context, nvrhi::ICommandList* commandList,
        nvrhi::GraphicsState& state, nvrhi::DrawArguments& args) override;
};
