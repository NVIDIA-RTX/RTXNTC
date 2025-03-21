/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if NTC_WITH_DLSS

#include <memory>
#include <nvrhi/nvrhi.h>

class RenderTargets;

namespace donut::engine
{
    class ShaderFactory;
    class PlanarView;
}

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

class DLSS
{
protected:
    bool m_FeatureSupported = false;
    bool m_IsAvailable = false;

    NVSDK_NGX_Handle* m_DlssHandle = nullptr;
    NVSDK_NGX_Parameter* m_Parameters = nullptr;

    static const uint32_t c_ApplicationID = 231313132;

    uint32_t m_InputWidth = 0;
    uint32_t m_InputHeight = 0;
    uint32_t m_OutputWidth = 0;
    uint32_t m_OutputHeight = 0;

    nvrhi::DeviceHandle m_Device;
    nvrhi::CommandListHandle m_FeatureCommandList;

public:
    
    DLSS(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory);

    [[nodiscard]] bool IsSupported() const { return m_FeatureSupported; }
    [[nodiscard]] bool IsAvailable() const { return m_FeatureSupported && m_IsAvailable; }

    virtual void SetRenderSize(
        uint32_t inputWidth, uint32_t inputHeight,
        uint32_t outputWidth, uint32_t outputHeight) = 0;

    virtual void Render(
        nvrhi::ICommandList* commandList,
        const RenderTargets& renderTargets,
        float sharpness,
        bool resetHistory,
        const donut::engine::PlanarView& view,
        const donut::engine::PlanarView& viewPrev) = 0;

    virtual ~DLSS() = default;

#if NTC_WITH_DX12
    static std::unique_ptr<DLSS> CreateDX12(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory);
#endif
#if NTC_WITH_VULKAN
    static std::unique_ptr<DLSS> CreateVK(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory);
#endif
    static void GetRequiredVulkanExtensions(std::vector<std::string>& instanceExtensions, std::vector<std::string>& deviceExtensions);
};

#endif
