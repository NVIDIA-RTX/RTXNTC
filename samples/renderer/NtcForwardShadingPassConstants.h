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

 #ifndef NTC_FORWARD_SHADING_PASS_CONSTANTS_H
 #define NTC_FORWARD_SHADING_PASS_CONSTANTS_H
 
// in FORWARD_SPACE_MATERIAL
#define FORWARD_BINDING_NTC_MATERIAL_CONSTANTS 4
#define FORWARD_BINDING_NTC_LATENTS_TEXTURE 0
#define FORWARD_BINDING_NTC_WEIGHTS_BUFFER 1
#define FORWARD_BINDING_MATERIAL_DIFFUSE_FEEDBACK_UAV 0
#define FORWARD_BINDING_MATERIAL_SPECULAR_FEEDBACK_UAV 1
#define FORWARD_BINDING_MATERIAL_NORMAL_FEEDBACK_UAV 2
#define FORWARD_BINDING_MATERIAL_EMISSIVE_FEEDBACK_UAV 3
#define FORWARD_BINDING_MATERIAL_OCCLUSION_FEEDBACK_UAV 4
#define FORWARD_BINDING_MATERIAL_TRANSMISSION_FEEDBACK_UAV 5
#define FORWARD_BINDING_MATERIAL_OPACITY_FEEDBACK_UAV 6

// in FORWARD_SPACE_SHADING
#define FORWARD_BINDING_NTC_PASS_CONSTANTS 5
#define FORWARD_BINDING_STF_SAMPLER 1
#define FORWARD_BINDING_LATENTS_SAMPLER 2

struct NtcForwardShadingPassConstants
{
    uint frameIndex;
    uint stfFilterMode;
    float feedbackThreshold;
};

#endif // NTC_FORWARD_SHADING_PASS_CONSTANTS_H