# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(
    ntc-renderer
    DESCRIPTION "Neural Texture Compression Scene Renderer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_executable(ntc-renderer WIN32)

set(implot_dir ${CMAKE_SOURCE_DIR}/external/implot)

target_sources(ntc-renderer PRIVATE
    NtcChannelMapping.h
    NtcMaterial.h
    NtcMaterialLoader.cpp
    NtcMaterialLoader.h
    NtcSceneRenderer.cpp
    NtcForwardShadingPass.cpp
    NtcForwardShadingPass.h
    NtcForwardShadingPassConstants.h
    Profiler.cpp
    Profiler.h
    RenderTargets.h
    ${implot_dir}/implot_internal.h
    ${implot_dir}/implot_items.cpp
    ${implot_dir}/implot.cpp
    ${implot_dir}/implot.h
)

file(GLOB feedback_sources
    feedbackmanager/include/*.h
    feedbackmanager/src/*.h
    feedbackmanager/src/*.cpp
    ../../libraries/RTXTS-TTM/include/rtxts-ttm/*.h
    ../../libraries/RTXTS-TTM/src/*.h
    ../../libraries/RTXTS-TTM/src/*.cpp
)

if (DONUT_WITH_DX12)
    target_sources(ntc-renderer PRIVATE ${feedback_sources})
endif()

if (DLSS_SHARED_LIBRARY_PATH)
    add_custom_command(TARGET ntc-renderer POST_BUILD COMMAND
        cmake -E copy_if_different "${DLSS_SHARED_LIBRARY_PATH}" "${NTC_BINARY_DIR}/")
endif()

set_target_properties(ntc-renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(ntc-renderer PRIVATE libntc argparse donut_render donut_app donut_engine)
target_link_libraries(ntc-renderer PRIVATE ntc-utils)

include(${CMAKE_SOURCE_DIR}/external/donut/compileshaders.cmake)

set(shader_sources
    ForwardShadingCommon.hlsli
    LegacyForwardShadingPass.hlsl
    NtcForwardShadingPass_CoopVec.slang
    NtcForwardShadingPass.hlsl
    ForwardShadingPassFeedback.hlsl
)

set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")

set(shader_outputs
    NtcForwardShadingPass
    LegacyForwardShadingPass
    ForwardShadingPassFeedback)

set(shader_outputs_coopvec_dxil
    NtcForwardShadingPass_CoopVec.dxil.h)
set(shader_outputs_coopvec_spirv
    NtcForwardShadingPass_CoopVec.spirv.h)

set(libntc_include_directory "${CMAKE_SOURCE_DIR}/libraries/RTXNTC-Library/include")
set(libstf_include_directory "${CMAKE_SOURCE_DIR}/libraries/RTXTF-Library")
    
target_include_directories(ntc-renderer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(ntc-renderer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../external/donut/shaders")
target_include_directories(ntc-renderer PRIVATE "${libstf_include_directory}")
target_include_directories(ntc-renderer PRIVATE "${CMAKE_SOURCE_DIR}/libraries/RTXTS-TTM/include")
target_include_directories(ntc-renderer PRIVATE "${implot_dir}")

add_custom_target(ntc-renderer-shaders
    DEPENDS ShaderMake
    SOURCES ${shader_sources})

set_source_files_properties(${shader_sources} PROPERTIES VS_TOOL_OVERRIDE "None") 

donut_compile_shaders_all_platforms(
    TARGET ntc-renderer-shaders
    PROJECT_NAME "NTC Renderer"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cfg
    SOURCES ${shader_sources}
    BYPRODUCTS_NO_EXT ${shader_outputs}
    OUTPUT_FORMAT HEADER
    OUTPUT_BASE ${shader_output_dir}
    INCLUDES ${libntc_include_directory} ${libstf_include_directory}
    SHADERMAKE_OPTIONS "--hlsl2021"
    SHADERMAKE_OPTIONS_DXIL "--shaderModel 6_5"
    SHADERMAKE_OPTIONS_SPIRV "--vulkanVersion 1.2"
)

donut_compile_shaders(
    TARGET ntc-renderer-shaders
    PROJECT_NAME "NTC Renderer"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/CoopVecShaders.cfg
    SOURCES ${shader_sources}
    BYPRODUCTS ${shader_outputs_coopvec_dxil}
    OUTPUT_FORMAT HEADER
    DXIL ${shader_output_dir}
    INCLUDES ${libntc_include_directory} ${libstf_include_directory}
    SHADER_MODEL 6_9
    SHADERMAKE_OPTIONS_DXIL "--embedPDB"
)

donut_compile_shaders(
    TARGET ntc-renderer-shaders
    PROJECT_NAME "NTC Renderer"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/CoopVecShaders.cfg
    SOURCES ${shader_sources}
    BYPRODUCTS ${shader_outputs_coopvec_spirv}
    OUTPUT_FORMAT HEADER
    SPIRV_SLANG ${shader_output_dir}
    INCLUDES ${libntc_include_directory} ${libstf_include_directory}
    SHADERMAKE_OPTIONS_SPIRV "-X \"-capability spvCooperativeVectorNV\""
)

add_dependencies(ntc-renderer ntc-renderer-shaders)
