# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.18)

project(ntc)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(NTC_WITH_TESTS "Build testing executables" ON)
option(NTC_WITH_NVTT3 "Include NVTT3 library support for BCTest" OFF)
set(NVTT3_SEARCH_PATH "" CACHE PATH "Custom search path for NVTT3")
option(NTC_WITH_RENDERER "Include the Renderer sample" ON)

option(DONUT_WITH_LZ4 "" OFF)
option(DONUT_WITH_MINIZ "" OFF)
option(DONUT_WITH_DX11 "" OFF)
if (WIN32)
    option(DONUT_WITH_DX12 "" ON)
else()
    option(DONUT_WITH_DX12 "" OFF)
endif()
option(DONUT_WITH_VULKAN "" ON)
option(DONUT_WITH_STATIC_SHADERS "" ON)
option(DONUT_WITH_DLSS "" ${NTC_WITH_RENDERER})

# Register our path for CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake")

if (MSVC)
    set (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

if (WIN32)
    set(NTC_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/windows-x64" CACHE PATH "Output directory for the NTC SDK build")
else()
    set(NTC_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/linux-x64" CACHE PATH "Output directory for the NTC SDK build")
endif()

file(MAKE_DIRECTORY ${NTC_BINARY_DIR})

# Place libntc.dll/.so directly into our /bin
set(LIBNTC_BIN_DIRECTORY ${NTC_BINARY_DIR})

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
endif()

# Configure the shader compiler versions and download options before including donut and therefore ShaderMake

include(libraries/RTXNTC-Library/tools/ConfigureCompilers.cmake)
LibNTC_ConfigureShaderCompilerVersions()

# Include the Donut framework with all of its dependencies like NVRHI and ShaderMake.
# Note: ShaderMake downloads the DXC and Slang binaries from GitHub when SHADERMAKE_FIND_COMPILERS is ON.

add_subdirectory(external/donut)

# Configure and include Argparse (custom version)

add_library(argparse STATIC
    external/argparse/argparse.h
    external/argparse/argparse.c)

target_include_directories(argparse PUBLIC external/argparse)
set_property (TARGET argparse PROPERTY FOLDER "Third-Party Libraries")

# Configure and include LodePNG

add_library(lodepng STATIC
    external/lodepng/lodepng.h
    external/lodepng/lodepng.cpp)

target_include_directories(lodepng PUBLIC external/lodepng)
set_property (TARGET lodepng PROPERTY FOLDER "Third-Party Libraries")

# Optionally include NVTT3

if (NTC_WITH_NVTT3)
    find_package(NVTT3)
    if (NVTT3_FOUND)
        include(ConfigureNVTT3)
    else()
        if (WIN32)
            message(SEND_ERROR "Cannot find NVTT3.\n"
                "Please install the package from https://developer.nvidia.com/gpu-accelerated-texture-compression "
                "or provide a custom path through NVTT3_SEARCH_PATH.")
        else()
            message(SEND_ERROR "Cannot find NVTT3.\n"
                "Please download the package from https://developer.nvidia.com/gpu-accelerated-texture-compression "
                "and provide a path to its files through NVTT3_SEARCH_PATH.")
        endif()
    endif()
endif()

# Configure and include LibNTC

option(NTC_WITH_DX12 "" "${DONUT_WITH_DX12}")
option(NTC_WITH_VULKAN "" "${DONUT_WITH_VULKAN}")

add_subdirectory(libraries/RTXNTC-Library)

if (DONUT_WITH_DX12)
    set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.717.1-preview")
    set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk" CACHE STRING "" FORCE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
    include(ConfigureAgilitySDK)
endif()

add_subdirectory(libraries/ntc-utils)
add_subdirectory(tools/cli)
add_subdirectory(tools/explorer)
if (NTC_WITH_RENDERER)
    add_subdirectory(samples/renderer)
endif()

if (NTC_WITH_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/support/tests/bctest/CMakeLists.txt")
    add_subdirectory(support/tests/bctest)
endif()

# Set the default startup project for VS
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ntc-explorer)

# Put the NTC targets into solution folders
set_target_properties(libntc                        PROPERTIES FOLDER "NTC SDK/Libraries")
if (TARGET libntc-shaders)
    set_target_properties(libntc-shaders            PROPERTIES FOLDER "NTC SDK/Libraries")
endif()
set_target_properties(ntc-utils                     PROPERTIES FOLDER "NTC SDK/Libraries")
if (NTC_WITH_RENDERER)
    set_target_properties(ntc-renderer              PROPERTIES FOLDER "NTC SDK/Samples")
    set_target_properties(ntc-renderer-shaders      PROPERTIES FOLDER "NTC SDK/Samples")
endif()
set_target_properties(ntc-cli                       PROPERTIES FOLDER "NTC SDK/Tools")
set_target_properties(ntc-explorer                  PROPERTIES FOLDER "NTC SDK/Tools")
set_target_properties(ntc-explorer-shaders          PROPERTIES FOLDER "NTC SDK/Tools")
if (TARGET bctest)
    set_target_properties(bctest                    PROPERTIES FOLDER "NTC SDK/Tests")
endif()
