# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_library(ntc-utils STATIC)

set_target_properties (ntc-utils PROPERTIES 
    POSITION_INDEPENDENT_CODE ON)

# Read Git branch name into NTC_SDK_VERSION_BRANCH
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_SDK_VERSION_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Read Git short hash into NTC_SDK_VERSION_HASH
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_SDK_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_compile_definitions(ntc-utils PRIVATE
    NTC_SDK_VERSION_BRANCH="${NTC_SDK_VERSION_BRANCH}"
    NTC_SDK_VERSION_HASH="${NTC_SDK_VERSION_HASH}"
)

message(STATUS "NTC SDK version: ${NTC_SDK_VERSION_BRANCH}-${NTC_SDK_VERSION_HASH}")

target_include_directories(ntc-utils PUBLIC include)

target_sources(ntc-utils PRIVATE
    include/ntc-utils/DDSHeader.h
    include/ntc-utils/BufferLoading.h
    include/ntc-utils/DeviceUtils.h
    include/ntc-utils/GraphicsBlockCompressionPass.h
    include/ntc-utils/GraphicsDecompressionPass.h
    include/ntc-utils/GraphicsImageDifferencePass.h
    include/ntc-utils/Manifest.h
    include/ntc-utils/Misc.h
    include/ntc-utils/Semantics.h
    src/BufferLoading.cpp
    src/DeviceUtils.cpp
    src/GraphicsBlockCompressionPass.cpp
    src/GraphicsDecompressionPass.cpp
    src/GraphicsImageDifferencePass.cpp
    src/Manifest.cpp
    src/Misc.cpp
    src/Semantics.cpp
)

target_link_libraries(ntc-utils PUBLIC libntc donut_app)

target_compile_definitions(ntc-utils PUBLIC
    NTC_WITH_DX12=$<BOOL:${DONUT_WITH_DX12}>
    NTC_WITH_VULKAN=$<BOOL:${DONUT_WITH_VULKAN}>)

if (WIN32)
    target_compile_definitions(ntc-utils PRIVATE
        DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION})
endif()
if (DONUT_WITH_DX12)
    target_include_directories(ntc-utils PUBLIC "${DIRECTSTORAGE_SDK_PATH}/native/include")
    target_link_libraries(ntc-utils PUBLIC "${DIRECTSTORAGE_SDK_PATH}/native/lib/x64/dstorage.lib")
endif()