# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(
    imagediff
    DESCRIPTION "Image Difference Tool"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_executable(imagediff)

set_target_properties(imagediff PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(imagediff PRIVATE libntc ntc-utils stb argparse donut_app donut_engine)

target_sources(imagediff PRIVATE
    ImageDiff.cpp
)
