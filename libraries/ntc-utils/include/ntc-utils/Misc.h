/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <libntc/ntc.h>

ntc::VersionInfo GetNtcSdkVersion();

void GetNvidiaSansFont(void const** pOutData, size_t* pOutSize);

#ifdef _MSC_VER // MSVC doesn't have strcasecmp
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif
