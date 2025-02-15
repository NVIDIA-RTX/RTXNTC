:: SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
:: SPDX-License-Identifier: LicenseRef-NvidiaProprietary
::
:: NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
:: property and proprietary rights in and to this material, related
:: documentation and any modifications thereto. Any use, reproduction,
:: disclosure or distribution of this material and related documentation
:: without an express license agreement from NVIDIA CORPORATION or
:: its affiliates is strictly prohibited.

@echo off
pushd %~dp0\..\..

set ARCHIVE=ntc-release.zip

del %ARCHIVE%
7z.exe a -tzip %ARCHIVE% @support/tools/release_files.txt

popd
