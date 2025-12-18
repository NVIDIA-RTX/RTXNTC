# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import argparse

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))

import ntc

parser = argparse.ArgumentParser()
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use')
args = parser.parse_args()


sourceDir = os.path.join(sdkroot, 'assets/materials/PavingStones070')
destDir = os.path.join(sdkroot, 'assets/testfiles')

tasks = []

ntcFileName = os.path.join(destDir, 'PavingStones070_5bpp.ntc')
ddsDir = os.path.join(destDir, 'PavingStones070_5bpp_DDS')

task = ntc.Arguments(
    tool=ntc.get_default_tool_path(),
    loadManifest=os.path.join(sourceDir, 'Manifest.json'),
    generateMips=True,
    compress=True,
    decompress=True,
    bitsPerPixel=5.0,
    randomSeed=1,
    optimizeBC=True,
    graphicsApi='vk',
    saveCompressed=ntcFileName,
)

tasks.append(task)

def ready1(task: ntc.Arguments, result: ntc.Result, originalTaskCount: int, tasksCompleted: int):
    print(f'{task.saveCompressed}: OK')

ntc.process_concurrent_tasks(tasks, args.devices, ready1)


task = ntc.Arguments(
    tool=ntc.get_default_tool_path(),
    loadCompressed=ntcFileName,
    decompress=True,
    noCoopVec=True,
    noDithering=True,
    graphicsApi='vk',
    saveImages=ddsDir,
    saveMips=True
)

tasks = [task]

def ready2(task: ntc.Arguments, result: ntc.Result, originalTaskCount: int, tasksCompleted: int):
    print(f'{task.saveImages}: OK')

ntc.process_concurrent_tasks(tasks, args.devices, ready2)
