#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import sys
import time

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

# =======================================================================================
# NOTE: Modify the `commonSettings` and `experiments` parameters here to suit your needs.

def apply_common_settings(args: ntc.Arguments):
    pass

# Array of (name, { parameter -> value })
experiments = [
    ("0.5",   { "bitsPerPixel": 0.5 }),
    ("1.0",   { "bitsPerPixel": 1.0 }),
    ("2.0",   { "bitsPerPixel": 2.0 }),
    ("4.0",   { "bitsPerPixel": 4.0 }),
    ("8.0",   { "bitsPerPixel": 8.0 }),
    ("16.0",  { "bitsPerPixel": 16 }),
]

# END of user-modifiable section
# =======================================================================================


root = ntc.get_sdk_root_path()
defaultTool = ntc.get_default_tool_path()
defaultDataset = 'data/TestingDataset'

parser = argparse.ArgumentParser()
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
parser.add_argument('--dataset', default = os.path.join(root, defaultDataset), help = 'Path to the directory with test materials, defaults to ../data/TestingDataset')
parser.add_argument('--limit', type = int, help = 'Maximum number of materials to test')
parser.add_argument('--skip', type = int, help = 'Skip the first N materials')
parser.add_argument('--stride', type = int, help = 'Test every Nth material in the dataset')
parser.add_argument('--filter', help = 'File with a list of materials within the dataset to test')
parser.add_argument('--mips', action = 'store_true', help = 'Generate mipmaps and report per-MIP PSNR values.')
parser.add_argument('--curve', action = 'store_true', help = 'Record the PSNR curve during compression.')
parser.add_argument('--trainingSteps', type = int, default = 100000, help = 'Total number of training steps.')
parser.add_argument('--stepsPerIteration', type = int, default = 1000, help = 'Number of training steps between each PSNR measurement.')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use')
parser.add_argument('--output', help = 'Path to the output CSV file')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.dataset):
    print(f"The specified dataset path '{args.dataset}' does not exist.", file = sys.stderr)
    sys.exit(1)

materialFilter = None
if args.filter is not None:
    if args.filter.startswith('@'):
        with open(args.filter[1:], 'r') as file:
            materialFilter = [s.strip() for s in file.readlines()]
    else:
        materialFilter = args.filter.split(',')

targetMipCount = 13

if args.output is not None:
    outputFile = open(args.output, 'w')
else:
    outputFile = sys.stdout

outputFile.write('Name,Experiment,')
if args.mips:
    for mip in range(targetMipCount):
        outputFile.write(f'MIP {mip},')
elif args.curve:
    for step in range(args.stepsPerIteration, args.trainingSteps + 1, args.stepsPerIteration):
        outputFile.write(f'{step},')
    outputFile.write('Final')
else:
    outputFile.write('PSNR')
outputFile.write(',BPP')
outputFile.write(',Time(s)')
outputFile.write('\n')


ordinal = 0
count = 0
tasks = []
for (dirname, subdirs, files) in os.walk(args.dataset):
    # We only want directories that contain image files and no other subdirectories
    if len(files) == 0 or len(subdirs) != 0:
        continue

    ordinal += 1
    if args.stride and (ordinal % args.stride != 0):
        continue
    if args.skip and (ordinal < args.skip):
        continue

    shortDirname = dirname[len(args.dataset)+1:] if dirname.startswith(args.dataset) else dirname
    if args.filter is not None and shortDirname not in materialFilter:
        continue

    for (experimentName, parameters) in experiments:
        task = ntc.Arguments(tool=args.tool)
        task.__dict__.update(parameters)
        task.loadImages = dirname
        task.generateMips = args.mips
        task.compress = True
        task.decompress = True
        task.trainingSteps = args.trainingSteps
        task.stepsPerIteration = args.stepsPerIteration
        apply_common_settings(task)
        tasks.append((task, shortDirname, experimentName))

    count += 1
    if args.limit and count >= args.limit:
        break

startTime = time.time()

def FormatDuration(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


def task_ready(task, result: ntc.Result, originalTaskCount: int, completedTaskCount: int):
    ntcArgs, shortDirname, experimentName = task

    outputFile.write(f'{shortDirname},{experimentName},')

    if args.mips:
        for mip in range(targetMipCount):
            if result.perMipPsnr is not None and mip < len(result.perMipPsnr):
                psnr = result.perMipPsnr[mip]
            else:
                psnr = ''
            outputFile.write(f'{psnr},')
    elif args.curve:
        run = result.compressionRuns[0]
        for psnr in result.compressionRuns:
            outputFile.write(f'{psnr},')
        outputFile.write(f'{result.overallPsnr}')
    else:
        outputFile.write(f'{result.overallPsnr}')

    outputFile.write(f',{result.bitsPerPixel or ""}')
    outputFile.write(f',{result.elapsedTime:.2f}')
    outputFile.write('\n')

    if args.output is not None:
        elapsedTime = time.time() - startTime
        eta = (originalTaskCount - completedTaskCount) * elapsedTime / float(completedTaskCount)
        etaString = FormatDuration(eta)

        print(f'\rDone: {completedTaskCount} / {originalTaskCount}, ETA: {etaString}', end = '', flush = True)


if args.output is not None:
    print('Starting tests...', end = '', flush = True)

terminated = ntc.process_concurrent_tasks(tasks, args.devices, task_ready)

if args.output is not None:
    outputFile.close()
    print('')
    if terminated:
        print('Test aborted.')
        sys.exit(2)
