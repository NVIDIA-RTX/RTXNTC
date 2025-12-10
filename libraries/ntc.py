# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This module provides an interface to the NTC-CLI tool that is more convenient than just calling it with subprocess.
Use like this:

  task = ntc.Arguments(
    tool = ntc.get_default_tool_path(),
    loadImages = '/path/to/images',
    compress = True,
    ...
  )
  result = ntc.run(task)
  print(f'Compression successful, PSNR = {result.overallPsnr})

When an error happens in ntc-cli, the run(...) function will raise a RuntimeError.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Callable
from argparse import Namespace
import subprocess
import re
import os
import signal
import sys
import threading
import time
import traceback


def get_sdk_root_path():
    "Returns the path to the NTC SDK root, assuming the original SDK directory structure."
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_default_tool_path():
    "Returns the path to the ntc-cli tool, assuming the original SDK directory structure."
    sdkroot = get_sdk_root_path()
    filename = 'bin/windows-x64/ntc-cli.exe' if os.name == 'nt' else 'bin/linux-x64/ntc-cli'
    return os.path.join(sdkroot, filename)

def get_default_imagediff_path():
    "Returns the path to the imagediff tool, assuming the original SDK directory structure."
    sdkroot = get_sdk_root_path()
    filename = 'bin/windows-x64/imagediff.exe' if os.name == 'nt' else 'bin/linux-x64/imagediff'
    return os.path.join(sdkroot, filename)

@dataclass
class LatentShape:
    gridSizeScale: int
    numFeatures: int

@dataclass
class Arguments:
    """
    A structure that defines arguments for executing ntc-cli.
    Most fields in Arguments map to command line arguments directly.
    """

    # Path to the ntc-cli executable, required
    tool: str

    # Extra arguments for ntc-cli, passed verbatim
    customArguments: str = ''

    # Content of the manifest to be passed via STDIN, if any
    stdin: Optional[str] = None
    
    # Graphics API, can be 'dx12' or 'vk'
    graphicsApi: str = ''

    # Inverse feature toggles - these map to '--no-coopVec' etc.
    noCoopVec: bool = False
    
    # All the below parameters are passed directly as command line arguments to ntc-cli
    adapter: Optional[int] = None
    bcFormat: str = ''
    bcPsnrThreshold: Optional[float] = None
    bcPsnrOffset: Optional[float] = None
    benchmark: Optional[int] = None
    bitsPerPixel: Optional[float] = None
    compress: bool = False
    cudaDevice: Optional[int] = None
    debug: bool = False
    decompress: bool = False
    describe: bool = False
    dimensions: str = ''
    discardMaskedOutPixels: bool = False
    experimentalKnob: Optional[float] = None
    generateMips: bool = False
    gdeflate: str = ''
    gdeflateLevel: Optional[int] = None
    gdeflateThreshold: Optional[float] = None
    gpuGDeflate: bool = False
    gridLearningRate: Optional[float] = None
    imageFormat: str = ''
    kPixelsPerBatch: Optional[int] = None
    latentShape: Optional[LatentShape] = None
    listAdapters: bool = False
    listCudaDevices: bool = False
    loadCompressed: str = ''
    loadImages: str = ''
    loadManifest: str = ''
    loadMips: bool = False
    matchBcPsnr: bool = False
    maxBcPsnr: Optional[float] = None
    maxBitsPerPixel: Optional[float] = None
    minBcPsnr: Optional[float] = None
    networkLearningRate: Optional[float] = None
    optimizeBC: bool = False
    randomSeed: Optional[int] = None
    readManifestFromStdin: bool = False
    saveCompressed: str = ''
    saveImages: str = ''
    saveManifest: str = ''
    saveMips: bool = False
    stableTraining: bool = False
    stepsPerIteration: Optional[int] = None
    targetPsnr: Optional[float] = None
    trainingSteps: Optional[int] = None

    def get_command_line(self) -> List[str]:
        "Returns the command line with the provided arguments, as a list passable to subprocess.call."

        result = [self.tool]
        for name, value in self.__dict__.items():
            # Process the special case fields
            if name == 'tool':
                pass
            elif name == 'stdin':
                pass
            elif name == 'customArguments':
                if value is not None: result += value.split(' ')
            elif name == 'graphicsApi':
                if value == 'vk': result.append('--vk')
                elif value == 'dx12': result.append('--dx12')
                elif value != '': raise ValueError(f'Unrecognized graphicsApi = {value}')
            elif name == 'noCoopVec':
                if value: result.append('--no-coopVec')
            else:
                # Generic fields - decide what to do based on the data type
                if value is None or value == '':
                    # Skip unset parameters
                    pass
                elif isinstance(value, bool):
                    # Boolean parameters are just switches if the value is True
                    if value: result.append(f'--{name}')
                elif isinstance(value, (int, float, str)):
                    # Simple data types are passed by value
                    result.append(f'--{name}')
                    result.append(str(value))
                elif isinstance(value, LatentShape):
                    # Expand the LatentShape members
                    for name2, value2 in value.__dict__.items():
                        result.append(f'--{name2}')
                        result.append(str(value2))
                else:
                    raise ValueError(f'Unrecognized value type for {name} = {repr(value)}')
        return result


@dataclass
class CompressionRun:
    bitsPerPixel: Optional[float] = None
    learningCurve: Optional[List[Tuple[int, float, float]]] = None # (steps, ms/step, psnr)

@dataclass
class Result:
    elapsedTime: float = 0 # total ntc-cli execution time in seconds
    overallPsnr: Optional[float] = None
    overallPsnrFP8: Optional[float] = None
    perMipPsnr: Optional[List[float]] = None
    bitsPerPixel: Optional[float] = None
    combinedBcPsnr: Optional[float] = None
    combinedBcBitsPerPixel: Optional[float] = None
    compressionRuns: Optional[List[CompressionRun]] = None
    decompressionTime: Optional[float] = None
    savedFileSize: Optional[int] = None
    savedFileBpp: Optional[float] = None
    gpuName: str = ''
    graphicsApi: str = ''
    gpuFeatures: Optional[List[str]] = None # may contain 'CoopVec', 'GDeflate'

    # describe command output:
    dimensions: Optional[Tuple[int, int]] = None # (width, height)
    channels: Optional[int] = None
    mipLevels: Optional[int] = None
    latentShape: Optional[LatentShape] = None # also available when using --compress --bitsPerPixel <bpp>
    # TODO: add texture info

@dataclass
class RuntimeError(Exception):
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        s = f'The following command failed with code {self.returncode}:\n'
        s += f'> {" ".join(self.command)}\n'
        if self.stdout: s += f'stdout:\n{self.stdout}'
        if self.stderr: s += f'stderr:\n{self.stderr}'
        return s

def _create_or_append_list(lst: Optional[List[Any]], x: Any) -> List[Any]:
    if lst is None:
        return [x]
    lst.append(x)
    return lst

class Regex:
    def __init__(self, pattern: str):
        self.regex = re.compile(pattern)

    def parse(self, line: str):
        match = self.regex.match(line)
        if not match:
            return None
        return Namespace(**match.groupdict())

_baseCompRateRegex = Regex(r'Base compression rate: --bitsPerPixel (?P<bpp>[0-9\.]+)')
_bppRegex = Regex(r'Selected compression rate: (?P<bpp>[0-9\.]+) bpp, (?P<psnr>[0-9\.]+|inf) dB PSNR')
_bcQualityRegex = Regex(r'Combined BCn PSNR: (?P<psnr>[0-9\.]+|inf) dB, bit rate: (?P<bpp>[0-9\.]+) bpp')
_cudaDecompressionTimeRegex = Regex(r'CUDA decompression time: (?P<milliseconds>[0-9\.]+) ms')
_dimensionsRegex = Regex(r'Dimensions: (?P<width>\d+)x(?P<height>\d+), (?P<channels>\d+) channels, (?P<mipLevels>\d+) mip level\(s\)')
_experimentRegex = Regex(r'Experiment (?P<index>\d+): (?P<bpp>[0-9\.]+) bpp')
_fileSizeRegex = Regex(r'(Estimated file|File) size: (?P<bytes>\d+) bytes, (?P<bpp>[0-9\.]+) bits per pixel')
_graphicsDecompressionTimeRegex = Regex(r'Median decompression time over \d+ iterations: (?P<milliseconds>[0-9\.]+) ms')
_latentShapeRegex = Regex(r'Latent shape: --gridSizeScale (?P<gss>\d+) --numFeatures (?P<nf>\d+)')
_mipRegex = Regex(r'MIP\s+(?P<mipLevel>\d+)\s+PSNR: (?P<psnr>[0-9\.]+|inf) dB')
_overallPsnrRegex = Regex(r'Overall PSNR \((?P<type>\w+) weights\): (?P<psnr>[0-9\.]+|inf) dB')
_stepRegex = Regex(r'Training: (?P<steps>\d+) steps, (?P<milliseconds>[0-9\.]+) ms/step, intermediate PSNR: (?P<psnr>[0-9\.]+|inf) dB')
_systemRegex = Regex(r'Using (?P<gpu>.+) with (?P<api>.+) API\. CoopVec \[(?P<coopVec>[YN])\], GDeflate \[(?P<gdeflate>[YN])\]')
_imageDiffRegex = Regex(r'PAIR (?P<pair>\d+) MIP\s+(?P<mipLevel>\d+): MSE = (?P<mse>[0-9\.]+|inf), PSNR = (?P<psnr>[0-9\.]+|inf) dB')


def run(args: Arguments) -> Result:
    "Executes the NTC-CLI tool with the provided arguments and returns its interpreted output as a Results object."

    command = args.get_command_line()

    taskStartTime = time.time()
    output = subprocess.run(command, input=args.stdin, capture_output=True, text=True)
    taskEndTime = time.time()

    if output.returncode != 0:
        raise RuntimeError(command, output.returncode, output.stdout, output.stderr)
    
    result = Result(
        elapsedTime=taskEndTime - taskStartTime,
        bitsPerPixel=args.bitsPerPixel # if the tool doesn't give us selected BPP, inherit it from the arguments
    )

    compressionRun = CompressionRun()

    for line in output.stdout.splitlines():
        if m := _baseCompRateRegex.parse(line):
            result.bitsPerPixel = float(m.bpp)

        elif m := _bppRegex.parse(line):
            result.bitsPerPixel = float(m.bpp)
            result.overallPsnr = float(m.psnr)

        elif m := _bcQualityRegex.parse(line):
            result.combinedBcPsnr = float(m.psnr)
            result.combinedBcBitsPerPixel = float(m.bpp)

        elif m := _cudaDecompressionTimeRegex.parse(line):
            result.decompressionTime = float(m.milliseconds)

        elif m := _dimensionsRegex.parse(line):
            result.dimensions = int(m.width), int(m.height)
            result.channels = int(m.channels)
            result.mipLevels = int(m.mipLevels)

        elif m := _experimentRegex.parse(line):
            if compressionRun.learningCurve:
                result.compressionRuns = _create_or_append_list(result.compressionRuns, compressionRun)
            compressionRun = CompressionRun(bitsPerPixel=float(m.bpp))

        elif m := _fileSizeRegex.parse(line):
            result.savedFileSize = int(m.bytes)
            result.savedFileBpp = float(m.bpp)

        elif m := _graphicsDecompressionTimeRegex.parse(line):
            result.decompressionTime = float(m.milliseconds)

        elif m := _latentShapeRegex.parse(line):
            result.latentShape = LatentShape(gridSizeScale=int(m.gss), numFeatures=int(m.nf))

        elif m := _mipRegex.parse(line):
            result.perMipPsnr = _create_or_append_list(result.perMipPsnr, float(m.psnr))

        elif m := _overallPsnrRegex.parse(line):
            if m.type == 'FP8':
                result.overallPsnrFP8 = float(m.psnr)
            else:
                result.overallPsnr = float(m.psnr)

        elif m := _stepRegex.parse(line):
            tuple = int(m.steps), float(m.milliseconds), float(m.psnr)
            compressionRun.learningCurve = _create_or_append_list(compressionRun.learningCurve, tuple)

        elif m := _systemRegex.parse(line):
            result.gpuName = m.gpu
            result.graphicsApi = m.api
            result.gpuFeatures = []
            if m.coopVec == 'Y': result.gpuFeatures.append('CoopVec')
            if m.gdeflate == 'Y': result.gpuFeatures.append('GDeflate')
        
    if compressionRun.learningCurve:
        result.compressionRuns = _create_or_append_list(result.compressionRuns, compressionRun)

    return result


def process_concurrent_tasks(tasks: List[Any], devices: List[int], ready: Callable) -> bool:
    """
    Executes the tasks from the list on one or more GPUs concurrently.
    The 0-based indices of CUDA devices are provided in the 'devices' argument.

    The 'tasks' argument is a list of tasks, where each task may be either an Arguments instance,
    or a tuple (Arguments, ...).

    The 'ready' argument is a function that gets called on each successful task completion,
    with the following arguments:
    
        def ready(task, result: ntc.Result, originalTaskCount: int, tasksCompleted: int):

    The 'task' argument to 'ready' is the original task from the input list,
    which may be Arguments or tuple. The 'ready' function is called from the worker threads,
    but under a mutex, so only one call at a time.
    """

    mutex = threading.Lock()
    terminate = False
    originalTaskCount = len(tasks)
    tasksCompleted = 0

    def _sigint_handler(number, stack):
        nonlocal terminate
        terminate = True
        print('\nSIGINT received, stopping.', file=sys.stderr)

    def _thread_function(device):
        nonlocal terminate

        while not terminate:
            # Take the next task from the list
            task = None
            with mutex:
                if len(tasks) > 0:
                    task = tasks[0]
                    del tasks[0]
            if task is None:
                break

            # Extract the Arguments from the task
            args : Arguments = task[0] if isinstance(task, Tuple) else task
            assert isinstance(args, Arguments)
    
            # Add the device argument to the command.
            args.cudaDevice = device

            # Run the task
            try:
                result = run(args)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    if not terminate:
                        print(f'\nNTC error: {e}', file=sys.stderr)
                else:
                    traceback.print_exception(e, file=sys.stderr)
                terminate = True
                return
            
            # Call the ready function
            with mutex:
                nonlocal tasksCompleted
                tasksCompleted += 1

                try:
                    ready(task, result, originalTaskCount, tasksCompleted)
                except Exception as e:
                    print('\nError in the "ready" callback:')
                    traceback.print_exception(e, file=sys.stderr)
                    terminate = True
                    return


    # Validate the tasks before starting threads
    for task in tasks:
        if isinstance(task, Tuple):
            if not isinstance(task[0], Arguments):
                raise ValueError(f'Task {task} is a tuple but its first member is not Arguments')
        elif not isinstance(task, Arguments):
            raise ValueError(f'Task {task} is neither a tuple nor Arguments.')

    # Install and then remove a SIGINT handler with a try-finally block
    try:
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _sigint_handler)

        # Launch worker threads, one per device
        threads = []
        for device in devices:
            thread = threading.Thread(target = _thread_function, args = [device])
            thread.start()
            threads.append(thread)
            
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    finally:
        signal.signal(signal.SIGINT, old_handler)

    return terminate

def run_imagediff(tool: str, files: List[str], extraArgs: List[str] = []):
    """
    Runs the ImageDiff tool on the two specified image files and returns the parsed result
    as a list of (PAIR, MIP, MSE, PSNR) tuples, one tuple per MIP level.
    
    In this list, PAIR is a 0-based index of the image pair.
    Files 0 and 1 are pair 0, files 2 and 3 are pair 1, and so on.
    """

    command = [tool] + files + extraArgs

    output = subprocess.run(command, capture_output=True, text=True)

    if output.returncode != 0:
        raise RuntimeError(command, output.returncode, output.stdout, output.stderr)
    
    results = []
    for line in output.stdout.splitlines():
        if m := _imageDiffRegex.parse(line):
            results.append((int(m.pair), int(m.mipLevel), float(m.mse), float(m.psnr)))

    return results
