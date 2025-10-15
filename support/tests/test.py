#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest
import os
import sys
import shutil
import numpy
from PIL import Image

try:
    import OpenEXR
    import Imath
    _OPEN_EXR_SUPPORTED = True
except ImportError:
    _OPEN_EXR_SUPPORTED = False

_COOPVEC_SUPPORTED = False

FL_DP4A = 0
FL_COOPVEC = 1

FL_STRINGS = {
    FL_DP4A: 'DP4a',
    FL_COOPVEC: 'CoopVec'
}

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))

import ntc

sourceDir = os.path.join(sdkroot, 'assets/materials')
testFilesDir = os.path.join(sdkroot, 'assets/testfiles')
scratchDir = os.path.join(sdkroot, 'assets/testscratch')

def _loadPillowImage(filename):
    image = Image.open(filename)
    return numpy.array(image, dtype=numpy.int16) # Convert to signed int so that (a-b) operation works both ways

def _loadExrImage(filename):
    assert _OPEN_EXR_SUPPORTED
    image = OpenEXR.InputFile(filename)

    header = image.header()
    dataWindow = header['dataWindow']
    width = dataWindow.max.x - dataWindow.min.x + 1
    height = dataWindow.max.y - dataWindow.min.y + 1
    channels = header['channels']
    numChannels = len(channels)
    
    data = numpy.zeros(shape=(width, height, numChannels), dtype=numpy.float32)

    for ch in range(numChannels):
        channelBytes = image.channel('RGBA'[ch], pixel_type=Imath.PixelType(Imath.PixelType.FLOAT))
        channelArray = numpy.frombuffer(channelBytes, dtype=numpy.float32)
        data[:, :, ch] = channelArray.reshape((width, height))

    return data

def _loadAutoImage(filename: str):
    if filename.lower().endswith('.exr'):
        return _loadExrImage(filename)
    else:
        return _loadPillowImage(filename)


class TestCase(unittest.TestCase):

    def setUp(self):
        self.tool = ntc.get_default_tool_path()
        self.prepareScratch()

    def prepareScratch(self):
        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)
        os.makedirs(scratchDir)

    def computePSNR(self, img_path1, img_path2, ignoreExtraChannels, hdr=False):
        "Loads two common format image files and returns PSNR between them in dB"
        
        # Load the images
        arr1 = _loadAutoImage(img_path1)
        arr2 = _loadAutoImage(img_path2)

        # Ensure both images have the same dimensions
        self.assertEqual(arr1.shape[0:2], arr2.shape[0:2])

        if len(arr1.shape) == 2: arr1 = numpy.expand_dims(arr1, 2)
        if len(arr2.shape) == 2: arr2 = numpy.expand_dims(arr2, 2)

        # Ensure that image2 has the same or greter number of channels
        if ignoreExtraChannels:
            self.assertGreaterEqual(arr2.shape[2], arr1.shape[2])
            # Delete the extra channels from arr2, if any
            arr2 = numpy.delete(arr2, range(arr1.shape[2], arr2.shape[2]), 2)
        else:
            self.assertEqual(arr2.shape[2], arr1.shape[2])

        # Compute mean squared error (MSE) between the two images
        mse = numpy.mean((arr1 - arr2) ** 2)

        # Compute PSNR from MSE, assuming that image data is in 0-255 integers
        psnr = 10 * numpy.log10(255**2 / mse)

        return psnr

    def assertFileExists(self, path):
        if not os.path.exists(path):
            raise self.failureException(f"'{path}' does not exist")
        
    def assertBetween(self, value, low, high):
        if value < low or value > high:
            raise self.failureException(f'{value} is not between {low} and {high}')


    def compareOutputImages(self, sourceDir, decompressedDir, expectedPsnrValues, toleranceDb, ignoreExtraChannels = False):

        imageNames = [
            'AmbientOcclusion',
            'Color',
            'Displacement',
            'NormalDX',
            'Roughness'
        ]

        for name, expectedPsnr in zip(imageNames, expectedPsnrValues):
            originalImageFileName = os.path.join(sourceDir, f'{name}.jpg')
            decompressedImageFileName = os.path.join(decompressedDir, f'{name}.tga')

            self.assertFileExists(decompressedImageFileName)

            psnr = self.computePSNR(originalImageFileName, decompressedImageFileName, ignoreExtraChannels)

            if expectedPsnr > 0:
                self.assertBetween(psnr, expectedPsnr - toleranceDb, expectedPsnr + toleranceDb)
            else:
                print(f'{name}: actual PSNR = {psnr:.2f} dB')

class DescribeTestCase(TestCase):

    def __str__(self):
        return 'Describe'
        
    def runTest(self):
        ntcFileName = os.path.join(testFilesDir, f'PavingStones070_5bpp.ntc')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            describe=True,
            graphicsApi='vk'
        )

        result = ntc.run(args)

        self.assertEqual(result.dimensions, (2048, 2048))
        self.assertEqual(result.channels, 10)
        self.assertEqual(result.mipLevels, 1)
        
        global _COOPVEC_SUPPORTED
        _COOPVEC_SUPPORTED = 'CoopVec' in result.gpuFeatures


class CompressionTestCase(TestCase):

    def __str__(self):
        return 'Compression'
    
    def runTest(self):
        sourceMaterialDir = os.path.join(sourceDir, 'PavingStones070')
        ntcFileName = os.path.join(scratchDir, 'PavingStones070.ntc')
        
        args = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=4.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName
        )

        result = ntc.run(args)

        self.assertTrue(os.path.exists(ntcFileName))
        self.assertEqual(result.bitsPerPixel, args.bitsPerPixel)
        self.assertBetween(result.overallPsnr, 27, 30)
        self.assertIsNotNone(result.compressionRuns)
        self.assertEqual(len(result.compressionRuns), 1)
        self.assertEqual(len(result.compressionRuns[0].learningCurve), args.trainingSteps / args.stepsPerIteration)

        decompressedDir = os.path.join(scratchDir, 'output')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='tga',
            bcFormat='none',
        )

        result = ntc.run(args)

        self.compareOutputImages(sourceMaterialDir, decompressedDir, (32, 28, 38, 27, 35), toleranceDb=2.0)

class DeterminismTestCase(TestCase):
    def __str__(self):
        return 'Determinism'
    
    def runTest(self):
        sourceMaterialDir = os.path.join(sourceDir, 'PavingStones070')
        ntcFileName1 = os.path.join(scratchDir, 'PavingStones070_1.ntc')
        ntcFileName2 = os.path.join(scratchDir, 'PavingStones070_2.ntc')

        args1 = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=4.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName1,
            stableTraining=True,
            randomSeed=1234
        )
        result1 = ntc.run(args1)

        args2 = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=4.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName2,
            stableTraining=True,
            randomSeed=1234
        )
        result2 = ntc.run(args2)

        self.assertBetween(result1.overallPsnr, 27, 30)
        self.assertBetween(result2.overallPsnr, result1.overallPsnr - 0.05, result1.overallPsnr + 0.05)

        self.assertTrue(os.path.exists(ntcFileName1))
        self.assertTrue(os.path.exists(ntcFileName2))

        with open(ntcFileName1, 'rb') as f1, open(ntcFileName2, 'rb') as f2:
            ntcData1 = f1.read()
            ntcData2 = f2.read()
            self.assertEqual(ntcData1, ntcData2)

class HdrCompressionTestCase(TestCase):

    def __str__(self):
        return 'HDR Compression'
    
    @unittest.skipIf(not _OPEN_EXR_SUPPORTED, 'Requires OpenEXR')
    def runTest(self):
        sourceMaterialDir = os.path.join(sourceDir, 'HdrChapel')
        ntcFileName = os.path.join(scratchDir, 'HdrChapel.ntc')
        
        args = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=5.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName,
            randomSeed=1337
        )

        result = ntc.run(args)

        self.assertTrue(os.path.exists(ntcFileName))
        self.assertEqual(result.bitsPerPixel, args.bitsPerPixel)
        self.assertBetween(result.overallPsnr, 38, 40)
        self.assertIsNotNone(result.compressionRuns)
        self.assertEqual(len(result.compressionRuns), 1)
        self.assertEqual(len(result.compressionRuns[0].learningCurve), args.trainingSteps / args.stepsPerIteration)

        compressionPsnr = result.overallPsnr

        decompressedDir = os.path.join(scratchDir, 'output')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='exr',
            bcFormat='none',
        )

        result = ntc.run(args)

        psnr = self.computePSNR(
            os.path.join(sourceMaterialDir, 'Color.exr'),
            os.path.join(decompressedDir, 'Color.exr'),
            ignoreExtraChannels=False)
        
        # Note: this PSNR range is different from the range we expect at compression time.
        # The reason is that we compute PSNR based on raw HDR data here. NTC, on the other hand, computes it
        # in the storage color space, which is HLG, and that gives a different answer. Doing HLG is numpy is very slow.
        # That is technically an NTC bug.
        self.assertBetween(psnr, 38, 42)


class DecompressionTestCase(TestCase):

    def __init__(self, api: str, featureLevel: int) -> None:
        super().__init__()
        self.api = api
        self.featureLevel = featureLevel

    def __str__(self):
        return f'Decompression ({self.api}, {FL_STRINGS[self.featureLevel]})'

    def runTest(self):
        if self.api == 'dx12' and os.name != 'nt':
            self.skipTest('DX12 is only available on Windows')
        if self.featureLevel >= FL_COOPVEC and not _COOPVEC_SUPPORTED:
            self.skipTest('CoopVec is not supported')

        sourceMaterialDir = os.path.join(sourceDir, 'PavingStones070')
        ntcFileName = os.path.join(testFilesDir, f'PavingStones070_5bpp.ntc')
        decompressedDir = os.path.join(scratchDir, 'output')
    
        isCuda = self.api == 'cuda'

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='tga',
            bcFormat='none',
            graphicsApi='' if isCuda else self.api
        )

        if not isCuda:
            args.noCoopVec = self.featureLevel != FL_COOPVEC

        result = ntc.run(args)
        
        if isCuda:
            self.assertEqual(result.graphicsApi, '')
        elif self.api == 'vk':
            self.assertEqual(result.graphicsApi, 'Vulkan')
        elif self.api == 'dx12':
            self.assertEqual(result.graphicsApi, 'D3D12')
        
        expectedPsnr = (34.3, 30.4, 40.0, 29.5, 35.8)
            
        self.compareOutputImages(sourceMaterialDir, decompressedDir, expectedPsnr, toleranceDb=1.5, ignoreExtraChannels=not isCuda)
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    # Describe should go first because it also queries the GPU capabilities
    suite.addTest(DescribeTestCase())
    suite.addTest(CompressionTestCase())
    suite.addTest(DeterminismTestCase())
    suite.addTest(HdrCompressionTestCase())

    for api in ('cuda', 'vk', 'dx12'):
        if api == 'cuda':
            suite.addTest(DecompressionTestCase(api=api, featureLevel=FL_DP4A))
        else:
            for featureLevel in (FL_DP4A, FL_COOPVEC):
                suite.addTest(DecompressionTestCase(api=api, featureLevel=featureLevel))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
