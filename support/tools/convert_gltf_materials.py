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

import argparse
import copy
import json
import os
import PIL
import PIL.Image
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

defaultTool = ntc.get_default_tool_path()

parser = argparse.ArgumentParser()
parser.add_argument('inputFiles', nargs = '+', help = 'Input GLTF files')
group = parser.add_mutually_exclusive_group()
group.add_argument('--bitsPerPixel', default = None, type = float, help = 'Compression bits per pixel value')
group.add_argument('--targetPsnr', default = None, type = float, help = 'Target PSNR value in dB for adaptive compressoin')
parser.add_argument('--maxBitsPerPixel', default = None, type = float, help = 'Maximum bits per pixel value for adaptive compression')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use, such as --devices 0 1')
parser.add_argument('--keepManifests', action = 'store_true', help = 'Don\'t delete the manifest files after compression')
parser.add_argument('--output', required = True, help = 'Path to the output directory for the material NTC files')
parser.add_argument('--skipExisting', action = 'store_true', help = 'Skip materials where the NTC file already exists')
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
parser.add_argument('--verbose', action = 'store_true', help = 'Print the commands before executing them')
parser.add_argument('--dryRun', action = 'store_true', help = 'Process materials and write manifest files, but do not compress')
parser.add_argument('--args', help = 'Custom arguments for the compression tool')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.output):
    os.makedirs(args.output)

NTC_MAX_CHANNELS = 16
NTC_MIME_TYPE = 'image/vnd-nvidia.ntc'
NV_TEXTURE_SWIZZLE_EXTENSION_NAME = 'NV_texture_swizzle'

@dataclass
class TextureParams:
    name: str
    gltfPath: str
    maxChannels: int
    semantics: dict
    bcFormat: str
    sRGB: bool = False

textureTypes = [
    TextureParams(
        name = 'BaseColor', 
        gltfPath = 'pbrMetallicRoughness/baseColorTexture', 
        maxChannels = 4,
        semantics = { 'Albedo': 'RGB', 'AlphaMask': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'DiffuseColor', 
        gltfPath = 'extensions/KHR_materials_pbrSpecularGlossiness/diffuseTexture', 
        maxChannels = 4,
        semantics = { 'Albedo': 'RGB', 'AlphaMask': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'ORM',
        gltfPath = 'pbrMetallicRoughness/metallicRoughnessTexture',
        maxChannels = 3,
        semantics = { 'Roughness': 'G', 'Metallic': 'B' },
        bcFormat = 'BC7'),
    TextureParams(
        name = 'SpecularGlossiness',
        gltfPath = 'extensions/KHR_materials_pbrSpecularGlossiness/specularGlossinessTexture',
        maxChannels = 4,
        semantics = { 'SpecularColor': 'RGB', 'Glossiness': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'Normal',
        gltfPath = 'normalTexture',
        maxChannels = 3,
        semantics = { 'Normal': 'RGB' },
        bcFormat = 'BC7'),
    TextureParams(
        name = 'Occlusion',
        gltfPath = 'occlusionTexture',
        maxChannels = 1,
        semantics = { 'Occlusion': 'R' },
        bcFormat = 'BC4'),
    TextureParams(
        name = 'Emissive',
        gltfPath = 'emissiveTexture',
        maxChannels = 3,
        semantics = { 'Emissive': 'RGB' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'Transmission',
        gltfPath = 'extensions/KHR_materials_transmission/transmissionTexture',
        maxChannels = 1,
        semantics = { 'Transmission': 'R' },
        bcFormat = 'BC4'),
]

def get_file_basename(fileName):
    head, tail = os.path.split(fileName)
    return tail if tail else head

def get_file_stem(fileName):
    return os.path.splitext(get_file_basename(fileName))[0]

@dataclass
class ManifestState:
    textures: List[Dict[str, Any]]
    numChannels: int = 0

def get_node_by_path(root, path):
    for part in path.split('/'):
        root = root.get(part, None)
        if root is None:
            return None
    return root

def add_texture_to_manifest(material, manifest: ManifestState, texture: TextureParams, texturesNode, imagesNode, newImageIndex,
                            gltfDir, manifestDir):
    materialTextureNode = get_node_by_path(material, texture.gltfPath)
    if materialTextureNode is None:
        return
    textureIndex = materialTextureNode['index']
    imageIndex = texturesNode[textureIndex]['source']
    image = imagesNode[imageIndex]

    try:
        imageUri = image['uri']
    except KeyError:
        imageName = image['name']
        print(f'  WARNING: Image "{imageName}" doesn\'t have a URI attribute, skipping.')
        print( '    Note: Only "glTF Separate" format models are supported.')
        return

    imageUri = os.path.join(gltfDir, imageUri)
    if (imageUri.endswith('.dds')):
        foundReplacement = False
        for ext in ('.png', 'jpg', '.jpeg', '.tga', '.bmp'):
            newUri = os.path.splitext(imageUri)[0] + ext
            if os.path.exists(newUri):
                imageUri = newUri
                foundReplacement = True
                break
        if not foundReplacement:
            print(f'  WARNING: Couldn\'t find a non-DDS replacement for {imageUri}, skipping.')
            return
    elif not os.path.exists(imageUri):
        print(f'  WARNING: {imageUri} does not exist, skipping.')
        return
    
    print(f'  {texture.name}: {get_file_basename(imageUri)}')
    
    imagePathRelativeToManifest = os.path.relpath(imageUri, manifestDir)

    if texture.name == 'Occlusion':
        # Try to merge the occlusion texture slice with a previously created roughness-metalness slice
        for entry in manifest.textures:
            if entry['fileName'] == imagePathRelativeToManifest:
                entry['semantics'].update(texture.semantics)
                materialTextureNode['index'] = entry['newTextureIndex']
                return
    
    # A new texture node will be appended to the end of the GLTF texture list
    newTextureIndex = len(texturesNode)

    manifestEntry = {
        'fileName': imagePathRelativeToManifest,
        'name': texture.name,
        'isSRGB': texture.sRGB,
        'bcFormat': texture.bcFormat,
        'newTextureIndex': newTextureIndex # Store this temporarily for texture merging, deleted later
    }

    semantics = copy.deepcopy(texture.semantics)

    # Find the name of the semantic that uses the Alpha channel, if any
    alphaSemantic = None
    for name, channels in semantics.items():
        if channels == 'A':
            alphaSemantic = name

    # Get the number of channels in the original image
    with PIL.Image.open(imageUri) as img:
        numChannels = len(img.getbands())
        # If there is such semantic, see if the image has an alpha channel.
        # Delete the semantic if there is no alpha channel.
        if numChannels < 4 and alphaSemantic is not None:
            del semantics[alphaSemantic]

    
    manifestEntry['semantics'] = semantics
    
    if numChannels > texture.maxChannels:
        manifestEntry['channelSwizzle'] = 'RGBA'[:texture.maxChannels]
        numChannels = texture.maxChannels

    firstChannel = manifest.numChannels
    manifestEntry['firstChannel'] = firstChannel
    if manifest.numChannels + numChannels > NTC_MAX_CHANNELS:
        print(f'  WARNING: Texture {texture.name} doesn\'t fit into the NTC {NTC_MAX_CHANNELS}-channel limit, skipping.')
        return

    manifest.textures.append(manifestEntry)
    manifest.numChannels += numChannels

    channelSlice = list(range(firstChannel, firstChannel + numChannels))

    # Replace the original texture index in the material with a new one
    materialTextureNode['index'] = newTextureIndex

    # Generate the new texture descriptor
    textureSliceNode = {
        'sampler': texturesNode[textureIndex]['sampler'],
        'source': imageIndex, # Pointer to the original, non-NTC image
        'extensions': {
            NV_TEXTURE_SWIZZLE_EXTENSION_NAME: {
                'options': [
                    {
                        'source': newImageIndex, 
                        'channels': channelSlice
                    }
                ]
            }
        }
    }
    texturesNode.append(textureSliceNode)

@dataclass
class MaterialDefinition:
    modelFileName: str
    materialName: str
    materialIndexInModel: int
    manifestTextures: List[Dict[str, Any]]
    references: Optional["MaterialDefinition"] = None
    manifestFileName: str = None
    ntcFileName: str = None
    newImageNode: Any = None

materialDefinitions : List[MaterialDefinition] = []
materialCountsPerModel : Dict[str, int] = {}

def process_gltf_file(inputFileName):
    print(f'\nProcessing {inputFileName}...')

    with open(inputFileName, 'r') as infile:
        gltf = json.load(infile)

    if NV_TEXTURE_SWIZZLE_EXTENSION_NAME in gltf.get('extensionsUsed', []):
        print(f'The asset already uses {NV_TEXTURE_SWIZZLE_EXTENSION_NAME}, skipping.')
        return

    try:    
        materialsNode = gltf['materials']
    except KeyError:
        print('The asset doesn\'t have any materials, skipping.')
        return

    materialCountsPerModel[inputFileName] = len(materialsNode)

    try:    
        texturesNode = gltf['textures']
        imagesNode = gltf['images']
    except KeyError:
        print('The asset doesn\'t have any textures, skipping.')
        return
    
    # Go over the materials in the GLTF file, create a manifest file and a compression task for each material.
    for materialIndex, material in enumerate(materialsNode):
        materialName = material.get('name', f'Material')

        print()
        print(f'Material: "{materialName}"')

        gltfDir = os.path.dirname(inputFileName)

        manifest = ManifestState(textures = [])

        # Add all supported textures that are present in the material to the manifest.
        for texture in textureTypes:
            add_texture_to_manifest(material, manifest, texture, texturesNode,
                                    imagesNode, len(imagesNode), gltfDir, manifestDir = args.output)

        if len(manifest.textures) == 0:
            print(f'No textures, skipping.')
            continue

        # Remove temporary items from the manifest
        for entry in manifest.textures:
            del entry['newTextureIndex']

        # Generate a node for the new NTC image, with no URI for now - it will be patched in
        # after global material de-duplication.
        newImageNode = {
            'mimeType': NTC_MIME_TYPE,
            'uri': None
        }

        imagesNode.append(newImageNode)

        matdef = MaterialDefinition(
            modelFileName = inputFileName,
            materialName = materialName,
            materialIndexInModel = materialIndex,
            manifestTextures = manifest.textures,
            newImageNode = newImageNode)
        
        materialDefinitions.append(matdef)

    return gltf

gltfObjects = {}
for inputFileName in args.inputFiles:
    gltfObjects[inputFileName] = process_gltf_file(inputFileName)
print()

materialCountsPerName = {}
def deduplicate_material_name(matdef: MaterialDefinition):
    # Deduplicate material names.
    # Sometimes, multiple glTF materials have the same name, and we need distinct names to associate
    # a single .ntc file with each material. Append _{count} to duplicate names.
    if matdef.materialName in materialCountsPerName:
        # Get the previous count and update the dictionary
        count = materialCountsPerName[matdef.materialName]
        count += 1
        materialCountsPerName[matdef.materialName] = count

        # Print the message, rename the material.
        newMaterialName = f'{matdef.materialName}_{count}'
        print(f'Renaming "{matdef.materialName}" to "{newMaterialName}"')
        matdef.materialName = newMaterialName
    else:
        # This is not a duplicate name yet, store a count of 1.
        materialCountsPerName[matdef.materialName] = 1


tasks : List[ntc.Arguments] = []

for materialIndex in range(len(materialDefinitions)):
    matdef = materialDefinitions[materialIndex]
    for searchIndex in range(materialIndex):
        matdef2 = materialDefinitions[searchIndex]
        if matdef.manifestTextures == matdef2.manifestTextures:
            print(f'Material [{get_file_basename(matdef.modelFileName)} / {matdef.materialName}] uses the same textures as '
                  f'[{get_file_basename(matdef2.modelFileName)} / {matdef2.materialName}], merging.')
            matdef.references = matdef2
            break

    if matdef.references is None:
        deduplicate_material_name(matdef)

        matdef.manifestFileName = os.path.join(args.output, f'{matdef.materialName}.manifest.json')
        matdef.ntcFileName = os.path.join(args.output, f'{matdef.materialName}.ntc')
        ntcFileName = matdef.ntcFileName

        with open(matdef.manifestFileName, 'w') as manifestFile:
            # Package the textures into a full manifest
            manifest = { 'textures': matdef.manifestTextures }
            # Save the JSON file
            json.dump(manifest, manifestFile, indent=2)

        if args.skipExisting and os.path.exists(matdef.ntcFileName):
            print(f'Output file {matdef.ntcFileName} exists, skipping.')
            continue

        task = ntc.Arguments(
            tool=args.tool,
            bitsPerPixel=args.bitsPerPixel,
            compress=True,
            decompress=True,
            generateMips=True,
            graphicsApi='vk',
            loadManifest=matdef.manifestFileName,
            maxBitsPerPixel=args.maxBitsPerPixel,
            optimizeBC=True,
            saveCompressed=matdef.ntcFileName,
            targetPsnr=args.targetPsnr,
            customArguments=args.args
        )
        
        tasks.append(task)
    else:
        ntcFileName = matdef.references.ntcFileName
        
    ntcFileNameRelativeToGltf = os.path.relpath(ntcFileName, os.path.dirname(matdef.modelFileName))
    # Use forward slashes for the relative path (on Windows)
    ntcFileNameRelativeToGltf = ntcFileNameRelativeToGltf.replace(os.sep, '/')
    # Patch the new NTC file name into the GLTF image node
    matdef.newImageNode['uri'] = ntcFileNameRelativeToGltf


def deduplicate_gltf_ntc_images(gltf):
    imagesNode = gltf['images']
    ntcFileToFirstImage = {}
    imageIndex = 0
    imageMapping = list(range(len(imagesNode)))
    imagesToDelete = []
    
    for image in imagesNode:
        # Validate that all image nodes have either a URI or a buffer view
        uri = image.get('uri')
        bufferView = image.get('bufferView')
        assert uri is not None or bufferView is not None

        # Find duplicate NTC images based on URI, add their indices to the deletion list
        if image.get('mimeType') == NTC_MIME_TYPE:
            firstOccurence = ntcFileToFirstImage.get(uri)
            if firstOccurence is None:
                ntcFileToFirstImage[uri] = imageIndex
            else:
                imagesToDelete.append(imageIndex)
                imageMapping[imageIndex] = firstOccurence

        imageIndex += 1

    if len(imagesToDelete) == 0:
        return
    
    # Delete the duplicate images
    deletedCount = 0
    for imageIndex in imagesToDelete:
        del imagesNode[imageIndex - deletedCount]

        # Shift the image mapping down for all images above this one
        for idx2 in range(len(imageMapping)):
            if imageMapping[idx2] >= imageIndex - deletedCount:
                imageMapping[idx2] -= 1

        deletedCount += 1

    # Patch the texture nodes with new image indices
    for textureNode in gltf['textures']:
        textureNode['source'] = imageMapping[textureNode['source']]

        extensionsNode = textureNode.get('extensions')
        if not extensionsNode:
            continue

        swizzleNode = extensionsNode.get(NV_TEXTURE_SWIZZLE_EXTENSION_NAME)
        if swizzleNode:
            for optionNode in swizzleNode['options']:
                optionNode['source'] = imageMapping[optionNode['source']]
        
        ddsNode = extensionsNode.get('MSFT_texture_dds')
        if ddsNode:
            ddsNode['source'] = imageMapping[ddsNode['source']]

def deduplicate_gltf_ntc_textures(gltf):
    texturesNode = gltf['textures']
    hashToFirstTexture = {}
    textureIndex = 0
    textureMapping = list(range(len(texturesNode)))
    texturesToDelete = []
    
    for texture in texturesNode:
        swizzleExt = texture.get('extensions', {}).get(NV_TEXTURE_SWIZZLE_EXTENSION_NAME)
        if swizzleExt is not None:
            assert len(swizzleExt['options']) == 1
            for optionNode in swizzleExt['options']:
                swizzleHash = optionNode['source']
                bitOffset = 16
                for ch in optionNode['channels']:
                    swizzleHash = swizzleHash ^ (max(ch, 0) << bitOffset)
                    bitOffset += 4

                firstOccurence = hashToFirstTexture.get(swizzleHash)
                if firstOccurence is None:
                    hashToFirstTexture[swizzleHash] = textureIndex
                else:
                    texturesToDelete.append(textureIndex)
                    textureMapping[textureIndex] = firstOccurence

        textureIndex += 1

    if len(texturesToDelete) == 0:
        return
    
    # Delete the duplicate textures
    deletedCount = 0
    for textureIndex in texturesToDelete:
        del texturesNode[textureIndex - deletedCount]

        # Shift the mapping down for all images above this one
        for idx2 in range(len(textureMapping)):
            if textureMapping[idx2] >= textureIndex - deletedCount:
                textureMapping[idx2] -= 1

        deletedCount += 1

    # Patch the material nodes with new texture indices
    for materialNode in gltf['materials']:
        for texture in textureTypes:
            materialTextureNode = get_node_by_path(materialNode, texture.gltfPath)
            if materialTextureNode is None:
                continue
            textureIndex = materialTextureNode['index']
            materialTextureNode['index'] = textureMapping[textureIndex]
    


for inputFileName in args.inputFiles:
    if inputFileName not in materialCountsPerModel:
        continue
    
    gltf = gltfObjects[inputFileName]
    if gltf is None:
        continue

    extensionsNode = gltf.get('extensionsUsed')
    if extensionsNode is None:
        extensionsNode = []
        gltf['extensionsUsed'] = extensionsNode
    extensionsNode.append(NV_TEXTURE_SWIZZLE_EXTENSION_NAME)

    deduplicate_gltf_ntc_images(gltf)
    deduplicate_gltf_ntc_textures(gltf)

    parts = os.path.splitext(inputFileName)
    newFileName = f'{parts[0]}.ntc.gltf'
    with open(newFileName, 'w') as newFile:
        json.dump(gltf, newFile, indent = 4)


if args.dryRun:
    sys.exit(0)

if len(tasks) == 0:
    print('Nothing to compress, exiting.')
    sys.exit(0)

maxNtcFilePathLen = max([len(args.saveCompressed) for args in tasks])

def task_ready(task, result, originalTaskCount, tasksCompleted):

    # Delete the manifest file unless instructed to keep it.
    if not args.keepManifests:
        os.unlink(task.loadManifest)

    # Print the status output
    bpp = f'{result.bitsPerPixel:.2f}' if result.bitsPerPixel else 'N/A'
    psnr = f'{result.overallPsnr:.2f}' if result.overallPsnr else 'N/A'
    warning = ' <-- WARNING' if (result.overallPsnr and result.overallPsnr <= 10) else ''
    print(f'[{tasksCompleted:2} of {originalTaskCount:2}] {task.saveCompressed:{maxNtcFilePathLen}}'
            f' : {bpp} bpp, {psnr} dB{warning}')

print()
print('Starting compression...')

terminated = ntc.process_concurrent_tasks(tasks, args.devices, task_ready)

if terminated:
    sys.exit(2)