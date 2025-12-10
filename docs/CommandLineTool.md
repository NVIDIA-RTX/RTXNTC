# NTC Command-Line Tool

The `ntc-cli` tool exposes all of the compression and decompression functionality available in the NTC library. It can load sets of images from a directory, optionally generate or load mips, and compress them into NTC files with given compression settings, providing a PSNR of the compression result compared to the input. It can also load compressed NTC files, decompress them, and save images into a directory. Encoding BCn textures is also supported.

The command line operation is based on specifying a set of actions to be performed, such as loading images or compressed files, compressing or decompressing, and saving output files. Optionally, input files or directory can be specified as positional arguments.

## Positional arguments

Inputs can be specified using positional arguments, and can be one of the following:

- One or several image files
- Directory with images, equivalent to `--loadImages`
- Manifest file with a `.json` extension, equivalent to `--loadManifest`
- Compressed texture set with an `.ntc` extension, equivalent to `--loadCompressed`

## Actions

The following actions are supported. The order in which they appear on the command line doesn't matter.

Argument | Description 
---------|------------
`--loadImages <path>` | Load all images from a directory into a texture set.
`--loadMips` | Load mip levels for images from the `mips` subdirectory of the directory specified in `--loadImages`.
`--loadManifest <file>` | Load images described by a [manifest file](Manifest.md) into a texture set.
`--loadCompressed <file>` | Load a compressed texture set from a file.
`-i <path>`, `--saveImages <path>` | Save all images from the texture set into a directory. See also `--imageFormat` and `--bcFormat`.
`--saveManifest <file>` | Save the final manifest into a JSON file. Useful to generate a template manifest from a directory to edit it later.
`--saveMips` | Save all mip levels with the images when processing `--saveImages`. Also affects DDS files produced for BCn textures.
`-o`, `--saveCompressed <file>` | Save the compressed texture set into a file.
`-g`, `--generateMips` | Generate all mip levels (1 and above) for the texture set from mip 0.
`-d`, `--describe` | Print out the texture set dimensions, textures, and other parameters.
`-c`, `--compress` | Perform NTC compression of the texture set.
`-D`, `--decompress` | Perform NTC decompression of the previously compressed or loaded texture set. <br> The decompression method depends on other parameters, default is CUDA. <br> The `--decompress` parameter is implied if decompression is required for other actions.
`--optimizeBC` | Perform BC7 transcoding optimization if any textures are set to use BC7.
`--readManifestFromStdin` | Load channel images using a [JSON manifest](Manifest.md) passed through standard input (stdin).
`--listCudaDevices` | Prints out the list of CUDA devices available in the system. Use `--cudaDevice <N>` to select a specific device.
`--listAdapters` | Prints out the list of Vulkan or DX12 adapters available in the system, requires `--vk` or `--dx12`. <br> Use `--adapter <N>` to select a specific one. When using CUDA operations, a matching adapter is selected automatically.

For the full set of options, please use `ntc-cli --help`.

## Loading and saving MIP chains

MIP chains can be loaded from individual image files when `--loadImages` and `--loadMips` are specified, in which case the `mips/` subdirectory next to the original textures is scanned for files named `<base-name>.<mip>.png` or similar. Alternatively, exact file names can be specified in the [manifest](Manifest.md).

For example, files for a material compatible with `--loadMips` could be named like this:

```sh
material-name/
    albedo.png # MIP 0 images
    normal.png
    mips/
        albedo.01.png
        normal.01.png
        albedo.02.png
        normal.02.png
        ...
```

When `--generateMips` is specified, MIP levels 1 and above are generated automatically before compression. They can also be saved to files in the same layout described above when `--saveMips` is specified.

## Examples

Compressing all textures from a directory to a specific bit rate:
```sh
ntc-cli --loadImages <input-dir> \
        --generateMips \
        --compress \
        --bitsPerPixel <value> \ # can be between 1.0 and 20
        --decompress
        --saveCompressed <file.ntc>

# Short version
ntc-cli <input-dir> -g -c -b <value> -D -o <file.ntc>
```

Loading textures using a manifest and compressing the texture set to reach a specific PSNR:
```sh
ntc-cli --loadManifest <manifest.json> \
        --generateMips \
        --compress \
        --targetPsnr <value> \ # PSNR in dB, reasonable values 25-50
        --decompress \
        --saveCompressed <file.ntc>

# Short version
ntc-cli <manifest.json> -g -c -p <value> -D -o <file.ntc>
```

Decompressing a texture set and saving the textures as TGA files:
```sh
ntc-cli --loadCompressed <file.ntc> \
        --decompress \
        --saveImages <output-dir> \
        --imageFormat tga \
        --bcFormat none

# Short version
ntc-cli <file.ntc> -i <output-dir> -F tga -B none
```

Decompressing a texture set, encoding the textures into automatically chosen BCn formats, and saving the textures as DDS files:
```sh
ntc-cli --vk \ # or --dx12 - needed for BC encoding
        --loadCompressed <file.ntc> \
        --saveImages <output-dir> \
        --bcFormat auto

# Short version
ntc-cli --vk <file.ntc> -i <output-dir> -B auto
```

Converting textures from common image formats into BC7:
```sh
ntc-cli --vk \ # or --dx12 - needed for BC encoding
        --loadImages <input-dir> \
        --bcFormat bc7 \
        --saveImages <output-dir>

# Short version
ntc-cli --vk <input-dir> -i <output-dir> -B bc7
```

Applying BC7 transcoding optimization to an existing texture set:
```sh
ntc-cli --vk \ # or --dx12
        --loadCompressed <input.ntc> \
        --bcFormat bc7 \ # overrides the target BC format for all textures in the set
        --optimizeBC \
        --saveCompressed <output.ntc>

# Short version
ntc-cli --vk <input.ntc> -B bc7 --optimizeBC -o <output.ntc>
```

Getting information about a texture set file:
```sh
ntc-cli --loadCompressed <file.ntc> \
        --describe
        
# Short version
ntc-cli <file.ntc> -d
```

Getting information about GPUs available in the system:
```
ntc-cli --listCudaDevices
ntc-cli --listAdapters --vk
ntc-cli --listAdapters --dx12
```
