# RTX Neural Texture Compression SDK Change Log

## 0.9.2 BETA

### LibNTC

- Fixed the `FileUnrecognized` error that happened in some cases when the binary chunk size was not a multiple of 4.

### Command-Line Tool

- Added `--keepFileNames` option that preserves the full original texture file names in the NTC file.
- Fixed the `--matchBcPsnr` feature on Vulkan.
- Fixed Vulkan initialization errors when using the tool over SSH.


## 0.9.1 BETA

### LibNTC

- Improved accuracy of DP4a based decompression and made it more consistent across GPUs and graphics APIs.

### Command-Line Tool

- Added `--no-dithering` option that disables color dithering when saving images after graphics API decompression.

### Testing Infrastructure

- Fixed reporting of PSNR values for mip chains in ImageDiff.
- Updated the BC transcoding tests to use much lower tolerances.

### Other

- Updated the Donut framework to reduce the number of external library dependencies.

### Known Issues

Please see the [Known Issues](/README.md#known-issues) section of the main Readme file.


## 0.9.0 BETA

### Breaking Changes

- Changed the shape of the image decoder network (Multi-Layer Perceptron, MLP) to a smaller one.
    * This change significantly improves inference performance at a minor quality cost (about -0.25 dB PSNR).
- Replaced the BC7 encoding optimization solution to store mode and partition values for each block.
    * This change makes the BC7 encoding process 2X-6X faster at the expense of about 0.3 bits per pixel per BC7 texture in storage (when GDeflate compression is used).
- NTC files compressed with earlier versions of LibNTC are incompatible with version 0.9.0.

### LibNTC

- Added support for changing the number and sizes of the MLP hidden layers individually with just a few constants.
- Added support for compressing the BC7 mode buffers and slices of the latent texture with GDeflate.
- Added support for loading NTC files using smaller MLP hidden layers than what the library is compiled for.
- Added functions for decompressing GDeflate-compressed data on the CPU using [libdeflate](https://github.com/NVIDIA/libdeflate) and on the GPU using [`VK_NV_memory_decompression`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_memory_decompression.html).
    * Note: GDeflate decompression on DX12 is also supported through DirectStorage, but that has to be implemented on the application side.
- Added CMake option `NTC_DEBUG_SHADERS` to make profiling easier.
- Refactored the code to explicitly use the same weight layout definition structures in all places.

### Explorer

- Added a mode for comparing two image files to each other when the `--compare` command line option is used.
- Added display of per-texture and per-channel PSNR values in the results window.
- Added support for saving the manifest file including all changes made in the GUI.
- Added support for specifying per-channel loss function scales in the GUI and from the manifest.
- Changed the default graphics API to Vulkan.

### Rendering Sample

- Added decompression of GDeflate-compressed buffers on the GPU with `VK_NV_memory_decompression` or DirectStorage (disabled by default, use `--gpuGDeflate`).

### Command-Line Tool

- Added decompression of GDeflate-compressed buffers on the GPU for testing (disabled by default, use `--gpuGDeflate`).
- Added support for reading the manifest from STDIN.
- Added support for saving the manifest file automatically generated from input image files.
- Added support for specifying per-channel loss function scales from the manifest.

### Testing Infrastructure

- Added ImageDiff, a small command line tool for comparing multiple pairs of images, including block compressed DDS textures.
- Added arguments to `test.py` to disable execution of some tests.
- Added functional tests for BC compression.

### Known Issues

Please see the [Known Issues](/README.md#known-issues) section of the main Readme file.

## 0.8.0 BETA

### Breaking Changes

- Replaced the two latent arrays that used custom bit packing (read through a `ByteAddressBuffer`) with a single BGRA4 texture array (sampled through a `Texture2DArray`).
    * This change significantly improves inference performance and simplifies the shader code at the expense of a minor quality loss.
    * The integrations of Inference on Load and Inference on Sample need to be updated.
- Removed the network version selection, only one configuration is available now.
- Removed the legacy inference shader versions that didn't use DP4a and the versions that used CoopVec with a full Int8 pipeline. FP8 inference is now the default, and the FP8 weights are always produced during compression.
- Removed the functions enabling operation with partially loaded latent images.
- NTC files compressed with earlier versions of LibNTC are incompatible with version 0.8.0.

### LibNTC

- Added support for specifying loss function scales per-channel.
- Added header file `<libntc/shaders/Bindings.h>` declaring all binding indices for the compute shaders provided by the library.
- Fixed decompression errors on small images with unusual dimensions, e.g. 94x94.
- Fixed the compiler warnings on template usage in CUDA code ([#3](https://github.com/NVIDIA-RTX/RTXNTC-Library/issues/3))
- Fixed the duplicate copies of weights ([#4](https://github.com/NVIDIA-RTX/RTXNTC-Library/issues/4))
- Fixed the grid dimensions in the quantizer ([#2](https://github.com/NVIDIA-RTX/RTXNTC-Library/issues/2))
- Improved optimizer performance.
- Redefined the known latent shapes to fit the new latent storage configurations available with BGRA4 latents.
- Simplified the math for positional encoding.
- Switched the CoopVec shader builds on DXIL to use DXC directly.
- Updated the compilers: DXC to v1.8.2505.1, Slang to v2025.18.2.

### Explorer

- Added support for calculating the compression ratio over BCn formats specified in the manifest.
- Added sliders for loss function scaling per-texture.
- Show compression progress when the window is out of focus.

### Command-Line Tool

- Fixed BC compression for use cases when the data comes from image files, not NTC.

## 0.7.2 BETA

### LibNTC

- Fixed multiple causes for crashes that happened when attempting compression of very large texture sets.
- Improved inference performance with Cooperative Vectors on DX12.

### Rendering Sample

- Added plots of frame and render time and tile counts.
- Improved performance of the NTC On Feedback mode by implementing stochastic feedback.
- Moved the DLSS integration into the Donut framework.

### Command-Line Tool

- Fixed integer overflows when compressing and decompressing very large texture sets.

## 0.7.1 BETA

### LibNTC

- Fixed Cooperative Vector inference on Intel Arc B-series GPUs.
- Minor code improvements.

## 0.7.0 BETA

### LibNTC

- Switched the DX12 Cooperative Vector implementation to use the Agility SDK Preview instead of the NVIDIA custom extensions.
- Moved the Cooperative Vector weight layout conversions to happen on the GPU.
- Added support for shuffling inference output channels to make channel mappings more flexible.
- Improved code quality around inference weight layouts.

### Rendering Sample

- Implemented a custom GLTF extension `NV_texture_swizzle` to define the NTC storage for materials.
- Improved the Inference on Feedback mode to transcode tiles in batches.
- Improved the Inference on Sample mode by replacing conditional texture channel usage with constant output channels.
- Added a display of the inference math versions that are being used for materials.

## 0.6.1 BETA

- Improved the Inference on Feedback mode to add support for standby tiles.
- Fixed the rendering mode display in the Renderer sample in the reference mode.
- Implemented handling of `VK_SUBOPTIMAL_KHR` in the SDK apps ([#3](https://github.com/NVIDIA-RTX/RTXNTC/issues/3))

## 0.6.0 BETA

Added the Inference on Feedback mode in the Renderer sample.

## 0.5.0 BETA

Initial release.