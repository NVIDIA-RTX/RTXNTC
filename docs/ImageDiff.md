# ImageDiff

ImageDiff is a command line application for comparing pairs image files in any supported format (PNG, TGA, JPG, BMP, EXR, DDS including block-compressed formats) and reporting the difference between files in each pair as MSE and PSNR numbers.

It is used by the NTC SDK test infrastructure.

## Building ImageDiff

ImageDiff is configured with the rest of the NTC SDK when the `NTC_WITH_TESTS` CMake variable is `ON` (it is by default).

## Running ImageDiff

A typical command line for ImageDiff looks like this:

```sh
imagediff <file1.dds> <file2.dds> <file3.dds> <file4.dds>...
```

For the full set of command line options, please run `imagediff --help`.
