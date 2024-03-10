# Fast, ray-traced, occlusion for raster models

## Key features

 - GPU accelerated
 - Automated tiling
 - Handles arbitrary large models

## Requirements & Build instructions

### Requirements

### Build instructions

#### Windows

1. Make sure you have the following software installed
    - [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/fr/free-developer-offers/) 
      - Install "Desktop development with C++" including :
          - MSVC C++ x64/x86 build tools
          - C++ CMake tools for Windows
          - Windows SDK for your current Windows installation (named "Windows X SDK")

    - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. Open a Developer Command Prompt for Visual Studio and navigate to the repository root.

3. Build the project using the command `cmake -B ./build`


#### Linux

1. Install the following packages for your distribution :
    - CMake
    - CUDA toolkit

    - Command line for various distributions :
      - Arch : `sudo pacman -Syu cuda cmake`
      - Ubuntu : TODO

#### MacOS

- MacOS is currently not supported. Contributions are welcome !


## Usage

### Parameters
- -i inputFile 
- [-o outputFile] 
- [-r raysPerPixel] 
- [-t tile size (in pixels)] 
- [-b tile buffer (in pixels)] 
- --info