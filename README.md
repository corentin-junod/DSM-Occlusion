<h1 align="center">DEM Occlusion</h1>
<h3 align="center">Fast, ray-traced, occlusion for raster models</h3>

![DEM Occlusion example](/assets/neuchatel_banner.jpg)

## Key features

 - GPU accelerated
 - Automated tiling
 - Handles arbitrary large models


## Installation

Download the latest release and 


### Usage

- ***-i inputFile***  
*mandatory parameter*  
Path to the input file. Must be a file supported by GDAL like .tif

- ***-o outputFile***  
*optional parameter*, default value : output.tif
Path to the output file, where the result will be written

- ***-r raysPerPixel***  
*optional parameter*, default value : TODO  
Number of rays to launch for each pixel. Increasing the parameter increases the render quality and rendering time.  
  - lower than 256 = low quality
  - lower than 1024 = medium quality
  - lower than 2048 = high quality 


- ***-t tile size (in pixels)***   
*optional parameter*, default value : TODO  
The input file is processed by square tiles. This parameter controls the tile side length.  
Smaller tiles are computed faster, but lead to a larger buffer surface (see next parameter) and may create border error if the buffers are not large enough.  
Larger tiles are more computation heavy for the GPU.

- ***-b tile buffer (in pixels)***   
*optional parameter*, default value : 1/3 of the tile size  
The input file is processed by square tiles. To avoid border error, tiles are overlapping. This parameter controls the tile overlapping amount (in pixels) in each direction. 

- ***--info***  
*optional parameter*  
Prints information about the GDAL driver and the graphic card. Does not impact the output.  

## Build instructions (setting up development environment)

### Windows

1. Make sure you have the following software installed
    - [Git](https://git-scm.com/downloads) (used to clone this repository)
    - [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/fr/free-developer-offers/) 
      - Install "Desktop development with C++" including :
          - MSVC C++ x64/x86 build tools
          - C++ CMake tools for Windows
          - Windows SDK for your current Windows installation (named "Windows X SDK")

    - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - GDAL
      - The easiest way to get GDAL is to install [OSGeo4W](https://trac.osgeo.org/osgeo4w)
      - You can either install GDAL by using the installer manually or by running the command `osgeo4w-setup.exe -q -r -s https://download.osgeo.org/osgeo4w/v2/ -P gdal`

2. Clone this repository

3. Open a "x64 Native Tools Command Prompt for VS 2022" (or your current VS version) and navigate to the repository root.

4. Navigate to the repository root and generate the project using the command `cmake -B ./build`

5. Build the project using 

#### Troubleshooting

- ***Unable to find GDAL library*** during build  
*Reason* : The GDAL library files (.dll) failed to be located  
*Solution* : Make sure GDAL is installed properly and the folder containing GDAL's .dll file is in the PATH environment variable.



### Linux

1. Install the following packages for your distribution :
    - Git
    - CMake
    - CUDA toolkit
    - GDAL

    - Command line for various distributions :
      - Arch : `sudo pacman -Syu git cuda cmake gdal`
      - Ubuntu : TODO

2. Clone this repository

3. Navigate to the repository root and build the project using the command `cmake -B ./build`

### MacOS

- MacOS is currently not supported. Contributions are welcome !
