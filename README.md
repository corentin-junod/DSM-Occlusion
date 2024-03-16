<h1 align="center">DSM Occlusion</h1>
<h3 align="center">Fast, ray-traced, occlusion for raster models</h3>

![DSM Occlusion example](/assets/neuchatel_banner.jpg)

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

## Build instructions (setting up a development environment)

### Windows

1. Make sure you have installed the following :
    - [Git](https://git-scm.com/downloads) (used to clone this repository)
    - [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/fr/free-developer-offers/) 
      - Install "Desktop development with C++" including :
          - "*MSVC C++ x64/x86 build tools*"
          - "*C++ CMake tools for Windows*"
          - Windows SDK for your current Windows installation (ex. "*Windows 11 SDK*" for Windows 11)
    - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - GDAL / GDAL-devel
      - The easiest way to get GDAL is through [OSGeo4W](https://trac.osgeo.org/osgeo4w)
      - You can either automatically install GDAL by running the following command in a terminal :
      `osgeo4w-setup.exe -q -r -s https://download.osgeo.org/osgeo4w/v2/ -P gdal -P gdal-devel`

2. Clone this repository

3. Open a "x64 Native Tools Command Prompt for VS 2022" (or your current VS version). Can be found by searching for it in the Windows.

4. Navigate to the repository root and generate the project using the command `cmake -B ./build`

5. Build the project using `msbuild ./build/DEM_Occlusion.sln`  
To build in x64 release, use `msbuild ./build/DEM_Occlusion.sln /p:Configuration=Release /p:Platform=x64`

6. By default, the executable is generated in *./build/Debug/*, *./build/Release/*, *./build/x64/Release/* or *./build/x64/Release/* based on the build parameters. To specify a custom output folder, add a `/p:OutDir=<output_directory>` parameter.

7. To start the executable, add the folder containing the GDAL .dll files to your PATH environment variable. If installed using OSGeo4W, the default path is *C:\OSGeo4W\bin*.

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
