<h1 align="center">DSM Occlusion</h1>
<h3 align="center">Fast, ray-traced, occlusion for raster models</h3>

![DSM Occlusion example](/assets/banner.jpg)

<br>
<h2 align="center">Key features</h2>

 - **GPU accelerated**  
   Using CUDA reduces the computation time by order of magnitudes compared to other programs.
 - **Automated tiling**  
   Large files are automatically split into tiles, with configurable size and margins (buffers to avoid border effects).
 - **Handles large models**  
   Given enough disk space, can process files of several terabytes.
 - **Ray-traced rendering**  
   Occlusion is computed using ray-tracing, simulating the real behavior of light rays.

<br>
<h2 align="center">Installation</h2>

This software requires a computer with an Nvidia graphic card and its driver installed.  
Supported operating systems are Windows and Linux.

### Windows

1. Download the  [release](https://github.com/Dolpic/DSM-Occlusion/releases/tag/latest-main)  from this repository and run the executable contained in it from a command line.

### Linux

1. Install the package of GDAL version 30 for your distribution. The package is generally named *gdal* or *libgdal*.

2. Download the latest release from this repository and run the executable contained in it from a command line.

<br>
<h2 align="center">Usage</h2>

- ***-i inputFile***  
*mandatory parameter*  
Path to the input file. Must be a file supported by GDAL like .tif

- ***-o outputFile***  
*optional parameter*, default value : output.tif  
Path to the output file, where the result will be written

- ***-r raysPerPixel***  
*optional parameter*, default value : 256  
Number of rays to launch for each pixel. Increasing the parameter decreases the noise and increases the render quality and rendering time.  
  - lower than 256 = low quality (noise is very noticeable)
  - lower than 1024 = medium quality (noise is noticeable, but limited)
  - lower than 2048 = high quality (noise is almost not noticeable)

- ***-t tileSize***   
*optional parameter*, default value : 2000, in pixels  
The input file is processed in square tiles. This parameter controls the tile side length.  
Smaller tiles are computed faster, but lead to a larger buffer surface (see next parameter) and may create border error if the buffers are not large enough.  
Larger tiles are more computation heavy for the GPU.  
The optimal value for this parameter strongly depends on your graphic card, don't hesitate to try multiple sizes.

- ***-b tileBuffer*** 
*optional parameter*, default value : 1/3 of the tile size, in pixels  
The input file is processed in square tiles. To avoid border error, tiles are overlapping. This parameter controls the tile overlapping amount (in pixels) in each direction. 

- ***-e exaggeration***   
*optional parameter*, default value : 1.0
This parameter scales all input values by the given factor. A value higher than 1.0 makes the shadows darker, revealing more details. 

- ***-B maximumBounces***
*optional parameter*, default value : 0
The maximum number of times a ray can bounce on the geometry before the ray is considered not reaching the sky.
The higher the value, the more accurate and luminous the result will be. *Increase this value will drastically increase the processing time!*. A value of 0 (the default) leads to the same result as computing the sky view factor.

- ***--bias biasValue***
*optional parameter*, default value : 1.0
Bias applied to rays distribution. A value of 1 means no bias, and the rays are uniformly sampled in all directions.
Values greater than 1 bias rays toward the horizon, revealing small terrain details but darkening already occluded areas.
Values smaller than 1 bias rays toward the zenith, brightening dark areas and discarding terrain details.

- ***--tiled***
*optional parameter*
Instead of creating one output file and writing everything at once inside it, creates the folder *./output_tiles* and render each tile separately inside it. This is especially usefull when rendering very large files.

- ***--startTile***
*optional parameter*, default value : 0
Id of the first tile to be processed. All tiles before it will be ignored and not rendered. This is usefull combined with the *--tiled* options to resume a render at a given tile.

- ***--info***  
*optional parameter*  
Prints information about the GDAL driver and the graphic card. This is purely informative. 

Example : `DSM_Occlusion -i /path/to/input.tif -o /path/to/output.tif -r 1024`

<br>
<h2 align="center">Build instructions<br>(setting up a development environment)</h2>

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
      - You can automatically install GDAL by running the following command in a terminal :
      `osgeo4w-setup.exe -q -r -s https://download.osgeo.org/osgeo4w/v2/ -P gdal -P gdal-devel`

2. Clone this repository

3. Open a "x64 Native Tools Command Prompt for VS 2022" (or your current VS version). Can be found by searching for it in the Windows.

4. Navigate to the repository root and generate the project using the command `cmake -B ./build`
   Then build the project using `cmake --build ./build --config Release`. The executable is located in ./build/Release.

5. To start the executable, add the folder containing the GDAL .dll files to your PATH environment variable. If installed using OSGeo4W, the default path is *C:\OSGeo4W\bin*.

Note : You can also generate the executable by running `msbuild ./build/DSM_Occlusion.sln /p:Configuration=Release /p:Platform=x64 /p:OutDir=<output_directory>`

### Linux

1. Install the following packages for your distribution :
    - **Git**, **CMake**, **CUDA Toolkit** and **GDAL**

    - For the following distributions :
      - Arch : `sudo pacman -Syu git cuda cmake gdal`
      - Ubuntu : 
        - Install CMake and GDAL : `sudo apt-get install git cmake libgdal-dev`
        - Download CUDA Toolkit from the official site (https://developer.nvidia.com/cuda-downloads). Select your preferred installer type and follow the instructions for the *Base Installer*.
      - Other distributions :
        - **Git and CMake** : these are widely available from any distribution
        - **GDAL** : Depends on the distribution. The development package is often called gdal-devel or libgdal-dev. 
        - **CUDA Toolkit** : Download it from the official site (https://developer.nvidia.com/cuda-downloads). Select your preferred installer type and follow the instructions for the *Base Installer*.

2. Clone this repository and navigate to its root.

3. Generate the project using the command `cmake -B ./build`.  
  Then build the project using `cmake --build ./build --config Release`

### macOS

- macOS is currently not supported. Contributions are welcome !

#### Troubleshooting

- ***Unable to find GDAL library*** during build  
*Reason* : The GDAL library files (.lib on Windows, or .so on Linux) failed to be located.  
*Solution* : Make sure GDAL is installed properly. If GDAL library is not installed in the expected directory (*/usr/lib/* on Linux, or *C:/OSGeo4W/lib/* on Windows) you can specify GDAL library location by defining GDAL_LIB_DIR during CMake generation. Example : `cmake -B ./build -DGDAL_LIB_DIR=<your_gdal_path>`

- ***gdal_priv.h not found*** during build  
*Reason* : The GDAL header files (.h) failed to be located.  
*Solution* : Make sure GDAL is installed properly. If GDAL headers are not installed in the expected directory (*/usr/include/* on Linux, or *C:/OSGeo4W/include/* on Windows) you can specify GDAL headers location by defining GDAL_INCLUDE_DIR during CMake generation. Example : `cmake -B ./build -DGDAL_INCLUDE_DIR=<your_gdal_path>`

- ***Unsupported gpu architecture*** during build  
*Reason* : By default, the project is compiled for the hardware on the current machine. This error happens if CUDA was not able to find the current machine architecture (for example, when the current machine doesn't have a GPU).  
*Solution* : You can specify the GPU architecture to use by defining CMAKE_CUDA_ARCHITECTURES during generation. Example : `cmake -B ./build -DCMAKE_CUDA_ARCHITECTURES=75`. See [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).

- ***Missing CUDA driver*** or ***CUDA driver version is insufficient*** when running the executable  
*Reason* : Your CUDA driver is either missing or too old.  
*Solution* : Install the latest [CUDA driver](https://www.nvidia.com/Download/index.aspx) for your system and graphic card.

