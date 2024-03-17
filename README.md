<h1 align="center">DSM Occlusion</h1>
<h3 align="center">Fast, ray-traced, occlusion for raster models</h3>

![DSM Occlusion example](/assets/neuchatel_banner.jpg)

<br>
<h2 align="center">Key features</h2>

 - **GPU accelerated**  
   Using CUDA, it reduces the computation time by order of magnitudes compared to other programs.
 - **Automated tiling**  
   Large files are automatically split into tiles, with configurable size and margins (buffers to avoid border effects).
 - **Handles large models**  
   Given enough disk space, can process files of several terabytes.
 - **Ray-traced algorithm**  
   Occlusion is computed using ray-tracing, simulating the real behaviour of light rays. *Currently, rays are not reflected on objects, leading to the same result as a sky-view factor*.

<br>
<h2 align="center">Installation</h2>

Download the latest release and 

<br>
<h2 align="center">Usage</h2>

- ***-i inputFile***  
*mandatory parameter*  
Path to the input file. Must be a file supported by GDAL like .tif

- ***-o outputFile***  
*optional parameter*, default value : output.tif  
Path to the output file, where the result will be written

- ***-r raysPerPixel***  
*optional parameter*, default value : TODO  
Number of rays to launch for each pixel. Increasing the parameter decreases the noise and increases the render quality and rendering time.  
  - lower than 256 = low quality (noise is very noticeable)
  - lower than 1024 = medium quality (noise is noticeable, but limited)
  - lower than 2048 = high quality (noise is almost not noticeable)


- ***-t tile size (in pixels)***   
*optional parameter*, default value : TODO  
The input file is processed in square tiles. This parameter controls the tile side length.  
Smaller tiles are computed faster, but lead to a larger buffer surface (see next parameter) and may create border error if the buffers are not large enough.  
Larger tiles are more computation heavy for the GPU.

- ***-b tile buffer (in pixels)***   
*optional parameter*, default value : 1/3 of the tile size  
The input file is processed in square tiles. To avoid border error, tiles are overlapping. This parameter controls the tile overlapping amount (in pixels) in each direction. 

- ***--info***  
*optional parameter*  
Prints information about the GDAL driver and the graphic card. Does not impact the output. 

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
      - You can either automatically install GDAL by running the following command in a terminal :
      `osgeo4w-setup.exe -q -r -s https://download.osgeo.org/osgeo4w/v2/ -P gdal -P gdal-devel`

2. Clone this repository

3. Open a "x64 Native Tools Command Prompt for VS 2022" (or your current VS version). Can be found by searching for it in the Windows.

4. Navigate to the repository root and generate the project using the command `cmake -B ./build`
   Then build the project using `cmake --build ./build --config Release`. The executable is located in ./build/Release.

5. To start the executable, add the folder containing the GDAL .dll files to your PATH environment variable. If installed using OSGeo4W, the default path is *C:\OSGeo4W\bin*.

Note : You can also generate the executable by running `msbuild ./build/DSM_Occlusion.sln /p:Configuration=Release /p:Platform=x64 /p:OutDir=<output_directory>`

#### Troubleshooting

- ***Unable to find GDAL library*** during build  
*Reason* : The GDAL library files (.dll) failed to be located  
*Solution* : Make sure GDAL is installed properly and the folder containing GDAL's .dll file is in the PATH environment variable.



### Linux

1. Install the following packages for your distribution :
    - **Git**, **CMake**, **CUDA Toolkit** and **GDAL**

    - For the following distributions :
      - Arch : `sudo pacman -Syu git cuda cmake gdal`
      - Ubuntu : 
        - Add the ubuntugis repository : `sudo add-apt-repository ppa:ubuntugis/ppa`
        - Update the package list and install : `sudo apt-get update && sudo apt-get install cmake libgdal-dev`
        - Download CUDA Toolkit from the official site (https://developer.nvidia.com/cuda-downloads). Select your preferred installer type and proceed through the installation process.
      - Other distributions :
        - **Git and CMake** : these are widely available form any distribution
        - **GDAL** : Depends on the distribution. The development package is often called gdal-devel or libgdal-dev. 
        - **CUDA Toolkit** : Download it from the official site (https://developer.nvidia.com/cuda-downloads). Select your preferred installer type and proceed through the installation process.

2. Clone this repository and navigate to its root.

3. Generate the project using the command `cmake -B ./build`.  
  Then build the project using `cmake --build ./build --config Release`

### macOS

- macOS is currently not supported. Contributions are welcome !

