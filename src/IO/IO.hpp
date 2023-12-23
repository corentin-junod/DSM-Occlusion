#pragma once

#include "gdal_priv.h"
#include <iostream>

struct Raster{
    GDALRasterBand* rasterBand;
    unsigned int width;
    unsigned int height;
    float* data;
};

Raster copyAndGetRaster(const char* const input, const char* const output);
void writeRaster(const Raster raster);