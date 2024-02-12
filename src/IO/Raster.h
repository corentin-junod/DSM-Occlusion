#pragma once

#include "gdal_priv.h"
#include "../utils/definitions.cuh"

class Raster{

public:
    Raster(const char* const filename, const Raster* const copyFrom = nullptr);
    ~Raster();

    uint getHeight() const { return height; }
    uint getWidth()  const { return width;  }
    float getPixelSize() const;
    void printInfos();
    void readData(float* data,  const uint x, const uint y, const uint width, const uint height) const;
    void writeData(float* data, const uint x, const uint y, const uint width, const uint height) const;
    Raster clone(const char* const newFileName);

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    uint width;
    uint height;
};