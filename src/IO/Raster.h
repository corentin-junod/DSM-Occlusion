#pragma once

#include "gdal_priv.h"
#include "../utils/definitions.cuh"
#include "../array/Array.cuh"

class Raster{

public:
    Raster(const char* const filename, const Raster* const copyFrom = nullptr, const bool isShadowMap = false, const float scale = 1.0, const uint sm_nb_dirs = 0);
    ~Raster();

    uint getHeight() const { return height; }
    uint getWidth()  const { return width;  }
    float getPixelSize() const { return pixelSize; }
    float getNoDataValue() const { return noDataValue; }
    void printInfos();
    void readData(float* data,  const uint x, const uint y, const uint width, const uint height) const;
    void writeData(float* data, const uint x, const uint y, const uint width, const uint height) const;
    void writeDataShadowMap(Array3D<byte>& data, const uint x, const uint y, const uint width, const uint height) const;

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    uint width;
    uint height;
    const float scale;
    float pixelSize;
    float noDataValue;
    bool isReadOnly;
};