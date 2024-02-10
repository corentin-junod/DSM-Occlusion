#pragma once

#include <iostream>
#include "gdal_priv.h"

class Raster{

public:
    Raster(const char* const filename, const Raster* const copyFrom = nullptr);
    ~Raster();

    unsigned int getHeight() const {return height;}
    unsigned int getWidth()  const {return width;}
    void printInfos();
    void readData(float* data,  const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height) const;
    void writeData(float* data, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height) const;
    Raster clone(const char* const newFileName);

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    unsigned int    width;
    unsigned int    height;
};