#pragma once

#include <iostream>
#include "gdal_priv.h"

class Raster{

public:
    Raster(const char* const inputName, const char* const outputName);
    ~Raster();

    unsigned int getHeight() const {return height;}
    unsigned int getWidth()  const {return width;}
    void printInfos();
    void readData(float* data) const;
    void writeData(float* data) const;

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    unsigned int    width;
    unsigned int    height;
};