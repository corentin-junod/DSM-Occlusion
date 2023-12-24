#pragma once

#include <iostream>
#include "gdal_priv.h"

class Raster{

public:
    Raster(const char* const inputName, const char* const outputName);
    ~Raster();

    unsigned int getHeight() const {return height;}
    unsigned int getWidth()  const {return width;}
    float* getData()         const {return data;}
    void printInfos();
    void writeData();

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    float*          data;
    unsigned int    width;
    unsigned int    height;
};