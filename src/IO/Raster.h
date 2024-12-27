#pragma once

#include "gdal_priv.h"
#include "../utils/definitions.cuh"

#include <iostream>

class Raster{

public:
    Raster(const char* const filename, const Raster* const copyFrom = nullptr);
    ~Raster();

    uint getHeight() const { return height; }
    uint getWidth()  const { return width;  }
    float getPixelSize() const { return pixelSize; }
    float getNoDataValue() const { return noDataValue; }
    void printInfos();
    void readData(float* data,  uint x, uint y, uint width, uint height) const;
    void writeData(float* data, uint x, uint y, uint width, uint height) const;
    Raster clone(const char* const newFileName);


    static void writeTile(float* data, int x, int y, int width, int height, const Raster* originalRaster, const char* outputPath) {
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        GDALDataset* outputDataset = driver->Create(outputPath, width, height, 1, GDT_Float32, nullptr);
        if (!outputDataset) {
            std::cerr << "Failed to create GeoTiff file at " << outputPath << std::endl;
            return;
        }

        const double* transform = &originalRaster->geoTransform[0];
        double outTransform[6];
        outTransform[0] = transform[0] + x * transform[1] + y * transform[2];
        outTransform[1] = transform[1];
        outTransform[2] = transform[2];
        outTransform[3] = transform[3] + x * transform[4] + y * transform[5];
        outTransform[4] = transform[4];
        outTransform[5] = transform[5];

        outputDataset->SetGeoTransform(outTransform);
        outputDataset->SetProjection(originalRaster->dataset->GetProjectionRef());

        GDALRasterBand* band = outputDataset->GetRasterBand(1);
        band->SetNoDataValue(originalRaster->noDataValue);
        if (band->RasterIO(GF_Write, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0) != CE_None) {
            std::cerr << "Failed to write data to " << outputPath << std::endl;
        }
        GDALClose(outputDataset);
    }

private:
    GDALDataset*    dataset;
    GDALRasterBand* dataBand;
    uint width;
    uint height;
    float pixelSize;
    double noDataValue;
    double geoTransform[6];
};