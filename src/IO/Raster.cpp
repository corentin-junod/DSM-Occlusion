#include "Raster.h"

#include <cpl_error.h>
#include <cpl_conv.h>
#include <stdexcept>

Raster::Raster(const char* const filename, const Raster* const copyFrom) {
    CPLSetConfigOption("GTIFF_SRS_SOURCE", "EPSG");
    CPLPushErrorHandler(CPLQuietErrorHandler);
    GDALAllRegister();
    if(copyFrom == nullptr){
        dataset = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
        if(dataset == nullptr){
            throw std::runtime_error("Unable to open input file : " + std::string(filename));
        }
    }else{
        dataset = copyFrom->dataset->GetDriver()->CreateCopy(filename, copyFrom->dataset, false, NULL, NULL, NULL ); 
    }
    dataBand = dataset->GetRasterBand(1);
    width  = dataBand->GetXSize();
    height = dataBand->GetYSize();
    noDataValue = dataBand->GetNoDataValue();

    dataset->GetGeoTransform(geoTransform);
    pixelSize = geoTransform[1];
}
 
Raster::~Raster() {
    double min=0, max=0, mean=0, dev=0;
    dataBand->ComputeStatistics(true, &min, &max, &mean, &dev, NULL, NULL);
    dataBand->SetStatistics(min, max, mean, dev);
    dataBand->SetNoDataValue(-9999.0);
    dataBand->FlushCache();
    dataset->FlushCache(true);
    GDALClose((GDALDatasetH)dataset);
}

void Raster::readData(float* data, uint x, uint y, uint width, uint height) const {
    const CPLErr result = dataBand->RasterIO(GF_Read, x, y, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file reading";
    }
}

void Raster::writeData(float* data, uint x, uint y, uint width, uint height) const {
    const CPLErr result = dataBand->RasterIO(GF_Write, x, y, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file writing";
    }
}

void Raster::printInfos(){
    int tileXSize, tileYSize;
    dataBand->GetBlockSize(&tileXSize, &tileYSize);

    int hasMin, hasMax;
    double minMax[] = {dataBand->GetMinimum(&hasMin), dataBand->GetMaximum(&hasMax)};
    if(!hasMin||!hasMax){
        GDALComputeRasterMinMax(dataBand, true, minMax);
    }

    std::cout
        << "***** GDAL Driver information *****"
        << "\nDriver used : " << dataset->GetDriverName()
        << "\nNumber of bands : " << dataset->GetRasterCount()
        << "\nTiles of " << tileXSize << "x" << tileYSize
        << "\nType=" << GDALGetDataTypeName(dataBand->GetRasterDataType())
        << "\nColorInterp="<< GDALGetColorInterpretationName(dataBand->GetColorInterpretation())
        << "\nMin : "<< minMax[0] << " Max : " << minMax[1]
        << "\nNumber of overviews : " << dataBand->GetOverviewCount()
        << "\nScale : " << dataBand->GetScale()
        << "\nPixel size (X axis) : " << getPixelSize()
        << "\n***********************************\n";
}
