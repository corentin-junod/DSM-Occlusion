#include "Raster.h"

#include <cpl_error.h>
#include <iostream>
#include <stdexcept>

Raster::Raster(const char* const filename, const Raster* const copyFrom, const bool isShadowMap, const float scale, const uint sm_nb_bands) : scale(scale) {
    CPLPushErrorHandler(CPLQuietErrorHandler);
    GDALAllRegister();
    if(copyFrom == nullptr){
        dataset = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
        if(dataset == nullptr){
            throw std::runtime_error("Unable to open input file : " + std::string(filename));
        }
    }else if(isShadowMap){
        const uint xSize = copyFrom->dataset->GetRasterBand(1)->GetXSize()*scale;
        const uint ySize = copyFrom->dataset->GetRasterBand(1)->GetYSize()*scale;
        dataset = copyFrom->dataset->GetDriver()->Create(filename, xSize, ySize, sm_nb_bands, GDT_Byte, nullptr);
        dataset->SetSpatialRef(copyFrom->dataset->GetSpatialRef());
        dataset->SetProjection(copyFrom->dataset->GetProjectionRef());
        double geoTransform[6];
        copyFrom->dataset->GetGeoTransform(geoTransform);
        geoTransform[1] /= scale;
        geoTransform[5] /= scale;
        dataset->SetGeoTransform(geoTransform);
        dataset->SetMetadata(copyFrom->dataset->GetMetadata());
    }else{
        dataset = copyFrom->dataset->GetDriver()->CreateCopy(filename, copyFrom->dataset, false, NULL, NULL, NULL ); 
    }
    dataBand = dataset->GetRasterBand(1);
    width  = dataBand->GetXSize();
    height = dataBand->GetYSize();
    noDataValue = dataBand->GetNoDataValue();

    double geoTransform[6];
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

void Raster::readData(float* data, const uint x, const uint y, const uint width, const uint height) const {
    const CPLErr result = dataBand->RasterIO(GF_Read, x, y, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file reading";
    }
}

void Raster::writeData(float* data, const uint x, const uint y, const uint width, const uint height) const {
    const CPLErr result = dataBand->RasterIO(GF_Write, x, y, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file writing";
    }
}

void Raster::writeDataShadowMap(Array3D<byte>& data, const uint x, const uint y, const uint width, const uint height) const {
    for(uint band=0; band<data.depth(); band++){
        const CPLErr result = dataset->GetRasterBand(band + 1)->RasterIO(GF_Write, x*scale, y*scale, width*scale, height*scale, data.atDepth(band), width, height, GDT_Byte, 0, 0);
        if(result == CE_Failure){
            std::cout << "Error during file writing";
        }
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