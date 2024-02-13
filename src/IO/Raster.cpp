#include "Raster.h"

#include <iostream>

Raster::Raster(const char* const filename, const Raster* const copyFrom) {
    GDALAllRegister();
    if(copyFrom == nullptr){
        dataset = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
    }else{
        dataset = copyFrom->dataset->GetDriver()->CreateCopy(filename, copyFrom->dataset, false, NULL, NULL, NULL ); 
    }
    dataBand = dataset->GetRasterBand(1);
    width  = dataBand->GetXSize();
    height = dataBand->GetYSize();
}
 
Raster::~Raster() {
    double min=0, max=0, mean=0, dev=0;
    dataset->ClearStatistics();
    dataBand->SetNoDataValue(-1);
    dataBand->ComputeStatistics(false, &min, &max, &mean, &dev, NULL, NULL);
    dataBand->SetStatistics(min, max, mean, dev);
    dataBand->FlushCache();
    dataset->FlushCache(true);
    GDALClose((GDALDatasetH)dataset);
}

float Raster::getPixelSize() const {
    double geoTransform[6];
    dataset->GetGeoTransform(geoTransform);
    return geoTransform[1];
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
      //<< "Number of color in table entry : " << dataBand->GetColorTable()->GetColorEntryCount() << '\n';
}