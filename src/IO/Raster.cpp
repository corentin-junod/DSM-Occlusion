#include "Raster.h"

#include <cpl_error.h>
#include <iostream>
#include <stdexcept>

uint SM_DIR_PER_BAND = 3;
uint SM_ELEVATION_SIZE = 5;
GDALDataType SM_BAND_TYPE = GDT_UInt16;

Raster::Raster(const char* const filename, const Raster* const copyFrom, const bool isShadowMap, const float scale, const uint sm_nb_dirs) : scale(scale) {
    CPLPushErrorHandler(CPLQuietErrorHandler);
    GDALAllRegister();
    if(copyFrom == nullptr){
        dataset = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
        if(dataset == nullptr){
            throw std::runtime_error("Unable to open input file : " + std::string(filename));
        }
        isReadOnly = true;
    }else if(isShadowMap){

        const uint xSize = copyFrom->dataset->GetRasterBand(1)->GetXSize()*scale;
        const uint ySize = copyFrom->dataset->GetRasterBand(1)->GetYSize()*scale;

        char** papszOptions = NULL;
        papszOptions = CSLSetNameValue(papszOptions, "STATISTICS", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "BIGTIFF", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
        papszOptions = CSLSetNameValue(papszOptions, "BIGTIFF", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "NUM_THREADS", "ALL_CPUS");
        papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "BLOCKXSIZE", "400");
        papszOptions = CSLSetNameValue(papszOptions, "BLOCKYSIZE", "400");

        dataset = ((GDALDriver*)GDALGetDriverByName("GTiff"))->Create(filename, xSize, ySize, std::ceil(sm_nb_dirs / SM_DIR_PER_BAND), SM_BAND_TYPE, papszOptions);
        dataset->SetSpatialRef(copyFrom->dataset->GetSpatialRef());
        dataset->SetProjection(copyFrom->dataset->GetProjectionRef());
        dataset->SetMetadata(copyFrom->dataset->GetMetadata());
        double geoTransform[6];
        copyFrom->dataset->GetGeoTransform(geoTransform);
        geoTransform[1] /= scale;
        geoTransform[5] /= scale;
        dataset->SetGeoTransform(geoTransform);
        isReadOnly = false;
    }else{
        dataset = copyFrom->dataset->GetDriver()->CreateCopy(filename, copyFrom->dataset, false, NULL, NULL, NULL ); 
        isReadOnly = false;
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
    if (!isReadOnly) {
        double min = 0, max = 0, mean = 0, dev = 0;
        dataBand->ComputeStatistics(true, &min, &max, &mean, &dev, NULL, NULL);
        dataBand->SetStatistics(min, max, mean, dev);
        dataBand->SetNoDataValue(noDataValue);
        dataBand->FlushCache();
        dataset->FlushCache(true);
    }
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

void Raster::writeDataShadowMap(Array3D<byte>& data, const uint x0, const uint y0, const uint width, const uint height) const {
    Array2D<short> resultArray = Array2D<short>(width, height);
    uint rasterBand = 1;
    uint currentShift = 0;

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            resultArray.at(x, y) = 0;
        }
    }
    for(uint band=0; band<data.depth(); band++){
        for (uint y = 0; y < height; y++) {
            for (uint x = 0; x < width; x++) {
                resultArray.at(x, y) += data.atDepth(band)[x + width * y] << (SM_ELEVATION_SIZE * currentShift);
            }
        }

        currentShift++;
        if (currentShift == SM_DIR_PER_BAND || band == data.depth() - 1) {
            const CPLErr result = dataset->GetRasterBand(rasterBand)->RasterIO(GF_Write, x0*scale, y0*scale, width*scale, height*scale, resultArray.begin(), width, height, SM_BAND_TYPE, 0, 0);
            if (result == CE_Failure) {
                std::cout << "Error during file writing";
            }
            currentShift = 0;
            rasterBand++;
            for (uint y = 0; y < height; y++) {
                for (uint x = 0; x < width; x++) {
                    resultArray.at(x, y) = 0;
                }
            }
        }
    }
}

void Raster::printInfos(){
    int tileXSize, tileYSize, hasMin, hasMax;
    dataBand->GetBlockSize(&tileXSize, &tileYSize);
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