#include "Raster.h"
#include <iostream>

Raster::Raster(const char* const inputName, const char* const outputName) {
    GDALAllRegister();
    GDALDataset* source = (GDALDataset*) GDALOpen(inputName, GA_ReadOnly);

    dataset = source->GetDriver()->CreateCopy(outputName, source, false, NULL, NULL, NULL ); 
    GDALClose((GDALDatasetH)source);

    dataBand = dataset->GetRasterBand(1);
    width  = dataBand->GetXSize();
    height = dataBand->GetYSize();
    data = (float*) CPLMalloc(width*height*sizeof(float));
    const CPLErr result = dataBand->RasterIO(GF_Read, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file reading";
    }
}

Raster::~Raster() {
    GDALClose((GDALDatasetH)dataset);
    CPLFree(data);
}

void Raster::writeData(){
    std::cout << data[1] << '\n';
    const CPLErr result = dataBand->RasterIO(GF_Write, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);
    dataBand->FlushCache();
    Raster::printInfos();
    //int bands[] = {1};
    //const CPLErr result = dataset->RasterIO(GF_Write, 0, 0, width, height, data, width, height, GDT_Float32, 0, bands, 0, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file writing";
    }
}

void Raster::printInfos(){
    int tileXSize, tileYSize;
    dataBand->GetBlockSize(&tileXSize, &tileYSize);

    int hasMin, hasMax;
    double minMax[] = {dataBand->GetMinimum(&hasMin), dataBand->GetMaximum(&hasMax)};
    //if(!hasMin||!hasMax){
        GDALComputeRasterMinMax(dataBand, true, minMax);
    //}

    std::cout<< "Driver used : " << dataset->GetDriverName() << '\n'
             << "Number of bands : " << dataset->GetRasterCount() << '\n'
             << "Tiles of " << tileXSize << "x" << tileYSize << '\n'
             << "Type=" << GDALGetDataTypeName(dataBand->GetRasterDataType()) << '\n'
             << "ColorInterp="<< GDALGetColorInterpretationName(dataBand->GetColorInterpretation()) << '\n'
             << "Min : "<< minMax[0] << " Max : " << minMax[1] << '\n'
             << "Number of overviews : " << dataBand->GetOverviewCount() << '\n';
             //<< "Number of color in table entry : " << dataBand->GetColorTable()->GetColorEntryCount() << '\n';
}