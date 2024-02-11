#include "Raster.h"

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
    dataBand->ComputeStatistics(false, &min, &max, &mean, &dev, NULL, NULL);
    dataBand->SetStatistics(min, max, mean, dev);
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

void Raster::printInfos(){
    int tileXSize, tileYSize;
    dataBand->GetBlockSize(&tileXSize, &tileYSize);

    int hasMin, hasMax;
    double minMax[] = {dataBand->GetMinimum(&hasMin), dataBand->GetMaximum(&hasMax)};
    if(!hasMin||!hasMax){
        GDALComputeRasterMinMax(dataBand, true, minMax);
    }

    std::cout<< "Driver used : " << dataset->GetDriverName() << '\n'
             << "Number of bands : " << dataset->GetRasterCount() << '\n'
             << "Tiles of " << tileXSize << "x" << tileYSize << '\n'
             << "Type=" << GDALGetDataTypeName(dataBand->GetRasterDataType()) << '\n'
             << "ColorInterp="<< GDALGetColorInterpretationName(dataBand->GetColorInterpretation()) << '\n'
             << "Min : "<< minMax[0] << " Max : " << minMax[1] << '\n'
             << "Number of overviews : " << dataBand->GetOverviewCount() << '\n';
             //<< "Number of color in table entry : " << dataBand->GetColorTable()->GetColorEntryCount() << '\n';
}