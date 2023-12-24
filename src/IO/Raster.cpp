#include "Raster.h"

Raster::Raster(const char* const inputName, const char* const outputName) {
    GDALAllRegister();
    GDALDataset* source = (GDALDataset*) GDALOpen(inputName, GA_ReadOnly);

    dataset = source->GetDriver()->CreateCopy(outputName, source, false, NULL, NULL, NULL ); 
    GDALClose((GDALDatasetH)source);

    dataBand = dataset->GetRasterBand(1);
    width  = dataBand->GetXSize();
    height = dataBand->GetYSize();
}
 
Raster::~Raster() {
    dataBand->FlushCache();
    dataset->FlushCache(true);
    GDALClose((GDALDatasetH)dataset);
}

void Raster::readData(float* data) const{
    const CPLErr result = dataBand->RasterIO(GF_Read, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file reading";
    }
}

void Raster::writeData(float* data) const {
    const CPLErr result = dataBand->RasterIO(GF_Write, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);
    if(result == CE_Failure){
        std::cout << "Error during file writing";
    }

    dataset->ClearStatistics();
    double min=0, max=0, mean=0, dev=0;
    dataBand->ComputeStatistics(false, &min, &max, &mean, &dev, NULL, NULL);
    dataBand->SetStatistics(min, max, mean, dev);
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