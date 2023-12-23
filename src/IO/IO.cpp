#include "IO.hpp"

Raster copyAndGetRaster(const char* const input, const char* const output){
    GDALAllRegister();

    GDALDataset* source = (GDALDataset*) GDALOpen(input, GA_ReadOnly);
    GDALDataset* dataFile = source->GetDriver()->CreateCopy(output, source, false, NULL, NULL, NULL );
    GDALClose((GDALDatasetH)source);

    std::cout<<"Loading input file..." << '\n';

    GDALRasterBand* dataBand = dataFile->GetRasterBand(1);

    int tileXSize, tileYSize;
    dataBand->GetBlockSize(&tileXSize, &tileYSize);

    int hasMin, hasMax;
    double minMax[] = {dataBand->GetMinimum(&hasMin), dataBand->GetMaximum(&hasMax)};
    if(!hasMin||!hasMax){
        GDALComputeRasterMinMax(dataBand, true, minMax);
    }

    std::cout<< "Number of bands : " << dataFile->GetRasterCount() << '\n'
             << "Tiles of " << tileXSize << "x" << tileYSize << '\n'
             << "Type=" << GDALGetDataTypeName(dataBand->GetRasterDataType()) << '\n'
             << "ColorInterp="<< GDALGetColorInterpretationName(dataBand->GetColorInterpretation()) << '\n'
             << "Min : "<< minMax[0] << " Max : " << minMax[1] << '\n'
             << "Number of overviews : " << dataBand->GetOverviewCount() << '\n';
             //<< "Number of color in table entry : " << dataBand->GetColorTable()->GetColorEntryCount() << '\n';

    unsigned int width = dataBand->GetXSize();
    unsigned int height = dataBand->GetYSize();
    float* data = (float*) CPLMalloc(width*height*sizeof(float)); // TODO unload with CPLFree()
    dataBand->RasterIO(GF_Read, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0); // TODO check return value
    return Raster{dataBand, width, height, data};
}

void writeRaster(const Raster raster){
    //TODO check return value
    raster.rasterBand->RasterIO(GF_Write, 0, 0, raster.width, raster.height, raster.data, raster.width, raster.height, GDT_Float32, 0, 0);
}