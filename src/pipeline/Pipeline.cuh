#pragma once

#include "../utils/definitions.cuh"
#include "../utils/logging.h"
#include "../IO/Raster.h"
#include "../array/Array.cuh"
#include "../tracer/Tracer.cuh"

#include <thread>

constexpr int NB_PIPELINE_STAGES = 4;

struct PipelineState {
    const Array2D<float>* dataIn = nullptr;
    Array2D<float>* dataOut      = nullptr;
    Tracer* tracer               = nullptr;
    Extent extent;
    int id = -1;
    uint width  = 0;
    uint height = 0;
    int x = 0;
    int y = 0;
    bool hasData  = false;
    bool finished = false;
};

struct PipelineStage {
    PipelineState* state = nullptr;
    std::thread* thread  = nullptr;
    bool ready = false;
    int id = -1;
};

class Pipeline{
public:
    Pipeline(Raster& rasterIn, Raster& rasterOut, const uint tileSize, const uint rayPerPoint, const uint tileBuffer);
    ~Pipeline();
    bool step();

private:
    PipelineStage stages[NB_PIPELINE_STAGES]; 
    Raster& rasterIn;
    Raster& rasterOut;

    static void waitForNextStep(PipelineStage* stage);
    static void readData(PipelineStage* stage, const Raster* rasterIn, const uint tileSize, const uint tileBuffer);
    static void initTile(PipelineStage* stage, const float pixelSize);
    static void trace(PipelineStage* stage, const uint rayPerPoint);
    static void writeData(PipelineStage* stage, const Raster* const rasterOut);
};