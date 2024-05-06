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
    Array3D<byte>* dataOutShadowMap = nullptr;
};

struct PipelineStage {
    PipelineState* state = nullptr;
    std::thread* thread  = nullptr;
    bool ready = false;
    int id = -1;
};

class Pipeline{
public:
    Pipeline(Raster& rasterIn, Raster& rasterOut, const uint tileSize, const uint rayPerPoint, const uint tileBuffer, const float exaggeration, const uint maxBounces, const float bias, const bool isShadowMap, const uint sm_rays_per_dir, const uint sm_nb_dirs);
    ~Pipeline();
    bool step();

private:
    PipelineStage stages[NB_PIPELINE_STAGES]; 
    Raster& rasterIn;
    Raster& rasterOut;

    static void waitForNextStep(PipelineStage* stage);
    static void readData(PipelineStage* stage, const Raster* rasterIn, const uint tileSize, const uint tileBuffer, const bool isShadowMap, const uint sm_nb_dirs);
    static void initTile(PipelineStage* stage, const float pixelSize, const float exaggeration=1.0, const uint maxBounces=0);
    static void trace(PipelineStage* stage, const uint rayPerPoint, const float bias, const bool isShadowMap, const uint sm_rays_per_dir, const uint sm_nb_dirs);
    static void writeData(PipelineStage* stage, const Raster* const rasterOut, const bool isShadowMap, const uint sm_nb_dirs);
};