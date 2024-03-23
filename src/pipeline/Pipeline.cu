#include "Pipeline.cuh"

#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>

std::condition_variable cv;
std::atomic<int> nbFinished(0);
std::mutex mutex;

Pipeline::Pipeline(Raster& rasterIn, Raster& rasterOut, const uint tileSize, const uint rayPerPoint, const uint tileBuffer, const float exaggeration, const uint maxBounces): 
rasterIn(rasterIn), rasterOut(rasterOut){
    const uint nbTiles = (uint)std::ceil((float)rasterIn.getHeight()/tileSize) * (uint)std::ceil((float)rasterIn.getWidth()/tileSize);

    for(uint i=0; i<NB_PIPELINE_STAGES; i++){
        stages[i].state = new PipelineState();
        stages[i].id = i;
    }

    stages[0].thread = new std::thread(Pipeline::readData, &stages[0], &rasterIn, tileSize, tileBuffer);
    stages[1].thread = new std::thread(Pipeline::initTile, &stages[1], rasterIn.getPixelSize(), exaggeration, maxBounces);
    stages[2].thread = new std::thread(Pipeline::trace, &stages[2], rayPerPoint);
    stages[3].thread = new std::thread(Pipeline::writeData, &stages[3], &rasterOut);
}

Pipeline::~Pipeline(){
    debug_print("Stopping pipeline \n");
    for(uint i=0; i<NB_PIPELINE_STAGES; i++){
        stages[i].thread->join();
        delete stages[i].state;
        delete stages[i].thread;
    }
}

bool Pipeline::step(){
    while(nbFinished < NB_PIPELINE_STAGES){}
    std::lock_guard<std::mutex> lock(mutex);
    debug_print("NEXT STEP \n");

    PipelineState* state1Tmp = stages[0].state;
    stages[0].state = stages[3].state;
    stages[3].state = stages[2].state;
    stages[2].state = stages[1].state;
    stages[1].state = state1Tmp;

    nbFinished = 0;
    for(uint i=0; i < NB_PIPELINE_STAGES; i++){
        stages[i].ready = true;
    }
    cv.notify_all();
    return !stages[3].state->finished;
}

void Pipeline::waitForNextStep(PipelineStage* stage){
    debug_print("Thread " + std::to_string(stage->id) +" waiting\n");
    std::unique_lock<std::mutex> lock(mutex);
    nbFinished++;
    cv.wait(lock, [stage]{ return stage->ready; });
    debug_print("Thread " + std::to_string(stage->id) + " released \n");
    stage->ready = false;
}


void Pipeline::readData(PipelineStage* stage, const Raster* rasterIn, const uint tileSize, const uint tileBuffer){
    const float noDataValue = rasterIn->getNoDataValue();
    const uint nbTiles = (uint)std::ceil((float)rasterIn->getHeight()/tileSize) * (uint)std::ceil((float)rasterIn->getWidth()/tileSize);
    uint nbTileProcessed = 0;
    for(uint y=0; y<rasterIn->getHeight(); y+=tileSize){
        for(uint x=0; x<rasterIn->getWidth(); x+=tileSize){

            Pipeline::waitForNextStep(stage);
            PipelineState* state = stage->state;
            print_atomic("Processing tile " + std::to_string(nbTileProcessed+1) + "/" + std::to_string(nbTiles) + " (" + std::to_string(100*nbTileProcessed/nbTiles) + "%)...\n");

            state->id = nbTileProcessed;
            state->x = x;
            state->y = y;
            state->width  = std::min(tileSize, rasterIn->getWidth()-x);
            state->height = std::min(tileSize, rasterIn->getHeight()-y);
            state->extent.xMin = std::max(0, (int)x-(int)tileBuffer);
            state->extent.yMin = std::max(0, (int)y-(int)tileBuffer);
            state->extent.xMax = std::min(rasterIn->getWidth(), x+state->width+tileBuffer);
            state->extent.yMax = std::min(rasterIn->getHeight(),y+state->height+tileBuffer);

            if(state->dataIn != nullptr){
                delete state->dataIn;
            }
            if(state->dataOut != nullptr){
                delete state->dataOut; 
            }
            state->dataIn = new Array2D<float>(state->extent.xMax-state->extent.xMin, state->extent.yMax-state->extent.yMin);
            state->dataOut = new Array2D<float>(state->extent.xMax-state->extent.xMin, state->extent.yMax-state->extent.yMin);

            rasterIn->readData(state->dataIn->begin(), state->extent.xMin, state->extent.yMin, state->extent.xMax-state->extent.xMin, state->extent.yMax-state->extent.yMin);

            for(uint i=0; i<state->dataIn->size(); i++){
                if((*state->dataIn)[i] != noDataValue){
                    state->hasData = true;
                }
                (*state->dataOut)[i] = (*state->dataIn)[i];
            }

            nbTileProcessed++;
        }
    }
    debug_print("> Thread 1 finished...\n");
    for(uint i=3; i>0; i--){
        PipelineState* state = stage->state;
        state->finished = true;
        Pipeline::waitForNextStep(stage);
    }
    debug_print("> Thread 1 exit\n");
}

void Pipeline::initTile(PipelineStage* stage, const float pixelSize, const float exaggeration, const uint maxBounces){
    PipelineState* state = stage->state;
    while(!state->finished){
        Pipeline::waitForNextStep(stage);
        state = stage->state;
        if(state->hasData && state->id >= 0){
            debug_print("> Building BVH of tile " + std::to_string(state->id+1) + "...\n");
            if(state->tracer != nullptr){
                delete state->tracer;
            }
            state->tracer = new Tracer(*state->dataOut, pixelSize, exaggeration, maxBounces);
            state->tracer->init(false);
        }else if(state->id >= 0){
            cout() << "> Tile "<< state->id+1  <<" skipped because it had no data \n";
        }
    }
    debug_print("> Thread 2 finished...\n");
    Pipeline::waitForNextStep(stage);
    Pipeline::waitForNextStep(stage);
    debug_print("> Thread 2 exit\n");
}

void Pipeline::trace(PipelineStage* stage, const uint rayPerPoint){
    PipelineState* state = stage->state;
    while(!state->finished){
        Pipeline::waitForNextStep(stage);
        state = stage->state;
        if(state->hasData && state->id >= 0){
            debug_print("> Tracing tile " + std::to_string(state->id+1) + "...\n");
            state->tracer->trace(true, rayPerPoint);
        }
    }
    debug_print("> Thread 3 finished...\n");
    Pipeline::waitForNextStep(stage);
    debug_print("> Thread 3 exit\n");
}

void Pipeline::writeData(PipelineStage* stage, const Raster* const rasterOut){
    PipelineState* state = stage->state;
    while(!state->finished){
        Pipeline::waitForNextStep(stage);
        state = stage->state;
        if(state->hasData && state->id >= 0){
            debug_print("> Writing tile " + std::to_string(state->id+1) + "...\n");
            const float noDataValue = rasterOut->getNoDataValue();
            Array2D<float> dataCropped(state->width, state->height);
            uint i=0, j=0;
            for(int curY=state->extent.yMin; curY < state->extent.yMax; curY++){
                for(int curX=state->extent.xMin; curX < state->extent.xMax; curX++){
                    if(curY>=state->y && curY < state->y+(int)state->height && curX>=state->x && curX < state->x+(int)state->width){
                        dataCropped[i++] = ((*state->dataIn)[j] == noDataValue ? noDataValue : (*state->dataOut)[j]);
                    }
                    j++;
                }
            }
            rasterOut->writeData(dataCropped.begin(), state->x, state->y, state->width, state->height);
        }
    }
    debug_print("> Thread 4 finished...\n> Thread 4 exit\n");
}