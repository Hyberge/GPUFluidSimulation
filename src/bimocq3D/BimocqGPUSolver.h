#ifndef BIMOCQGPUSOLVER_H
#define BIMOCQGPUSOLVER_H
#include "../include/array.h"
#include "../include/fluid_buffer3D.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <string>
#include "../include/vec.h"
#include "../utils/pcg_solver.h"
#include "../include/array3.h"
#include "../utils/GeometricLevelGen.h"
#include "GPU_Advection.h"
#include <chrono>
#include "../utils/color_macro.h"
#include "Mapping.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/volumeMeshTools.h"

class BimocqGPUSolver {
public:
    BimocqGPUSolver() = default;
    BimocqGPUSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, VirtualGpuMapper *mymapper);
    ~BimocqGPUSolver() = default;

    void advance(int framenum, float dt);

    float semilagAdvect(float cfldt, float dt);

    float getCFL();

    float emitSmoke(int framenum, float dt);

    float addBuoyancy(float dt);

    float diffuseField(float *field, float nu, float dt);

    void initBoundary();

    void velocityReinitialize();

    void scalarReinitialize();

    // smoke parameter
    float _alpha;
    float _beta;

    // AMGPCG solver data
    SparseMatrixd matrix;
    std::vector<double> rhs;
    std::vector<double> pressure;
    buffer3Dc _b_desc;

    // simulation data
    uint   CellNumberX, CellNumberY, CellNumberZ;
    float  CellSize;
    float  MaxVelocity;
    float  Viscosity;
    float* VelocityU = nullptr, *VelocityV = nullptr, *VelocityW = nullptr;
    float* VelocityUInit = nullptr, *VelocityVInit = nullptr, *VelocityWInit = nullptr;
    float* VelocityUPrev = nullptr, *VelocityVPrev = nullptr, *VelocityWPrev = nullptr;
    float* VelocityUTemp = nullptr, *VelocityVTemp = nullptr, *VelocityWTemp = nullptr;

    float* TempSrcU = nullptr, *TempSrcV = nullptr, *TempSrcW = nullptr;

    float* Density = nullptr, *DensityInit = nullptr, *DensityPrev = nullptr, *DensityTemp = nullptr;
    float* Temperature = nullptr, *TemperatureInit = nullptr, *TemperaturePrev = nullptr, *TemperatureTemp = nullptr;

    float* boundaryDesc = nullptr;

    uint VelocityBufferSizeX = 0;
    uint VelocityBufferSizeY = 0;
    uint VelocityBufferSizeZ = 0;
    uint ScaleFieldSize = 0;

    VirtualGpuMapper *GpuSolver;

    MapperBaseGPU VelocityAdvector;
    MapperBaseGPU ScalarAdvector;

    int vel_lastReinit = 0;
    int scalar_lastReinit = 0;
    Scheme sim_scheme;

    std::vector<Emitter> sim_emitter;
};

#endif  // BIMOCQGPUSOLVER_H