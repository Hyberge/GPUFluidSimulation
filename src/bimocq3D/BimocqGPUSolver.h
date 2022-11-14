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
#include "BimocqSolver.h"

class BimocqGPUSolver {
public:
    BimocqGPUSolver() = default;
    BimocqGPUSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, Scheme inScheme, gpuMapper *mymapper);
    ~BimocqGPUSolver() = default;

    void advance(int framenum, float dt);

    void advanceBimocq(int framenum, float dt);

    void advanceReflection(int framenum, float dt);

    void semilagAdvect(float cfldt, float dt);

    float getCFL();

    void emitSmoke(int framenum, float dt);

    void addBuoyancy(float dt);

    void diffuseField(float *field, float *fieldTemp0, float *fieldTemp1, int ni, int nj, int nk, int iter, float nu, float dt);

    void projection();

    void initBoundary();

    void velocityReinitialize();

    void scalarReinitialize();

    void setSmoke(float drop, float raise, const std::vector<Emitter> &emitters);

    void outputResult(uint frame, string filepath);

    // smoke parameter
    float _alpha;
    float _beta;

    Scheme myscheme;

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
    float *VelocityU = nullptr, *VelocityV = nullptr, *VelocityW = nullptr;
    float *VelocityUInit = nullptr, *VelocityVInit = nullptr, *VelocityWInit = nullptr;
    float *VelocityUPrev = nullptr, *VelocityVPrev = nullptr, *VelocityWPrev = nullptr;
    float *VelocityUTemp = nullptr, *VelocityVTemp = nullptr, *VelocityWTemp = nullptr;

    float *duProj = nullptr, *dvProj = nullptr, *dwProj = nullptr;
    float *duExtern = nullptr, *dvExtern = nullptr, *dwExtern = nullptr;

    float *TempSrcU = nullptr, *TempSrcV = nullptr, *TempSrcW = nullptr;

    float *Density = nullptr, *DensityInit = nullptr, *DensityPrev = nullptr, *DensityTemp = nullptr, *DensityExtern = nullptr;
    float *Temperature = nullptr, *TemperatureInit = nullptr, *TemperaturePrev = nullptr, *TemperatureTemp = nullptr, *TemperatureExtern = nullptr;

    float *p = nullptr, *p_temp = nullptr;
    float *div = nullptr;

    float *boundaryDesc = nullptr;

    buffer3Df output_density;
    float *host_density;
    buffer3Df output_u, output_v, output_w;
    float *host_u, *host_v, *host_w;

    uint VelocityBufferSizeX = 0;
    uint VelocityBufferSizeY = 0;
    uint VelocityBufferSizeZ = 0;
    uint ScaleFieldSize = 0;

    gpuMapper *GpuSolver;

    MapperBaseGPU VelocityAdvector;
    MapperBaseGPU ScalarAdvector;

    int vel_lastReinit = -11;
    int scalar_lastReinit = -31;

    std::vector<Emitter> sim_emitter;
};

#endif  // BIMOCQGPUSOLVER_H