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

    VirtualGpuMapper *GpuSolver;

    // simulation data
    uint _nx, _ny, _nz;     // voxel number
    float max_v;
    float _h;               // cell size
    float viscosity;
    buffer3Df _un, _vn, _wn;                // velocity
    buffer3Df _uinit, _vinit, _winit;       // v0
    buffer3Df _uprev, _vprev, _wprev;       // velocity in prev-frame
    buffer3Df _utemp, _vtemp, _wtemp;
    buffer3Df _duproj, _dvproj, _dwproj;
    buffer3Df _duextern, _dvextern, _dwextern;
    buffer3Df _rho, _rhotemp, _rhoinit, _rhoprev, _drhoextern;  // density
    buffer3Df _T, _Ttemp, _Tinit, _Tprev, _dTextern;            // Temperature

    buffer3Df _usolid, _vsolid, _wsolid;
    Array3c u_valid, v_valid, w_valid;
    // initialize advector
    MapperBase VelocityAdvector;
    MapperBase ScalarAdvector;
    int vel_lastReinit = 0;
    int scalar_lastReinit = 0;
    Scheme sim_scheme;

    std::vector<Emitter> sim_emitter;
    std::vector<Boundary> sim_boundary;
};

#endif  // BIMOCQGPUSOLVER_H