#include "BimocqGPUSolver.h"

BimocqGPUSolver::BimocqGPUSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, VirtualGpuMapper *mymapper)
{
    CellNumberX = nx;
    CellNumberY = ny;
    CellNumberZ = nz;
    CellSize = L/nx;
    MaxVelocity = 0.f;
    Viscosity = vis_coeff;
    GpuSolver = mymapper;

    VelocityBufferSizeX = (nx + 1) * ny * nz * sizeof(float);
    VelocityBufferSizeY = nx * (ny + 1) * nz * sizeof(float);
    VelocityBufferSizeZ = nx * ny * (nz + 1) * sizeof(float);
    ScaleFieldSize = nx * ny * nz * sizeof(float);


    GpuSolver->allocGPUBuffer((void**)&VelocityU, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&VelocityUInit, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&VelocityUPrev, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&VelocityUTemp, VelocityBufferSizeX);

    GpuSolver->allocGPUBuffer((void**)&VelocityV, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&VelocityVInit, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&VelocityVPrev, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&VelocityVTemp, VelocityBufferSizeY);

    GpuSolver->allocGPUBuffer((void**)&VelocityW, VelocityBufferSizeZ);
    GpuSolver->allocGPUBuffer((void**)&VelocityWInit, VelocityBufferSizeZ);
    GpuSolver->allocGPUBuffer((void**)&VelocityWPrev, VelocityBufferSizeZ);
    GpuSolver->allocGPUBuffer((void**)&VelocityWTemp, VelocityBufferSizeZ);

    GpuSolver->allocGPUBuffer((void**)&duExtern, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&dvExtern, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&dwExtern, VelocityBufferSizeZ);

    GpuSolver->allocGPUBuffer((void**)&TempSrcU, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&TempSrcV, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&TempSrcW, VelocityBufferSizeZ);

    GpuSolver->allocGPUBuffer((void**)&Density, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityInit, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityPrev, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityTemp, ScaleFieldSize);

    GpuSolver->allocGPUBuffer((void**)&Temperature, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperatureInit, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperaturePrev, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperatureTemp, ScaleFieldSize);

    VelocityAdvector.init(CellNumberX, CellNumberY, CellNumberZ, CellSize, blend_coeff, mymapper);
    ScalarAdvector.init(CellNumberX, CellNumberY, CellNumberZ, CellSize, blend_coeff, mymapper);
}

void BimocqGPUSolver::advance(int framenum, float dt)
{
    float time = 0;

    float proj_coeff = 2.f;
    bool velReinit = false;
    bool scalarReinit = false;
    float cfldt = getCFL();
    if (framenum == 0) max_v = CellSize;

    time += VelocityAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);
    time += ScalarAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);

    time += semilagAdvect(cfldt, dt);

    time += VelocityAdvector.advectVelocity(VelocityU, VelocityV, VelocityW, VelocityUInit, VelocityVInit, VelocityWInit, VelocityUPrev, VelocityVPrev, VelocityWPrev, TempSrcU, TempSrcV, TempSrcW);
    time += ScalarAdvector.advectField(Density, DensityInit, DensityPrev, TempSrcU);
    time += ScalarAdvector.advectField(Temperature, TemperatureInit, TemperaturePrev, TempSrcV);

    // no sim_boundary 
    //time += blendBoundary(VelocityU, VelocityUTemp);
    //time += blendBoundary(VelocityV, VelocityVTemp);
    //time += blendBoundary(VelocityW, VelocityWTemp);
    //time += blendBoundary(Density, DensityTemp);
    //time += blendBoundary(Temperature, TemperatureTemp);

    GpuSolver->copyDeviceToDevice(VelocityUTemp, VelocityU, VelocityBufferSizeX);
    GpuSolver->copyDeviceToDevice(VelocityVTemp, VelocityV, VelocityBufferSizeY);
    GpuSolver->copyDeviceToDevice(VelocityWTemp, VelocityW, VelocityBufferSizeZ);
    
    GpuSolver->copyDeviceToDevice(DensityTemp, Density, ScaleFieldSize);
    GpuSolver->copyDeviceToDevice(TemperatureTemp, Temperature, ScaleFieldSize);

    // not need
    //clearBoundary(Density);

    time += emitSmoke(framenum, dt);
    time += addBuoyancy(dt);

    if (Viscosity)
    {
        diffuseField(VelocityU, TempSrcU, CellNumberX + 1, CellNumberY, CellNumberZ, Viscosity, dt);
        diffuseField(VelocityV, TempSrcV, CellNumberX, CellNumberY + 1, CellNumberZ, Viscosity, dt);
        diffuseField(VelocityW, TempSrcW, CellNumberX, CellNumberY, CellNumberZ + 1, Viscosity, dt);
    }
    
    // calculate velocity change due to external forces(e.g. buoyancy)
    GpuSolver->addFields(duExtern, VelocityU, VelocityUTemp, -1, (ni + 1) * nj * nk);
    GpuSolver->addFields(dvExtern, VelocityV, VelocityVTemp, -1, ni * (nj + 1) * nk);
    GpuSolver->addFields(dwExtern, VelocityW, VelocityWTemp, -1, ni * nj * (nk + 1));

    GpuSolver->copyDeviceToDevice(VelocityUTemp, VelocityU, VelocityBufferSizeX);
    GpuSolver->copyDeviceToDevice(VelocityVTemp, VelocityV, VelocityBufferSizeY);
    GpuSolver->copyDeviceToDevice(VelocityWTemp, VelocityW, VelocityBufferSizeZ);

    projection();
}

float BimocqGPUSolver::semilagAdvect(float cfldt, float dt)
{
    float time = 0.f;
    
    time += GpuSolver->semilagAdvectVelocity(VelocityU, VelocityV, VelocityW, VelocityUTemp, VelocityVTemp, VelocityWTemp, TempSrcU, TempSrcV, TempSrcW, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);

    time += GpuSolver->semilagAdvectField(DensityTemp, Density, VelocityU, VelocityV, VelocityW, 0, 0, 0, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);

    time += GpuSolver->semilagAdvectField(TemperatureTemp, Temperature, VelocityU, VelocityV, VelocityW, 0, 0, 0, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);

    return time;
}

float BimocqGPUSolver::getCFL()
{
    max_v = 1e-4;
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx+1;i++)
    {
        if (fabs(_un(i,j,k))>max_v)
        {
            max_v = fabs(_un(i,j,k));
        }
    }
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny+1;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_vn(i,j,k))>max_v)
        {
            max_v = fabs(_vn(i,j,k));
        }
    }
    for (uint k=0; k<_nz+1;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_wn(i,j,k))>max_v)
        {
            max_v = fabs(_wn(i,j,k));
        }
    }
    return _h / max_v;
}


void BimocqGPUSolver::emitSmoke(int framenum, float dt)
{
    float time = 0;

    sim_emitter[0].update(framenum, CellSize, dt);
    sim_emitter[1].update(framenum, CellSize, dt);
    if (framenum < sim_emitter[0].emitFrame)
    {
        time += GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            sim_emitter[0].e_pos[0], sim_emitter[0].e_pos[1], sim_emitter[0].e_pos[2], 0.015f, sim_emitter[0].emit_density, sim_emitter[0].emit_temperature, 1.f);

        time += GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            sim_emitter[1].e_pos[0], sim_emitter[1].e_pos[1], sim_emitter[1].e_pos[2], 0.015f, sim_emitter[1].emit_density, sim_emitter[1].emit_temperature, -1.f);
    }

    return time;
}

float BimocqGPUSolver::addBuoyancy(float dt)
{
    return GpuSolver->add_buoyancy(VelocityV, Density, Temperature, CellNumberX, CellNumberY, CellNumberZ, alpha, beta);
}

float BimocqGPUSolver::diffuseField(float *field, float *fieldTemp, int ni, int nj, int nk, float nu, float dt)
{
    float coef = nu * (dt / (CellSize * CellSize));

    float time = GpuSolver->diffuseField(field, fieldTemp, ni, nj, nk, coef);

    GpuSolver->copyDeviceToDevice(field, fieldTemp, ni * nj * nk * sizeof(float));

    return time;
}

float BimocqGPUSolver::projection()
{
#if 1   // jacobi iteration

#else   // AMG solver
#endif
}

void BimocqGPUSolver::initBoundary()
{
    if (boundaryDesc == nullptr)
    {
        uint cellSize = CellNumberX * CellNumberY * CellNumberZ;
        GpuSolver->allocGPUBuffer((void**)&boundaryDesc, ScaleFieldSize);

        float* initTemp = new float[cellSize];
        memset(initTemp, 0, bufferSize);
        tbb::parallel_for(0, (int)CellNumberZ, 1, [&](int thread_idx) {

            int k = thread_idx;
            for (size_t j = 0; j < CellNumberY; j++)
            {
                for (size_t i = 0; i < CellNumberX; i++)
                {
                    uint bufferIndex = i + j * CellNumberX + k * CellNumberY * CellNumberX;
                    
                    if (i < 1) initTemp[bufferIndex] = 2;
                    if (j < 1) initTemp[bufferIndex] = 2;
                    if (k < 1) initTemp[bufferIndex] = 2;

                    if (i < CleeNumberX - 1) initTemp[bufferIndex] = 2;
                    if (j < CleeNumberY - 1) initTemp[bufferIndex] = 1;
                    if (k < CleeNumberZ - 1) initTemp[bufferIndex] = 2;
                }
            }
            
        });

        GpuSolver->copyHostToDevice(initTemp, boundaryDesc, ScaleFieldSize);
    }
}

void BimocqSolver::velocityReinitialize()
{
    uint bufferSizeU = (CellNumberX + 1) * CellNumberY * CellNumberZ * sizeof(float);
    uint bufferSizeV = CellNumberX * (CellNumberY + 1) * CellNumberZ * sizeof(float);
    uint bufferSizeW = CellNumberX * CellNumberY * (CellNumberZ + 1) * sizeof(float);

    GpuSolver->copyDeviceToDevice(VelocityUPrev, VelocityUInit, bufferSizeU);
    GpuSolver->copyDeviceToDevice(VelocityVPrev, VelocityVInit, bufferSizeV);
    GpuSolver->copyDeviceToDevice(VelocityWPrev, VelocityWInit, bufferSizeW);

    GpuSolver->copyDeviceToDevice(VelocityUInit, VelocityU, bufferSizeU);
    GpuSolver->copyDeviceToDevice(VelocityVInit, VelocityV, bufferSizeV);
    GpuSolver->copyDeviceToDevice(VelocityWInit, VelocityW, bufferSizeW);
}

void BimocqSolver::scalarReinitialize()
{
    uint bufferSize = CellNumberX * CellNumberY * CellNumberZ * sizeof(float);

    GpuSolver->copyDeviceToDevice(DensityPrev, DensityInit, BufferSize);
    GpuSolver->copyDeviceToDevice(TemperaturePrev, TemperatureInit, BufferSize);
    
    GpuSolver->copyDeviceToDevice(DensityInit, Density, BufferSize);
    GpuSolver->copyDeviceToDevice(TemperatureInit, Temperature, BufferSize);
}