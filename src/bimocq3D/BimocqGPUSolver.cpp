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

    uint VelocityBufferSizeX = (nx + 1) * ny * nz * sizeof(float);
    uint VelocityBufferSizeY = nx * (ny + 1) * nz * sizeof(float);
    uint VelocityBufferSizeZ = nx * ny * (nz + 1) * sizeof(float);

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

    GpuSolver->allocGPUBuffer((void**)&TempSrcU, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&TempSrcV, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&TempSrcW, VelocityBufferSizeZ);

    uint ScaleFieldSize = nx * ny * nz * sizeof(float);

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
    float totalTime = 0;

    float proj_coeff = 2.f;
    bool velReinit = false;
    bool scalarReinit = false;
    float cfldt = getCFL();
    if (framenum == 0) max_v = CellSize;

    time += VelocityAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);
    time += ScalarAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);

    
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
    sim_emitter[0].update(framenum, CellSize, dt);
    sim_emitter[1].update(framenum, CellSize, dt);
    if (framenum < sim_emitter[0].emitFrame)
    {
        GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            sim_emitter[0].e_pos[0], sim_emitter[0].e_pos[1], sim_emitter[0].e_pos[2], 0.015f, sim_emitter[0].emit_density, sim_emitter[0].emit_temperature, 1.f);

        GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            sim_emitter[1].e_pos[0], sim_emitter[1].e_pos[1], sim_emitter[1].e_pos[2], 0.015f, sim_emitter[1].emit_density, sim_emitter[1].emit_temperature, -1.f);
    }
}

void BimocqGPUSolver::addBuoyancy(float dt)
{
    GpuSolver->add_buoyancy(VelocityV, Density, Temperature, CellNumberX, CellNumberY, CellNumberZ, alpha, beta);
}

void BimocqGPUSolver::initBoundary()
{
    if (boundaryDesc == nullptr)
    {
        uint cellSize = CellNumberX * CellNumberY * CellNumberZ;
        uint bufferSize = cellSize * sizeof(float);
        GpuSolver->allocGPUBuffer((void**)&boundaryDesc, bufferSize`);

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