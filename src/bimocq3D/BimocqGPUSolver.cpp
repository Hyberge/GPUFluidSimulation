#include "BimocqGPUSolver.h"

BimocqGPUSolver::BimocqGPUSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, gpuMapper *mymapper)
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

    GpuSolver->allocGPUBuffer((void**)&duProj, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&dvProj, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&dwProj, VelocityBufferSizeZ);

    GpuSolver->allocGPUBuffer((void**)&duExtern, VelocityBufferSizeX);
    GpuSolver->allocGPUBuffer((void**)&dvExtern, VelocityBufferSizeY);
    GpuSolver->allocGPUBuffer((void**)&dwExtern, VelocityBufferSizeZ);

    //GpuSolver->allocGPUBuffer((void**)&TempSrcU, VelocityBufferSizeX);
    //GpuSolver->allocGPUBuffer((void**)&TempSrcV, VelocityBufferSizeY);
    //GpuSolver->allocGPUBuffer((void**)&TempSrcW, VelocityBufferSizeZ);

    GpuSolver->allocGPUBuffer((void**)&Density, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityInit, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityPrev, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityTemp, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&DensityExtern, ScaleFieldSize);

    GpuSolver->allocGPUBuffer((void**)&Temperature, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperatureInit, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperaturePrev, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperatureTemp, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&TemperatureExtern, ScaleFieldSize);

    GpuSolver->allocGPUBuffer((void**)&p, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&p_temp, ScaleFieldSize);
    GpuSolver->allocGPUBuffer((void**)&div, ScaleFieldSize);

    VelocityAdvector.init(CellNumberX, CellNumberY, CellNumberZ, CellSize, blend_coeff, mymapper);
    ScalarAdvector.init(CellNumberX, CellNumberY, CellNumberZ, CellSize, blend_coeff, mymapper);

    output_density.init(CellNumberX, CellNumberY, CellNumberZ);
    host_density = new float[CellNumberX*CellNumberY*CellNumberZ];

    output_u.init(CellNumberX+1, CellNumberY, CellNumberZ);
    output_v.init(CellNumberX, CellNumberY+1, CellNumberZ);
    output_w.init(CellNumberX, CellNumberY, CellNumberZ+1);
    host_u = new float[(CellNumberX+1)*CellNumberY*CellNumberZ];
    host_v = new float[CellNumberX*(CellNumberY+1)*CellNumberZ];
    host_w = new float[CellNumberX*CellNumberY*(CellNumberZ+1)];
}

void BimocqGPUSolver::advance(int framenum, float dt)
{
    if (framenum == 0) MaxVelocity = CellSize;
    
    float proj_coeff = 2.f;
    bool velReinit = false;
    bool scalarReinit = false;
    float cfldt = getCFL();

    GpuSolver->startEventRecord();

    VelocityAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);
    ScalarAdvector.updateMapping(VelocityU, VelocityV, VelocityW, cfldt, dt);

    //semilagAdvect(cfldt, dt);

    VelocityAdvector.advectVelocity(VelocityU, VelocityV, VelocityW, VelocityUInit, VelocityVInit, VelocityWInit, VelocityUPrev, VelocityVPrev, VelocityWPrev);
    ScalarAdvector.advectField(Density, DensityInit, DensityPrev);
    ScalarAdvector.advectField(Temperature, TemperatureInit, TemperaturePrev);

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

    emitSmoke(framenum, dt);
    addBuoyancy(dt);

    if (Viscosity)
    {
        diffuseField(VelocityU, VelocityUTemp, CellNumberX + 1, CellNumberY, CellNumberZ, 20, Viscosity, dt);
        diffuseField(VelocityV, VelocityVTemp, CellNumberX, CellNumberY + 1, CellNumberZ, 20, Viscosity, dt);
        diffuseField(VelocityW, VelocityWTemp, CellNumberX, CellNumberY, CellNumberZ + 1, 20, Viscosity, dt);
    }
    
    // calculate velocity change due to external forces(e.g. buoyancy)
    GpuSolver->addFields(duExtern, VelocityU, VelocityUTemp, -1, (CellNumberX+1)*CellNumberY*CellNumberZ);
    GpuSolver->addFields(dvExtern, VelocityV, VelocityVTemp, -1, CellNumberX*(CellNumberY+1)*CellNumberZ);
    GpuSolver->addFields(dwExtern, VelocityW, VelocityWTemp, -1, CellNumberX*CellNumberY*(CellNumberZ+1));

    GpuSolver->copyDeviceToDevice(VelocityUTemp, VelocityU, VelocityBufferSizeX);
    GpuSolver->copyDeviceToDevice(VelocityVTemp, VelocityV, VelocityBufferSizeY);
    GpuSolver->copyDeviceToDevice(VelocityWTemp, VelocityW, VelocityBufferSizeZ);

    projection();

    GpuSolver->copyDeviceToDevice(duProj, VelocityU, VelocityBufferSizeX);
    GpuSolver->copyDeviceToDevice(dvProj, VelocityV, VelocityBufferSizeY);
    GpuSolver->copyDeviceToDevice(dwProj, VelocityW, VelocityBufferSizeZ);
    GpuSolver->add(duProj, VelocityUTemp, -1.f, (CellNumberX+1)*CellNumberY*CellNumberZ);
    GpuSolver->add(dvProj, VelocityVTemp, -1.f, CellNumberX*(CellNumberY+1)*CellNumberZ);
    GpuSolver->add(dwProj, VelocityWTemp, -1.f, CellNumberX*CellNumberY*(CellNumberZ+1));

    GpuSolver->copyDeviceToDevice(DensityExtern, Density, ScaleFieldSize);
    GpuSolver->copyDeviceToDevice(TemperatureExtern, Temperature, ScaleFieldSize);
    GpuSolver->add(DensityExtern, DensityTemp, -1.f, CellNumberX*CellNumberY*CellNumberZ);
    GpuSolver->add(TemperatureExtern, TemperatureTemp, -1.f, CellNumberX*CellNumberY*CellNumberZ);

    if (framenum - vel_lastReinit > 10)
    {
        velReinit = true;
        vel_lastReinit = framenum;
        proj_coeff = 1.f;
    }

    if (framenum - scalar_lastReinit > 30)
    {
        scalarReinit = true;
        scalar_lastReinit = framenum;
    }

    VelocityAdvector.accumulateVelocity(VelocityUInit, VelocityVInit, VelocityWInit, duExtern, dvExtern, dwExtern, 1.f);
    VelocityAdvector.accumulateVelocity(VelocityUInit, VelocityVInit, VelocityWInit, duProj, dvProj, dwProj, proj_coeff);
    ScalarAdvector.accumulateField(DensityInit, DensityExtern);
    ScalarAdvector.accumulateField(TemperatureInit, TemperatureExtern);

    if (1)
    {
        VelocityAdvector.reinitializeMapping();
        velocityReinitialize();
        VelocityAdvector.accumulateVelocity(VelocityUInit, VelocityVInit, VelocityWInit, duProj, dvProj, dwProj, 1.f);
    }

    if (1)
    {
        ScalarAdvector.reinitializeMapping();
        scalarReinitialize();
    }

    float time = GpuSolver->endEventRecord();

    cout << "[Bimocq GPU Time: " << time << "ms ]" << endl;
}

void BimocqGPUSolver::semilagAdvect(float cfldt, float dt)
{
    GpuSolver->semilagAdvectVelocity(VelocityU, VelocityV, VelocityW, VelocityUTemp, VelocityVTemp, VelocityWTemp, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);

    GpuSolver->semilagAdvectField(DensityTemp, Density, VelocityU, VelocityV, VelocityW, 0, 0, 0, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);

    GpuSolver->semilagAdvectField(TemperatureTemp, Temperature, VelocityU, VelocityV, VelocityW, 0, 0, 0, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);
}

float BimocqGPUSolver::getCFL()
{
    //MaxVelocity = 1e-4;
    //for (uint k=0; k<CellNumberZ;k++) for (uint j=0; j<CellNumberY;j++) for (uint i=0; i<CellNumberX+1;i++)
    //{
    //    if (fabs(VelocityU(i,j,k))>MaxVelocity)
    //    {
    //        MaxVelocity = fabs(_un(i,j,k));
    //    }
    //}
    //for (uint k=0; k<CellNumberZ;k++) for (uint j=0; j<CellNumberY+1;j++) for (uint i=0; i<CellNumberX;i++)
    //{
    //    if (fabs(_vn(i,j,k))>MaxVelocity)
    //    {
    //        MaxVelocity = fabs(_vn(i,j,k));
    //    }
    //}
    //for (uint k=0; k<CellNumberZ+1;k++) for (uint j=0; j<CellNumberY;j++) for (uint i=0; i<CellNumberX;i++)
    //{
    //    if (fabs(_wn(i,j,k))>MaxVelocity)
    //    {
    //        MaxVelocity = fabs(_wn(i,j,k));
    //    }
    //}
    return CellSize / MaxVelocity;
}


void BimocqGPUSolver::emitSmoke(int framenum, float dt)
{
    sim_emitter[0].update(framenum, CellSize, dt);
    sim_emitter[1].update(framenum, CellSize, dt);
    if (framenum < sim_emitter[0].emitFrame)
    {
        //GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
        //    sim_emitter[0].e_pos[0], sim_emitter[0].e_pos[1], sim_emitter[0].e_pos[2], 0.015f, sim_emitter[0].emit_density, sim_emitter[0].emit_temperature, 1.f);
        //GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
        //    sim_emitter[1].e_pos[0], sim_emitter[1].e_pos[1], sim_emitter[1].e_pos[2], 0.015f, sim_emitter[1].emit_density, sim_emitter[1].emit_temperature, -1.f);

        GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            0.04f, 0.2f, 0.2f, 0.015f, sim_emitter[0].emit_density, sim_emitter[0].emit_temperature, 1.f);
        GpuSolver->emitSmoke(VelocityU, VelocityV, VelocityW, Density, Temperature, CellSize, CellNumberX, CellNumberY, CellNumberZ, 
            0.16f, 0.201f, 0.2f, 0.015f, sim_emitter[1].emit_density, sim_emitter[1].emit_temperature, -1.f);
    }
}

void BimocqGPUSolver::addBuoyancy(float dt)
{
    GpuSolver->add_buoyancy(VelocityV, Density, Temperature, CellNumberX, CellNumberY, CellNumberZ, _alpha, _beta, dt);
}

void BimocqGPUSolver::diffuseField(float *field, float *fieldTemp, int ni, int nj, int nk, int iter, float nu, float dt)
{
    float coef = nu * (dt / (CellSize * CellSize));

    GpuSolver->diffuseField(field, fieldTemp, ni, nj, nk, iter, coef);
}

void BimocqGPUSolver::projection()
{
#if 1   // jacobi iteration
    GpuSolver->projectionJacobi(VelocityU, VelocityV, VelocityW, div, p, p_temp, CellNumberX, CellNumberY, CellNumberZ, 20, 0.5, -1, 1.0 / 6.0);
#else   // AMG solver
#endif
}

void BimocqGPUSolver::initBoundary()
{
    //if (boundaryDesc == nullptr)
    //{
    //    uint cellSize = CellNumberX * CellNumberY * CellNumberZ;
    //    GpuSolver->allocGPUBuffer((void**)&boundaryDesc, ScaleFieldSize);
//
    //    float* initTemp = new float[cellSize];
    //    memset(initTemp, 0, bufferSize);
    //    tbb::parallel_for(0, (int)CellNumberZ, 1, [&](int thread_idx) {
//
    //        int k = thread_idx;
    //        for (size_t j = 0; j < CellNumberY; j++)
    //        {
    //            for (size_t i = 0; i < CellNumberX; i++)
    //            {
    //                uint bufferIndex = i + j * CellNumberX + k * CellNumberY * CellNumberX;
    //                
    //                if (i < 1) initTemp[bufferIndex] = 2;
    //                if (j < 1) initTemp[bufferIndex] = 2;
    //                if (k < 1) initTemp[bufferIndex] = 2;
//
    //                if (i < CleeNumberX - 1) initTemp[bufferIndex] = 2;
    //                if (j < CleeNumberY - 1) initTemp[bufferIndex] = 1;
    //                if (k < CleeNumberZ - 1) initTemp[bufferIndex] = 2;
    //            }
    //        }
    //        
    //    });
//
    //    GpuSolver->copyHostToDevice(initTemp, boundaryDesc, ScaleFieldSize);
    //}
}

void BimocqGPUSolver::velocityReinitialize()
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

void BimocqGPUSolver::scalarReinitialize()
{
    uint bufferSize = CellNumberX * CellNumberY * CellNumberZ * sizeof(float);

    GpuSolver->copyDeviceToDevice(DensityPrev, DensityInit, ScaleFieldSize);
    GpuSolver->copyDeviceToDevice(TemperaturePrev, TemperatureInit, ScaleFieldSize);
    
    GpuSolver->copyDeviceToDevice(DensityInit, Density, ScaleFieldSize);
    GpuSolver->copyDeviceToDevice(TemperatureInit, Temperature, ScaleFieldSize);
}

void BimocqGPUSolver::setSmoke(float drop, float raise, const std::vector<Emitter> &emitters)
{
    _alpha = drop;
    _beta = raise;
    sim_emitter = emitters;
}

void BimocqGPUSolver::outputResult(uint frame, string filepath)
{
    GpuSolver->copyDeviceToHost(output_density, host_density, Density);
    GpuSolver->copyDeviceToHost(output_u, host_u, VelocityU);
    GpuSolver->copyDeviceToHost(output_v, host_v, VelocityV);
    GpuSolver->copyDeviceToHost(output_w, host_w, VelocityW);
    writeVDB(frame + 1, filepath, CellSize, output_density);
}