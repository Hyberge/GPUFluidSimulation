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
    for(auto &emitter : sim_emitter)
    {
        emitter.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
        if(framenum < emitter.emitFrame)
        {
//            float in_value = -emitter.e_sdf->background();

            int compute_elements = _rho._blockx*_rho._blocky*_rho._blockz;
            int slice = _rho._blockx*_rho._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_rho._blockx;
                uint bi = thread_idx%(_rho._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_rho._nx && j<_rho._ny && k<_rho._nz)
                    {
                        float w_x = ((float)i-_rho._ox)*_h;
                        float w_y = ((float)j-_rho._oy)*_h;
                        float w_z = ((float)k-_rho._oz)*_h;
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _rho(i,j,k) = emitter.emit_density;
                            _T(i,j,k) = emitter.emit_temperature;
                        }
                    }
                }
            });

            compute_elements = _un._blockx*_un._blocky*_un._blockz;
            slice = _un._blockx*_un._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_un._blockx;
                uint bi = thread_idx%(_un._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                    if(i<_un._nx && j<_un._ny && k<_un._nz)
                    {
                        float w_x = ((float)i-_un._ox)*_h;
                        float w_y = ((float)j-_un._oy)*_h;
                        float w_z = ((float)k-_un._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _un(i,j,k) = emitter.emit_velocity(world_pos)[0];
                        }
                    }
                }
            });

            compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
            slice = _vn._blockx*_vn._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_vn._blockx;
                uint bi = thread_idx%(_vn._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                    {
                        float w_x = ((float)i-_vn._ox)*_h;
                        float w_y = ((float)j-_vn._oy)*_h;
                        float w_z = ((float)k-_vn._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _vn(i,j,k) = emitter.emit_velocity(world_pos)[1];
                        }
                    }
                }
            });

            compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
            slice = _wn._blockx*_wn._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_wn._blockx;
                uint bi = thread_idx%(_wn._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                    {
                        float w_x = ((float)i-_wn._ox)*_h;
                        float w_y = ((float)j-_wn._oy)*_h;
                        float w_z = ((float)k-_wn._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _wn(i,j,k) = emitter.emit_velocity(world_pos)[2];
                        }
                    }
                }
            });
        }
    }
}
