//
// Created by ziyin on 19-3-19.
//

#include "Mapping.h"

void MapperBase::updateBackward(float cfldt, float dt)
{
    // the backward mapping will be updated in gpu.x_in, gpu.y_in, gpu.z_in
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    float T = 0.f;
    float substep = cfldt;
    while (T < dt)
    {
        if (T + substep > dt) substep = dt - T;
        gpuSolver->solveBackwardDMC(substep);
        T += substep;
    }
    gpuSolver->copyDeviceToHost(backward_x, gpuSolver->x_host, gpuSolver->x_in);
    gpuSolver->copyDeviceToHost(backward_y, gpuSolver->y_host, gpuSolver->y_in);
    gpuSolver->copyDeviceToHost(backward_z, gpuSolver->z_host, gpuSolver->z_in);
}

void MapperBase::updateForward(float cfldt, float dt)
{
    // the forward mapping will be updated in gpu.x_out, gpu.y_out, gpu.z_out
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->solveForward(cfldt, dt);
    gpuSolver->copyDeviceToHost(forward_x, gpuSolver->x_host, gpuSolver->x_out);
    gpuSolver->copyDeviceToHost(forward_y, gpuSolver->y_host, gpuSolver->y_out);
    gpuSolver->copyDeviceToHost(forward_z, gpuSolver->z_host, gpuSolver->z_out);
}

void MapperBase::updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt)
{
    // copy velocity buffer from host to device
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w, _ni*_nj*(_nk+1)*sizeof(float));
    // update forward and backward mapping
    // forward mapping will be stored in GPU.x_out, GPU.y_out, GPU.z_out
    // backward mapping will be stored in GPU.x_in, GPU,y_in, GPU.z_in
    // NOTE: ORDER MUST NOT CHANGE DUE TO REUSE OF GPU BUFFER
    updateBackward(cfldt, dt);
    updateForward(cfldt, dt);
}

void MapperBase::accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    // copy accumulated velocity change at initial state to gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // copy velocity change (e.g. buoyancy, projection) at time t to gpu.u, gpu.v, gpu.w
    gpuSolver->copyHostToDevice(u_change, gpuSolver->u_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_change, gpuSolver->v_host, gpuSolver->v, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_change, gpuSolver->w_host, gpuSolver->w, _ni*_nj*(_nk+1)*sizeof(float));
    // now add velocity change multiply with coefficient back to initial state, which is stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->accumulateVelocity(false, coeff);
    // copy accumulated velocity back to host buffer
    gpuSolver->copyDeviceToHost(u_init, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(v_init, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(w_init, gpuSolver->w_host, gpuSolver->dw);
}

void MapperBase::accumulateField(buffer3Df &field_init, const buffer3Df &field_change)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    // reuse gpu.u, gpu.du to save GPU buffer
    // copy accumulated field change at initial state to gpu.du
    gpuSolver->copyHostToDevice(field_init, gpuSolver->x_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    // copy field change from external(e.g. source emitter) at time t to gpu.u, gpu.v, gpu.w
    gpuSolver->copyHostToDevice(field_change, gpuSolver->x_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    // now add field change back to initial state, which is stored in gpu.du
    gpuSolver->accumulateField(false, 1.f);
    // copy accumulated field back to host buffer
    gpuSolver->copyDeviceToHost(field_init, gpuSolver->u_host, gpuSolver->du);
}

float MapperBase::estimateDistortion(const buffer3Dc &boundary)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->estimateDistortionCUDA();
    cudaMemcpy(gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float), cudaMemcpyDeviceToHost);
    float max_dist = 0.f;
    for (uint i = 0; i < forward_x._nx; i++)
    {
        for (uint j = 0; j < forward_x._ny; j++)
        {
            for (uint k = 0; k < forward_x._nz; k++)
            {
                if (boundary(i,j,k) != 2)
                {
                    uint index = i + forward_x._nx*j + forward_x._nx * forward_x._ny * k;
                    max_dist = max(max_dist, gpuSolver->u_host[index]);
                }
            }
        }
    }
    
    return sqrt(max_dist);
}

void MapperBase::init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper)
{
    _ni = ni;
    _nj = nj;
    _nk = nk;
    _h = h;
    blend_coeff = coeff;
    total_reinit_count = 0;
    forward_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_xprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_yprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_zprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    gpuSolver = mymapper;
    int compute_elements = forward_x._blockx*forward_x._blocky*forward_x._blockz;
    int slice = forward_x._blockx*forward_x._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/forward_x._blockx;
        uint bi = thread_idx%(forward_x._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<forward_x._nx && j<forward_x._ny && k<forward_x._nz)
                    {
                        float world_x = ((float)i-forward_x._ox)*_h;
                        float world_y = ((float)j-forward_x._oy)*_h;
                        float world_z = ((float)k-forward_x._oz)*_h;
                        forward_x(i,j,k) = world_x;
                        forward_y(i,j,k) = world_y;
                        forward_z(i,j,k) = world_z;
                        backward_x(i,j,k) = world_x;
                        backward_y(i,j,k) = world_y;
                        backward_z(i,j,k) = world_z;
                    }
                }
    });
    backward_xprev.copy(backward_x);
    backward_yprev.copy(backward_y);
    backward_zprev.copy(backward_z);
}

void MapperBase::advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev)
{
    // forward and backward mapping are already in gpu.x_out and gpu.x_in correspondingly
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    // ready for advection and compensation
    // init velocity buffer will be stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // updated velocity(no compensation) will be store in gpu.u, gpu.v, gpu.w
    gpuSolver->advectVelocity(false);
    // compensated and clamped velocity will be stored in gpu.u, gpu.v, gpu.w
    gpuSolver->compensateVelocity(false);
    // now copy backward_prev to gpu.x_out, gpu._yout, gpu.z_out
    gpuSolver->copyHostToDevice(backward_xprev, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_yprev, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_zprev, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(u_prev, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_prev, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_prev, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // velocity from backward_prev will be blended with velocity from backward_curr
    // if reinitialization is not happen, prev buffers are not valid
    if (total_reinit_count != 0)
        gpuSolver->advectVelocityDouble(false, blend_coeff);
    else
        gpuSolver->advectVelocityDouble(false, 1.f);
    // copy velocity back to host
    gpuSolver->copyDeviceToHost(un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(wn, gpuSolver->w_host, gpuSolver->w);
}

void MapperBase::advectField(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    // reuse gpu.du for other fields(e.g. density, temperature)
    gpuSolver->copyHostToDevice(field_init, gpuSolver->x_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    // updated field will be stored in gpu.u
    gpuSolver->advectField(false);
    // compensated field will be stored in gpu.u
    gpuSolver->compensateField(false);
    // now copy backward_prev to gpu.x_out, gpu._yout, gpu.z_out
    gpuSolver->copyHostToDevice(backward_xprev, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_yprev, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_zprev, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
//     copy field_prev to gpu.du
    gpuSolver->copyHostToDevice(field_prev, gpuSolver->x_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    // field from backward_prev will be blended with field from backward_curr
    // if reinitialization is not happen, prev buffers are not valid
    if (total_reinit_count != 0)
        gpuSolver->advectFieldDouble(false, blend_coeff);
    else
        gpuSolver->advectFieldDouble(false, 1.f);
    // copy field back to host
    gpuSolver->copyDeviceToHost(field, gpuSolver->u_host, gpuSolver->u);
}

void MapperBase::reinitializeMapping()
{
    total_reinit_count ++;
    backward_xprev.copy(backward_x);
    backward_yprev.copy(backward_y);
    backward_zprev.copy(backward_z);

    int compute_elements = forward_x._blockx*forward_x._blocky*forward_x._blockz;
    int slice = forward_x._blockx*forward_x._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/forward_x._blockx;
        uint bi = thread_idx%(forward_x._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<forward_x._nx && j<forward_x._ny && k<forward_x._nz)
            {
                float world_x = ((float)i-forward_x._ox)*_h;
                float world_y = ((float)j-forward_x._oy)*_h;
                float world_z = ((float)k-forward_x._oz)*_h;
                forward_x(i,j,k) = world_x;
                forward_y(i,j,k) = world_y;
                forward_z(i,j,k) = world_z;
                backward_x(i,j,k) = world_x;
                backward_y(i,j,k) = world_y;
                backward_z(i,j,k) = world_z;
            }
        }
    });
}

/*
    class MapperBaseGPU.
*/
void MapperBaseGPU::init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper)
{
    CellNumberX = ni;
    CellNumberY = nj;
    CellNumberZ = nk;
    CellSize = h;
    BlendCoeff = coeff;
    TotalReinitCount = 0;

    gpuSolver = mymapper;

    uint bufferSize = CellNumberX * CellNumberY * CellNumberZ * sizeof(float);
    gpuSolver->allocGPUBuffer((void**)&ForwardX, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&ForwardY, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&ForwardZ, bufferSize);

    gpuSolver->allocGPUBuffer((void**)&BackwardX, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&BackwardY, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&BackwardZ, bufferSize);

    gpuSolver->allocGPUBuffer((void**)&BackwardXPrev, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&BackwardYPrev, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&BackwardZPrev, bufferSize);

    gpuSolver->allocGPUBuffer((void**)&InitX, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&InitY, bufferSize);
    gpuSolver->allocGPUBuffer((void**)&InitZ, bufferSize);

    gpuSolver->allocGPUBuffer((void**)&Distortion, bufferSize);

    TempX = new float[bufferSize];
    TempY = new float[bufferSize];
    TempZ = new float[bufferSize];
    tbb::parallel_for(0, (int)CellNumberZ, 1, [&](int thread_idx) {

        int k = thread_idx;
        for (size_t j = 0; j < CellNumberY; j++)
        {
            for (size_t i = 0; i < CellNumberX; i++)
            {
                uint BufferIndex = i + j * CellNumberX + k * CellNumberY * CellNumberX;
                TempX[BufferIndex] = i * CellSize;
                TempY[BufferIndex] = j * CellSize;
                TempZ[BufferIndex] = k * CellSize;
            }
        }
         
    });

    gpuSolver->copyHostToDevice(TempX, InitX, bufferSize);
    gpuSolver->copyHostToDevice(TempY, InitY, bufferSize);
    gpuSolver->copyHostToDevice(TempZ, InitZ, bufferSize);

    gpuSolver->copyHostToDevice(InitX, ForwardX, bufferSize);
    gpuSolver->copyHostToDevice(InitY, ForwardY, bufferSize);
    gpuSolver->copyHostToDevice(InitZ, ForwardZ, bufferSize);

    gpuSolver->copyDeviceToDevice(InitX, BackwardX, bufferSize);
    gpuSolver->copyDeviceToDevice(InitY, BackwardY, bufferSize);
    gpuSolver->copyDeviceToDevice(InitZ, BackwardZ, bufferSize);

    gpuSolver->copyDeviceToDevice(ForwardX, BackwardXPrev, bufferSize);
    gpuSolver->copyDeviceToDevice(ForwardY, BackwardYPrev, bufferSize);
    gpuSolver->copyDeviceToDevice(ForwardZ, BackwardZPrev, bufferSize);

    //delete TempX;
    //delete TempY;
    //delete TempY;
}

void MapperBaseGPU::updateMapping(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt)
{
    updateBackward(velocityU, velocityV, velocityW, cfldt, dt);

    updateForward(velocityU, velocityV, velocityW, cfldt, dt);
}

void MapperBaseGPU::updateBackward(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt)
{
    float T = 0.f;
    float substep = cfldt;
    while (T < dt)
    {
        if (T + substep > dt)
        {
            substep = dt - T;
        }
        gpuSolver->solveBackwardDMC(velocityU, velocityV, velocityW, BackwardX, BackwardY, BackwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, substep);

        T += substep;
    }
}

void MapperBaseGPU::updateForward(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt)
{
    gpuSolver->solveForward(velocityU, velocityV, velocityW, ForwardX, ForwardY, ForwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, cfldt, dt);
}

void MapperBaseGPU::advectVelocity(float *velocityU, float *velocityV, float *velocityW,
                        float *velocityUInit, float *velocityVInit, float *velocityWInit,
                        float *velocityUPrev, float *velocityVPrev, float *velocityWPrev)
{
    gpuSolver->advectVelocity(velocityU, velocityV, velocityW, velocityUInit, velocityVInit, velocityWInit, BackwardX, BackwardY, BackwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false);

    gpuSolver->compensateVelocity(velocityU, velocityV, velocityW, velocityUInit, velocityVInit, velocityWInit, ForwardX, ForwardY, ForwardZ, BackwardX, BackwardY, BackwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false);

    if (TotalReinitCount != 0)
    {
        gpuSolver->advectVelocityDouble(velocityU, velocityV, velocityW, velocityUPrev, velocityVPrev, velocityWPrev, BackwardX, BackwardY, BackwardZ, BackwardXPrev, BackwardYPrev, BackwardZPrev, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, BlendCoeff);
    }
    else
    {
        gpuSolver->advectVelocityDouble(velocityU, velocityV, velocityW, velocityUPrev, velocityVPrev, velocityWPrev, BackwardX, BackwardY, BackwardZ, BackwardXPrev, BackwardYPrev, BackwardZPrev, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, 1.f);
    }
}

void MapperBaseGPU::advectField(float *field, float *fieldInit, float *fieldPrev)
{
    gpuSolver->advectField(field, fieldInit, BackwardX, BackwardY, BackwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false);

    gpuSolver->compensateField(field, fieldInit, ForwardX, ForwardY, ForwardZ, BackwardX, BackwardY, BackwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false);

    if (TotalReinitCount != 0)
    {
        gpuSolver->advectFieldDouble(field, fieldPrev, BackwardX, BackwardY, BackwardZ, BackwardXPrev, BackwardYPrev, BackwardZPrev, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, BlendCoeff);
    }
    else
    {
        gpuSolver->advectFieldDouble(field, fieldPrev, BackwardX, BackwardY, BackwardZ, BackwardXPrev, BackwardYPrev, BackwardZPrev, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, 1.f);
    }
}

//float MapperBaseGPU::estimateDistortion(float* boundary)
//{
//    float time = 0.f;
//
//    gpuSolver->estimateDistortionCUDA(Distortion, BackwardX, BackwardY, BackwardZ, ForwardX, ForwardY, ForwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ);
//
//    
//
//    return time;
//}

void MapperBaseGPU::accumulateVelocity(float *duInit, float *dvInit, float *dwInit, float *uChange, float *vChange, float *wChange, float coeff)
{
    gpuSolver->accumulateVelocity(uChange, vChange, wChange, duInit, dvInit, dwInit, ForwardX, ForwardY, ForwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, coeff);
}

void MapperBaseGPU::accumulateField(float *dfieldInit, float *fieldChange)
{
    gpuSolver->accumulateField(fieldChange, dfieldInit, ForwardX, ForwardY, ForwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ, false, 1.0f);
}

void MapperBaseGPU::reinitializeMapping()
{
    uint bufferSize = CellNumberX * CellNumberY * CellNumberZ * sizeof(float);

    gpuSolver->copyDeviceToDevice(BackwardXPrev, BackwardX, bufferSize);
    gpuSolver->copyDeviceToDevice(BackwardYPrev, BackwardY, bufferSize);
    gpuSolver->copyDeviceToDevice(BackwardZPrev, BackwardZ, bufferSize);

    gpuSolver->copyDeviceToDevice(BackwardX, InitX, bufferSize);
    gpuSolver->copyDeviceToDevice(BackwardY, InitY, bufferSize);
    gpuSolver->copyDeviceToDevice(BackwardZ, InitZ, bufferSize);

    gpuSolver->copyDeviceToDevice(ForwardX, InitX, bufferSize);
    gpuSolver->copyDeviceToDevice(ForwardY, InitY, bufferSize);
    gpuSolver->copyDeviceToDevice(ForwardZ, InitZ, bufferSize);
}

// Test code
void MapperBaseGPU::updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt)
{
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));

    updateBackward(gpuSolver->u, gpuSolver->v, gpuSolver->w, cfldt, dt);

    updateForward(gpuSolver->u, gpuSolver->v, gpuSolver->w, cfldt, dt);
}

void MapperBaseGPU::advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev)
{
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));

    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));

    gpuSolver->copyHostToDevice(u_prev, gpuSolver->u_host, gpuSolver->u_src, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(v_prev, gpuSolver->v_host, gpuSolver->v_src, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(w_prev, gpuSolver->w_host, gpuSolver->w_src, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));

    advectVelocity(gpuSolver->u, gpuSolver->v, gpuSolver->w, gpuSolver->du, gpuSolver->dv, gpuSolver->dw, gpuSolver->u_src, gpuSolver->v_src, gpuSolver->w_src);

    gpuSolver->copyDeviceToHost(un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(wn, gpuSolver->w_host, gpuSolver->w);
}

void MapperBaseGPU::advectField(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev)
{
    gpuSolver->copyHostToDevice(field, gpuSolver->u_host, gpuSolver->u, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(field_init, gpuSolver->u_host, gpuSolver->du, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(field_prev, gpuSolver->u_host, gpuSolver->u_src, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));

    advectField(gpuSolver->u, gpuSolver->du, gpuSolver->u_src);

    gpuSolver->copyDeviceToHost(field, gpuSolver->u_host, gpuSolver->u);
}

float MapperBaseGPU::estimateDistortion(const buffer3Dc &boundary)
{
    gpuSolver->estimateDistortionCUDA(gpuSolver->du, BackwardX, BackwardY, BackwardZ, ForwardX, ForwardY, ForwardZ, CellSize, CellNumberX, CellNumberY, CellNumberZ);
    cudaMemcpy(gpuSolver->u_host, gpuSolver->du, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float), cudaMemcpyDeviceToHost);

    float max_dist = 0.f;
    for (uint i = 0; i < CellNumberX; i++)
    {
        for (uint j = 0; j < CellNumberY; j++)
        {
            for (uint k = 0; k < CellNumberZ; k++)
            {
                if (boundary(i,j,k) != 2)
                {
                    uint index = i + CellNumberX*j + CellNumberX * CellNumberY * k;
                    max_dist = max(max_dist, gpuSolver->u_host[index]);
                }
            }
        }
    }
    
    return sqrt(max_dist);
}

void MapperBaseGPU::accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff)
{
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));
    
    gpuSolver->copyHostToDevice(u_change, gpuSolver->u_host, gpuSolver->u, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(v_change, gpuSolver->v_host, gpuSolver->v, CellNumberX*(CellNumberY+1)*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(w_change, gpuSolver->w_host, gpuSolver->w, CellNumberX*CellNumberY*(CellNumberZ+1)*sizeof(float));

    accumulateVelocity(gpuSolver->du, gpuSolver->dv, gpuSolver->dw, gpuSolver->u, gpuSolver->v, gpuSolver->w, coeff);

    gpuSolver->copyDeviceToHost(u_init, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(v_init, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(w_init, gpuSolver->w_host, gpuSolver->dw);
}

void MapperBaseGPU::accumulateField(buffer3Df &field_init, const buffer3Df &field_change)
{
    gpuSolver->copyHostToDevice(field_init, gpuSolver->u_host, gpuSolver->du, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));
    gpuSolver->copyHostToDevice(field_change, gpuSolver->u_host, gpuSolver->u, (CellNumberX+1)*CellNumberY*CellNumberZ*sizeof(float));

    accumulateField(gpuSolver->du, gpuSolver->u);

    gpuSolver->copyDeviceToHost(field_init, gpuSolver->u_host, gpuSolver->du);
}