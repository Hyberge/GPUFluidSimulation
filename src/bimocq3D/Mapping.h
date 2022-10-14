#ifndef BIMOCQ_MAPPING_H
#define BIMOCQ_MAPPING_H

#include <iostream>
#include <cstdint>
#include "tbb/tbb.h"
#include "../utils/color_macro.h"
#include "../include/fluid_buffer3D.h"
#include "GPU_Advection.h"

// two level BIMOCQ advector
class MapperBase
{
public:
    MapperBase() = default;
    virtual ~MapperBase() = default;

    virtual void init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper);
    virtual void updateForward(float cfldt, float dt);
    virtual void updateBackward(float cfldt, float dt);
    virtual void updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt);

    virtual void accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff);
    virtual void accumulateField(buffer3Df &field_init, const buffer3Df &field_change);
    virtual float estimateDistortion(const buffer3Dc &boundary);

    virtual void reinitializeMapping();
    virtual void advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev);
    virtual void advectField(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev);

    float _h;
    // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
    float blend_coeff;
    uint total_reinit_count;
    uint _ni, _nj, _nk;
    buffer3Df forward_x, forward_y, forward_z;
    buffer3Df backward_x, backward_y, backward_z;
    buffer3Df backward_xprev, backward_yprev, backward_zprev;
    /// gpu solver
    gpuMapper *gpuSolver;
};

class MapperBaseGPU
{
public:
    MapperBaseGPU() = default;
    ~MapperBaseGPU() = default;

    void init(uint ni, uint nj, uint nk, float h, float coeff, VirtualGpuMapper *mymapper);

    void updateForward(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt);
    void updateBackward(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt);
    void updateMapping(float *velocityU, float *velocityV, float *velocityW, float cfldt, float dt);

    void advectVelocity(float *velocityU, float *velocityV, float *velocityW,
                        float *velocityUInit, float *velocityVInit, float *velocityWInit,
                        float *velocityUPrev, float *velocityVPrev, float *velocityWPrev);
    void advectField(float *field, float *fieldInit, float *fieldPrev);

    //float estimateDistortion(float *boundary);

    void accumulateVelocity(float *uChange, float *vChange, float *wChange, float *uInit, float *vInit, float *wInit, float coeff);

    void accumulateField(float *fieldChange, float *dfieldInit);

    void reinitializeMapping();

private:
    uint   CellNumberX, CellNumberY, CellNumberZ;
    float  CellSize;
    float  BlendCoeff;      // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
    uint   TotalReinitCount;
    float *ForwardX = nullptr, *ForwardY = nullptr, *ForwardZ = nullptr;
    float *BackwardX = nullptr, *BackwardY = nullptr, *BackwardZ = nullptr;
    float *BackwardXPrev = nullptr, *BackwardYPrev = nullptr, *BackwardZPrev = nullptr;

    float *InitX = nullptr, *InitY = nullptr, *InitZ = nullptr;

    float *Distortion = nullptr;

    float* TempX;
    float* TempY;
    float* TempZ;

    VirtualGpuMapper *GpuSolver;
};

#endif //BIMOCQ_MAPPING_H
