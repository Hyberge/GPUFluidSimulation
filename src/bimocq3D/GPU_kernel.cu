#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include "GPU_Advection.h"
#include <iostream>


__device__ float clamp(float a, float minv, float maxv)
{
    return fminf(fmaxf(minv, a),maxv);
}

__device__ float3 clampv3(float3 in, float3 minv, float3 maxv)
{
    float xout = clamp(in.x,minv.x,maxv.x);
    float yout = clamp(in.y,minv.y,maxv.y);
    float zout = clamp(in.z,minv.z,maxv.z);
    return make_float3(xout, yout, zout);
}

__device__ float lerp(float a, float b, float c)
{
    return (1.0-c)*a + c*b;
}

__device__ float triLerp(float v000, float v001, float v010, float v011, float v100, float v101,
        float v110, float v111, float a, float b, float c)
{
    return lerp(
            lerp(
                    lerp(v000, v001, a),
                    lerp(v010, v011, a),
                    b),
            lerp(
                    lerp(v100, v101, a),
                    lerp(v110, v111, a),
                    b),
            c);

}

__device__ float sample_buffer(float * b, int nx, int ny, int nz, float h, float3 off_set, float3 pos)
{
    float3 samplepos = make_float3(pos.x-off_set.x, pos.y-off_set.y, pos.z-off_set.z);
    int i = int(floorf(samplepos.x/h));
    int j = int(floorf(samplepos.y/h));
    int k = int(floorf(samplepos.z/h));
    float fx = samplepos.x/h - float(i);
    float fy = samplepos.y/h - float(j);
    float fz = samplepos.z/h - float(k);

    int idx000 = i + nx*j + nx*ny*k;
    int idx001 = i + nx*j + nx*ny*k + 1;
    int idx010 = i + nx*j + nx*ny*k + nx;
    int idx011 = i + nx*j + nx*ny*k + nx + 1;
    int idx100 = i + nx*j + nx*ny*k + nx*ny;
    int idx101 = i + nx*j + nx*ny*k + nx*ny + 1;
    int idx110 = i + nx*j + nx*ny*k + nx*ny + nx;
    int idx111 = i + nx*j + nx*ny*k + nx*ny + nx + 1;
    return triLerp(b[idx000], b[idx001],b[idx010],b[idx011],b[idx100],b[idx101],b[idx110],b[idx111], fx, fy, fz);
}

__device__ float3 getVelocity(float *u, float *v, float *w, float h, float nx, float ny, float nz, float3 pos)
{

    float _u = sample_buffer(u, nx+1, ny, nz, h, make_float3(-0.5*h,0,0), pos);
    float _v = sample_buffer(v, nx, ny+1, nz, h, make_float3(0,-0.5*h,0), pos);
    float _w = sample_buffer(w, nx, ny, nz+1, h, make_float3(0,0,-0.5*h), pos);

    return make_float3(_u,_v,_w);
}

__device__ float3 traceRK3(float *u, float *v, float *w, float h, int ni, int nj, int nk, float dt, float3 pos)
{
    float c1 = 2.0/9.0*dt, c2 = 3.0/9.0 * dt, c3 = 4.0/9.0 * dt;
    float3 input = pos;
    float3 v1 = getVelocity(u,v,w,h,ni,nj,nk, input);
    float3 midp1 = make_float3(input.x + 0.5*dt*v1.x, input.y + 0.5*dt*v1.y, input.z + 0.5*dt*v1.z);
    float3 v2 = getVelocity(u,v,w,h,ni,nj,nk, midp1);
    float3 midp2 = make_float3(input.x + 0.75*dt*v2.x, input.y + 0.75*dt*v2.y, input.z + 0.75*dt*v2.z);
    float3 v3 = getVelocity(u,v,w,h,ni,nj,nk, midp2);

    float3 output = make_float3(input.x + c1*v1.x + c2*v2.x + c3*v3.x,
                                input.y + c1*v1.y + c2*v2.y + c3*v3.y,
                                input.z + c1*v1.z + c2*v2.z + c3*v3.z);
    output = clampv3(output, make_float3(h,h,h),
            make_float3(float(ni) * h - h, float(nj) * h - h, float(nk) * h - h ));
    return output;
}

__device__ float3 trace(float *u, float *v, float *w, float h, int ni, int nj, int nk, float cfldt, float dt, float3 pos)
{
    if(dt>0)
    {
        float T = dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK3(u,v,w,h,ni,nj,nk,substep,opos);

            t+=substep;
        }
        return opos;
    }
    else
    {
        float T = -dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK3(u,v,w,h,ni,nj,nk,-substep,opos);
            t+=substep;
        }
        return opos;
    }
}

__global__ void forward_kernel(float *u, float *v, float *w,
                            float *x_fwd, float *y_fwd, float *z_fwd,
                            float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(x_fwd[index], y_fwd[index], z_fwd[index]);
        float3 pointout = trace(u,v,w,h,ni,nj,nk,cfldt,dt,point);
        x_fwd[index] = pointout.x;
        y_fwd[index] = pointout.y;
        z_fwd[index] = pointout.z;
    }
    __syncthreads();
}

__global__ void clampExtrema_kernel(float *before, float *after, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    float max_value = before[index];
    float min_value = before[index];
    if (i>0 && i<ni-1 && j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        for(int kk=k-1;kk<=k+1;kk++)for(int jj=j-1;jj<=j+1;jj++)for(int ii=i-1;ii<=i+1;ii++)
        {
            int idx = ii + jj*ni + kk*ni*nj;
            if(before[idx]>max_value)
                max_value = before[idx];
            if(before[idx]<min_value)
                min_value = before[idx];
        }
        after[index] = min(max(min_value, after[index]), max_value);
    }
    __syncthreads();
}

__global__ void DMC_backward_kernel(float *u, float *v, float *w,
                                    float *x_in, float *y_in, float *z_in,
                                    float *x_out, float *y_out, float *z_out,
                                    float h, int ni, int nj, int nk, float substep)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));

        float3 vel = getVelocity(u, v, w, h, ni, nj, nk, point);

        float temp_x = (vel.x > 0)? point.x - h: point.x + h;
        float temp_y = (vel.y > 0)? point.y - h: point.y + h;
        float temp_z = (vel.z > 0)? point.z - h: point.z + h;
        float3 temp_point = make_float3(temp_x, temp_y, temp_z);
        float3 temp_vel = getVelocity(u, v, w, h, ni, nj, nk, temp_point);

        float a_x = (vel.x - temp_vel.x) / (point.x - temp_point.x);
        float a_y = (vel.y - temp_vel.y) / (point.y - temp_point.y);
        float a_z = (vel.z - temp_vel.z) / (point.z - temp_point.z);

        float new_x = (fabs(a_x) > 1e-4)? point.x - (1 - exp(-a_x*substep))*vel.x/a_x : point.x - vel.x*substep;
        float new_y = (fabs(a_y) > 1e-4)? point.y - (1 - exp(-a_y*substep))*vel.y/a_y : point.y - vel.y*substep;
        float new_z = (fabs(a_z) > 1e-4)? point.z - (1 - exp(-a_z*substep))*vel.z/a_z : point.z - vel.z*substep;
        float3 pointnew = make_float3(new_x, new_y, new_z);

        x_out[index] = sample_buffer(x_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        y_out[index] = sample_buffer(y_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        z_out[index] = sample_buffer(z_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
    }
    __syncthreads();
}

__global__ void semilag_kernel(float *field, float *field_src,
                               float *u, float *v, float *w,
                               int dim_x, int dim_y, int dim_z,
                               float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float3 buffer_origin = make_float3(-float(dim_x)*0.5f*h, -float(dim_y)*0.5f*h, -float(dim_z)*0.5f*h);

    int field_buffer_i = ni + dim_x;
    int field_buffer_j = nj + dim_y;
    int field_buffer_k = nk + dim_z;

    int i = index % field_buffer_i;
    int j = (index % (field_buffer_i * field_buffer_j)) / field_buffer_i;
    int k = index/(field_buffer_i*field_buffer_j);

    if (i > 1 && i < field_buffer_i-2-dim_x && j > 1 && j < field_buffer_j-2-dim_y && k > 1 && k < field_buffer_k-2-dim_z)
    {
        float3 point = make_float3(h*float(i) + buffer_origin.x,
                                   h*float(j) + buffer_origin.y,
                                   h*float(k) + buffer_origin.z);

        float3 pointnew = trace(u, v, w, h, ni, nj, nk, cfldt, dt, point);

        field[index] = sample_buffer(field_src, field_buffer_i, field_buffer_j, field_buffer_k, h, buffer_origin, pointnew);
    }
    __syncthreads();
}


__global__ void doubleAdvect_kernel(float *field, float *temp_field,
                                    float *backward_x, float *backward_y, float * backward_z,
                                    float *backward_xprev, float *backward_yprev, float *backward_zprev,
                                    float h, int ni, int nj, int nk,
                                    int dimx, int dimy, int dimz, bool is_point, float blend_coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);


    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }


    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);
            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 midpos = make_float3(x_init, y_init, z_init);
            midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
            float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float3 finalpos = make_float3(x_orig, y_orig, z_orig);

            finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);
        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 midpos = make_float3(x_init, y_init, z_init);
        midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
        float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float3 finalpos = make_float3(x_orig, y_orig, z_orig);

        finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        float prev_value = 0.5f*(sum + value);
        field[index] = field[index]*blend_coeff + (1-blend_coeff)*prev_value;
    }
    __syncthreads();
}

__global__ void advect_kernel(float *field, float *field_init,
                              float *backward_x, float *backward_y, float *backward_z,
                              float h, int ni, int nj, int nk,
                              int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);

            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 pos_init = make_float3(x_init, y_init, z_init);

            pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);

        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 pos_init = make_float3(x_init, y_init, z_init);

        pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        field[index] = 0.5f*sum + 0.5f*value;
    }
    __syncthreads();
}

__global__ void cumulate_kernel(float *dfield, float *dfield_init,
                                float *x_map, float *y_map, float *z_map,
                                float h, int ni, int nj, int nk,
                                int dimx, int dimy, int dimz, bool is_point, float coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            // forward mapping position
            // also used in compensation
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5 * value;
        dfield_init[index] += sum;
    }
    __syncthreads();
}

__global__ void compensate_kernel(float *src_buffer, float *temp_buffer, float *test_buffer,
                                  float *x_map, float *y_map, float *z_map,
                                  float h, int ni, int nj, int nk,
                                  int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5*value;
        test_buffer[index] = sum - temp_buffer[index];
//        sum -= temp_buffer[index];
//        sum *= 0.5f;
//        temp_buffer[index] = sum;
    }
    __syncthreads();
}

__global__ void estimate_kernel(float *dist_buffer, float *x_first, float *y_first, float *z_first,
                                float *x_second, float *y_second, float *z_second,
                                float h, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>1 && i<ni-2 && j>1 && j<nj-2 && k>1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));
        // backward then forward
        float back_x = sample_buffer(x_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float back_y = sample_buffer(y_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float back_z = sample_buffer(z_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 back_pos = make_float3(back_x, back_y, back_z);
        float fwd_x = sample_buffer(x_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float fwd_y = sample_buffer(y_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float fwd_z = sample_buffer(z_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float dist_bf = (point.x-fwd_x)*(point.x-fwd_x) +
                        (point.y-fwd_y)*(point.y-fwd_y) +
                        (point.z-fwd_z)*(point.z-fwd_z);
        // forward then backward
        fwd_x = sample_buffer(x_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        fwd_y = sample_buffer(y_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        fwd_z = sample_buffer(z_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 fwd_pos = make_float3(fwd_x, fwd_y, fwd_z);
        back_x = sample_buffer(x_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        back_y = sample_buffer(y_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        back_z = sample_buffer(z_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        float dist_fb = (point.x-back_x)*(point.x-back_x) +
                        (point.y-back_y)*(point.y-back_y) +
                        (point.z-back_z)*(point.z-back_z);
        dist_buffer[index] = max(dist_bf, dist_fb);
    }
    __syncthreads();
}

__global__ void reduce0(float *g_idata, float *g_odata, int N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    sdata[tid] = (i<N)?g_idata[i]:0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s && i < N)
        {
            sdata[tid] = max(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void add_kernel(float *field1, float *field2, float coeff)
{
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    field1[i] += coeff*field2[i];
    __syncthreads();
}

extern "C" void gpu_solve_forward(float *u, float *v, float *w,
                                  float *x_fwd, float *y_fwd, float *z_fwd,
                                  float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    forward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_fwd, y_fwd, z_fwd, h, ni, nj, nk, cfldt, dt);
}

extern "C" void gpu_solve_backwardDMC(float *u, float *v, float *w,
                                      float *x_in, float *y_in, float *z_in,
                                      float *x_out, float *y_out, float *z_out,
                                      float h, int ni, int nj, int nk, float substep)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    DMC_backward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_in, y_in, z_in, x_out, y_out, z_out, h, ni, nj, nk, substep);
}

extern "C" void gpu_advect_velocity(float *u, float *v, float *w,
                                    float *u_init, float *v_init, float *w_init,
                                    float *backward_x, float *backward_y, float *backward_z,
                                    float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    advect_kernel<<< numBlocks_u, blocksize >>>(u, u_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    advect_kernel<<< numBlocks_v, blocksize >>>(v, v_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    advect_kernel<<< numBlocks_w, blocksize >>>(w, w_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point);
}

extern "C" void gpu_advect_vel_double(float *u, float *v, float *w,
                                      float *utemp, float *vtemp, float *wtemp,
                                      float *backward_x, float *backward_y, float *backward_z,
                                      float *backward_xprev,  float *backward_yprev,  float *backward_zprev,
                                      float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    doubleAdvect_kernel<<< numBlocks_u, blocksize >>> (u, utemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 1, 0, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_v, blocksize >>> (v, vtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 1, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_w, blocksize >>> (w, wtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 1, is_point, blend_coeff);
}

extern "C" void gpu_advect_field(float *field, float *field_init,
                                 float *backward_x, float *backward_y, float *backward_z,
                                 float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    advect_kernel<<< numBlocks, blocksize >>>(field, field_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point);
}

extern "C" void gpu_advect_field_double(float *field, float *field_prev,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float *backward_xprev, float *backward_yprev,   float *backward_zprev,
                                        float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    doubleAdvect_kernel<<< numBlocks, blocksize >>> (field, field_prev, backward_x, backward_y, backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 0, is_point, blend_coeff);
}

extern "C" void gpu_compensate_velocity(float *u, float *v, float *w,
                                        float *du, float *dv, float *dw,
                                        float *u_src, float *v_src, float *w_src,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    // error at time 0 will be in du, dv, dw
    compensate_kernel<<< numBlocks_u, blocksize>>>(u, du, u_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    compensate_kernel<<< numBlocks_v, blocksize>>>(v, dv, v_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    compensate_kernel<<< numBlocks_w, blocksize>>>(w, dw, w_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.u, gpu.v, gpu.w
    cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dv, v, sizeof(float)*ni*(nj+1)*nk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dw, w, sizeof(float)*ni*nj*(nk+1), cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks_u, blocksize >>>(u_src, u, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_v, blocksize >>>(v_src, v, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_w, blocksize >>>(w_src, w, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.u, gpu.v, gpu.w
    clampExtrema_kernel<<< numBlocks_u, blocksize >>>(du, u, ni+1, nj, nk);
    clampExtrema_kernel<<< numBlocks_v, blocksize >>>(dv, v, ni, nj+1, nk);
    clampExtrema_kernel<<< numBlocks_w, blocksize >>>(dw, w, ni, nj, nk+1);
}

extern "C" void gpu_compensate_field(float *u, float *du, float *u_src,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float *backward_x, float *backward_y, float *backward_z,
                                     float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    // error at time 0 will be in du
    compensate_kernel<<< numBlocks_u, blocksize>>>(u, du, u_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.u
    cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks_u, blocksize >>>(u_src, u, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.u
    clampExtrema_kernel<<< numBlocks_u, blocksize >>>(du, u, ni, nj, nk);
}

extern "C" void gpu_accumulate_velocity(float *u_change, float *v_change, float *w_change,
                                        float *du_init, float *dv_init, float *dw_init,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    cumulate_kernel<<< numBlocks_u, blocksize >>> (u_change, du_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_v, blocksize >>> (v_change, dv_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_w, blocksize >>> (w_change, dw_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point, coeff);
}

extern "C" void gpu_accumulate_field(float *field_change, float *dfield_init,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    cumulate_kernel<<< numBlocks, blocksize >>> (field_change, dfield_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point, coeff);
}

extern "C" void gpu_estimate_distortion(float *du,
                                        float *x_back, float *y_back, float *z_back,
                                        float *x_fwd, float *y_fwd, float *z_fwd,
                                        float h, int ni, int nj, int nk)
{
    int blocksize = 256;
    int est_numBlocks = ((ni*nj*nk) + 255)/256;
    // distortion will be stored in gpu.du
    estimate_kernel<<< est_numBlocks, blocksize>>> (du, x_back, y_back, z_back, x_fwd, y_fwd, z_fwd, h, ni, nj, nk);
}

extern "C" void gpu_semilag(float *field, float *field_src,
                            float *u, float *v, float *w,
                            int dim_x, int dim_y, int dim_z,
                            float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int blocksize = 256;
    int total_num = (ni+dim_x)*(nj+dim_y)*(nk+dim_z);
    int numBlocks = (total_num + 255)/256;
    semilag_kernel<<<numBlocks, blocksize>>>(field, field_src, u, v, w, dim_x, dim_y, dim_z, h, ni, nj, nk, cfldt, dt);
}

extern "C" void gpu_add(float *field1, float *field2, float coeff, int number)
{
    int blocksize = 256;
    int numBlocks = (number + 255)/256;
    add_kernel<<<numBlocks, blocksize>>>(field1, field2, coeff);
}

__global__ void emit_smoke_velocity_kernel(float *field,
                            float h, int ni, int nj, int nk, 
                            float centerX, float centerY, float centerZ, float radius, float emiter)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 dir = make_float3(((float)i - 0.5) * h - centerX, j * h - centerY, k * h - centerZ);
        
        float length = norm3df(dir.x, dir.y ,dir.z);
        if (length < radius)
        {
            float theta = acosf(dir.y / hypotf(dir.y, dir.z));

            float vel_x = emiter * 0.06 *(1.0 + 0.01*cosf(8.0*theta));
            field[index] = vel_x;
        }
    }
    __syncthreads();
}

__global__ void emit_smoke_field_kernel(float *rho, float *T,
                            float h, int ni, int nj, int nk, 
                            float centerX, float centerY, float centerZ, float radius, float density, float temperature)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 dir = make_float3(i * h - centerX, j * h - centerY, k * h - centerZ);
        
        float length = norm3df(dir.x, dir.y ,dir.z);
        if (length < radius)
        {
            rho[index] = density;
            T[index] = temperature;
        }
    }
    __syncthreads();
}

extern "C" void gpu_emit_smoke(float *u, float *v, float *w, float *rho, float *T,
                            float h, int ni, int nj, int nk, 
                            float centerX, float centerY, float centerZ, float radius, float density, float temperature, float emiter)
{
    int blocksize = 256;
    int number = (ni+1) * nj * nk;
    int numBlocks = (number + 255)/256;
    emit_smoke_velocity_kernel<<<numBlocks, blocksize>>>(u, h, ni+1, nj, nk, centerX, centerY, centerZ, radius, emiter);

    number = ni * (nj + 1) * nk;
    numBlocks = (number + 255)/256;
    emit_smoke_velocity_kernel<<<numBlocks, blocksize>>>(v, h, ni, nj+1, nk, centerX, centerY, centerZ, radius, 0);

    number = ni * nj * (nk+1);
    numBlocks = (number + 255)/256;
    emit_smoke_velocity_kernel<<<numBlocks, blocksize>>>(w, h, ni, nj, nk+1, centerX, centerY, centerZ, radius, 0);

    number = ni * nj * nk;
    numBlocks = (number + 255)/256;
    emit_smoke_field_kernel<<<numBlocks, blocksize>>>(rho, T, h, ni, nj, nk, centerX, centerY, centerZ, radius, density, temperature);
}

__global__ void add_buoyancy_kernel(float *field, float* density, float* temperature,
                            int ni, int nj, int nk, float alpha, float beta, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i<ni&& j > 0 && j<nj && k<nk)
    {
        float d0 = density[index];
        float T0 = temperature[index];
        int index1 = k*ni*nj + (j-1)*ni + i;
        float d1 = density[index1];
        float T1 = temperature[index1];
        float f = 0.5 * dt * (beta*(T0+T1) - alpha*(d0+d1));

        field[index] += f;
    }
    __syncthreads();
}

extern "C" void gpu_add_buoyancy(float *field, float* density, float* temperature,
                            int ni, int nj, int nk, float alpha, float beta, float dt)
{
    int blocksize = 256;
    int number = ni * (nj + 1) * nk;
    int numBlocks = (number + 255)/256;
    add_buoyancy_kernel<<<numBlocks, blocksize>>>(field, density, temperature, ni, nj+1, nk, alpha, beta, dt);
}

__global__ void diffuse_field_kernel(float *field, float *field_in, float* field_out, int ni, int nj, int nk, float coef)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        float value = field[index];
        float value0 = field_in[k*ni*nj + j*ni + i - 1];
        float value1 = field_in[k*ni*nj + j*ni + i + 1];
        float value2 = field_in[k*ni*nj + (j - 1)*ni + i];
        float value3 = field_in[k*ni*nj + (j + 1)*ni + i];
        float value4 = field_in[(k - 1)*ni*nj + j*ni + i];
        float value5 = field_in[(k + 1)*ni*nj + j*ni + i];

        field_out[index] = (value + coef*(value0 + value1 + value2 + value3 + value4 + value5)) / (1.0f + 6.0f*coef);
    }
    __syncthreads();
}

extern "C" void gpu_diffuse_field(float *field, float *fieldTemp0, float *filedTemp1, int ni, int nj, int nk, int iter, float coef)
{
    int blocksize = 256;
    int number = ni * nj * nk;
    int numBlocks = (number + 255)/256;

    float *in = fieldTemp0;
    float *out = filedTemp1;

    cudaMemcpy(in, field, number * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < iter; i++)
    {
        diffuse_field_kernel<<<numBlocks, blocksize>>>(field, in, out, ni, nj, nk, coef);

        float *temp = out;
        out = in;
        in = temp;
    }
    
    cudaMemcpy(field, out, number * sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ void add_field_kernel(float *out, float *field1, float *field2, float coeff)
{
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    out[i] = field1[i] + coeff*field2[i];
    __syncthreads();
}

extern "C" void gpu_add_field(float *out, float *field1, float *field2, float coeff, int number)
{
    int blocksize = 256;
    int numBlocks = (number + 255)/256;
    add_field_kernel<<<numBlocks, blocksize>>>(out, field1, field2, coeff);
}

__global__ void clamp_extrema_kernel(float *field, float *fieldTemp, float *u, float *v, float *w, int ni, int nj, int nk, int dimx, int dimy, int dimz, float ox, float oy, float oz, float h, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i<ni&& j<nj && k<nk)
    {
        float3 point = make_float3(h*(float(i)+ox),h*(float(j)+oy),h*(float(k)+oz));

        //float3 vel = getVelocity(u, v, w, h, ni, nj, nk, point);
        float3 vel = getVelocity(u, v, w, h, ni-dimx, nj-dimy, nk-dimz, point);

        float halfdt = 0.5f * dt;
        float3 px = make_float3(point.x - vel.x*halfdt, point.y - vel.y*halfdt, point.z - vel.z*halfdt);

        //vel = getVelocity(u, v, w, h, ni, nj, nk, px);
        vel = getVelocity(u, v, w, h, ni-dimx, nj-dimy, nk-dimz, px);

        px = make_float3(point.x - vel.x*dt, point.y - vel.y*dt, point.z - vel.z*dt);

        int grid_i = (int)floor(px.x);
		int grid_j = (int)floor(px.y);
		int grid_k = (int)floor(px.z);

        float cx = px.x - (float)grid_i;
		float cy = px.y - (float)grid_j;
		float cz = px.z - (float)grid_k;

        float v0 = field[grid_k * nj * ni + grid_j * ni + grid_i];
        float v1 = field[grid_k * nj * ni + grid_j * ni + grid_i + 1];
        float v2 = field[grid_k * nj * ni + (grid_j+1) * ni + grid_i];
        float v3 = field[grid_k * nj * ni + (grid_j+1) * ni + grid_i + 1];
        float v4 = field[(grid_k+1) * nj * ni + grid_j * ni + grid_i];
        float v5 = field[(grid_k+1) * nj * ni + grid_j * ni + grid_i + 1];
        float v6 = field[(grid_k+1) * nj * ni + (grid_j+1) * ni + grid_i];
        float v7 = field[(grid_k+1) * nj * ni + (grid_j+1) * ni + grid_i + 1];

        float min_value = min(v0,min(v1,min(v2,min(v3,min(v4,min(v5,min(v6,v7)))))));
        float max_value = max(v0,max(v1,max(v2,max(v3,max(v4,max(v5,max(v6,v7)))))));

        float temp = fieldTemp[grid_k * nj * ni + grid_j * ni + grid_i];
        if (temp < min_value || temp > max_value)
        {
            float iv1 = lerp(lerp(v0, v1, cx), lerp(v2, v3, cx), cy);
            float iv2 = lerp(lerp(v4, v5, cx), lerp(v6, v7, cx), cy);

            fieldTemp[grid_k * nj * ni + grid_j * ni + grid_i] = lerp(iv1, iv2, cz);
        }
    }
}

extern "C" void gpu_clamp_extrema(float *field, float *fieldTemp, float *u, float *v, float *w, int ni, int nj, int nk, int dimx, int dimy, int dimz, float ox, float oy, float oz, float h, float dt)
{
    int blocksize = 256;
    int number = ni * nj * nk;
    int numBlocks = (number + 255)/256;
    clamp_extrema_kernel<<<numBlocks, blocksize>>>(field, fieldTemp, u, v, w, ni, nj, nk, dimx, dimy, dimz, ox, oy, oz, h, dt);
}

__global__ void mad_kernel(float *field, float *field1, float *field2, float coeff1, float coeff2)
{
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    field[i] = coeff1*field1[i] + coeff2*field2[i];
    __syncthreads();
}

extern "C" void gpu_mad(float *field, float *field1, float *field2, float coeff1, float coeff2, int number)
{
    int blocksize = 256;
    int numBlocks = (number + 255)/256;
    mad_kernel<<<numBlocks, blocksize>>>(field, field1, field2, coeff1, coeff2);
}

// Conjugate Gradient solver
__global__ void divergence_kernel(float *u, float *v, float *w, float *div, int ni, int nj, int nk, float halfrdx/*0.5 / gridscale*/)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>=0 && i<ni&& j>=0 && j<nj && k>=0 && k<nk)
    {
        float u_left = u[k*(ni+1)*nj + j*(ni+1) + i];
        float u_right = u[k*(ni+1)*nj + j*(ni+1) + i + 1];
        float v_front = v[k*ni*(nj+1) + (j)*ni + i];
        float v_back = v[k*ni*(nj+1) + (j + 1)*ni + i];
        float w_down = w[(k)*ni*nj + j*ni + i];
        float w_up = w[(k + 1)*ni*nj + j*ni + i];

        div[index] = halfrdx * ((u_right - u_left) + (v_back - v_front) + (w_up - w_down));
    }
    __syncthreads();
}
__global__ void divergence_kernel(float *u, float *v, float *w, double *div, int ni, int nj, int nk, double halfrdx/*0.5 / gridscale*/)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>=0 && i<ni&& j>=0 && j<nj && k>=0 && k<nk)
    {
        double u_left = u[k*(ni+1)*nj + j*(ni+1) + i];
        double u_right = u[k*(ni+1)*nj + j*(ni+1) + i + 1];
        double v_front = v[k*ni*(nj+1) + (j)*ni + i];
        double v_back = v[k*ni*(nj+1) + (j + 1)*ni + i];
        double w_down = w[(k)*ni*nj + j*ni + i];
        double w_up = w[(k + 1)*ni*nj + j*ni + i];

        div[index] = halfrdx * ((u_right - u_left) + (v_back - v_front) + (w_up - w_down));
    }
    __syncthreads();
}

__global__ void gradient_kernel(float *field, double *p, int ni, int nj, int nk, int dimx, int dimy, int dimz, double halfrdx)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    int pi = ni - dimx;
    int pj = nj - dimy;
    int pk = nk - dimz;
    if (i>1 && i<pi&& j>1 && j<pj && k>1 && k<pk)
    {
        double p0 = p[k*pj*pi + j*pi + i];
        double p1 = p[(k-dimz)*pj*pi + (j-dimy)*pi + i - dimx];

        field[index] -= float(halfrdx * (p0 - p1));
    }
    __syncthreads();
}
__global__ void gradient_kernel(float *field, float *p, int ni, int nj, int nk, int dimx, int dimy, int dimz, float halfrdx)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    int pi = ni - dimx;
    int pj = nj - dimy;
    int pk = nk - dimz;
    if (i>1 && i<pi&& j>1 && j<pj && k>1 && k<pk)
    {
        float p0 = p[k*pj*pi + j*pi + i];
        float p1 = p[(k-dimz)*pj*pi + (j-dimy)*pi + i - dimx];

        field[index] -= halfrdx * (p0 - p1);
    }
    __syncthreads();
}

/*
                          [0   1  0]
  return A * x, where A = [1  -4  1]
                          [0   1  0]
*/
template<typename T>
__device__ T calc_poisson_value(T *x, int i, int j, int k, int ni, int nj, int nk)
{
    T x_center = x[k*ni*nj + j*ni + i];
    T x_left   = x[k*ni*nj + j*ni + i - 1];
    T x_right  = x[k*ni*nj + j*ni + i + 1];
    T x_front  = x[k*ni*nj + (j - 1)*ni + i];
    T x_back   = x[k*ni*nj + (j + 1)*ni + i];
    T x_down   = x[(k - 1)*ni*nj + j*ni + i];
    T x_up     = x[(k + 1)*ni*nj + j*ni + i];

    return (x_left + x_right + x_front + x_back + x_down + x_up) - x_center*6;
}

__global__ void calc_poisson_kernel(float *x, float *b, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        b[index] = calc_poisson_value<float>(x, i, j, k, ni, nj, nk);
    }
}

__global__ void calc_poisson_kernel(double *x, double *b, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        b[index] = calc_poisson_value<double>(x, i, j, k, ni, nj, nk);
    }
}

template<typename T>
__device__ void dot_vector(T *v0, T *v1, T *output, T *sharedMem, int count)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < count)
    {
        sharedMem[threadIdx.x] = v0[index] * v1[index];
    }
    else
    {
        sharedMem[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x < 16)
    {
        float sum0 = sharedMem[threadIdx.x * 16 + 0]  + sharedMem[threadIdx.x * 16 + 1]  + sharedMem[threadIdx.x * 16 + 2]  + sharedMem[threadIdx.x * 16 + 3] +
                     sharedMem[threadIdx.x * 16 + 4]  + sharedMem[threadIdx.x * 16 + 5]  + sharedMem[threadIdx.x * 16 + 6]  + sharedMem[threadIdx.x * 16 + 7] +
                     sharedMem[threadIdx.x * 16 + 8]  + sharedMem[threadIdx.x * 16 + 9]  + sharedMem[threadIdx.x * 16 + 10] + sharedMem[threadIdx.x * 16 + 11] +
                     sharedMem[threadIdx.x * 16 + 12] + sharedMem[threadIdx.x * 16 + 13] + sharedMem[threadIdx.x * 16 + 14] + sharedMem[threadIdx.x * 16 + 15];
        
        sharedMem[threadIdx.x+256] = sum0;
    }

    if (threadIdx.x == 0)
    {
        float sum  = sharedMem[256+0]  + sharedMem[256+1]  + sharedMem[256+2]  + sharedMem[+3] +
                     sharedMem[256+4]  + sharedMem[256+5]  + sharedMem[256+6]  + sharedMem[+7] +
                     sharedMem[256+8]  + sharedMem[256+9]  + sharedMem[256+10] + sharedMem[+11] +
                     sharedMem[256+12] + sharedMem[256+13] + sharedMem[256+14] + sharedMem[+15];

        output[blockIdx.x] = sum;
    }
}

__shared__ float dotResultF[272];
__global__ void dot_vector_kernel(float *v0, float *v1, float *output, int count)
{
    dot_vector<float>(v0, v1, output, dotResultF, count);
}

__shared__ double dotResultD[272];
__global__ void dot_vector_kernel(double *v0, double *v1, double *output, int count)
{
    dot_vector<double>(v0, v1, output, dotResultD, count);
}

template<typename T>
__device__ void calc_sum(T *v, T *output, T *sharedMem, int count, int countPerThread, int iterIndex, bool useAbs)
{
    sharedMem[threadIdx.x] = 0;

    int indexStart = threadIdx.x * countPerThread;
    for (int i = 0; i < countPerThread; ++i)
    {
        if (indexStart + i < count)
        {
            if (useAbs)
            {
                sharedMem[threadIdx.x] += abs(v[indexStart + i]);
            }
            else
            {
                sharedMem[threadIdx.x] += v[indexStart + i];
            }
            
        }
    }

    __syncthreads();

    if (threadIdx.x < 16)
    {
        T sum0 = sharedMem[threadIdx.x * 16 + 0]  + sharedMem[threadIdx.x * 16 + 1]  + sharedMem[threadIdx.x * 16 + 2]  + sharedMem[threadIdx.x * 16 + 3] +
                 sharedMem[threadIdx.x * 16 + 4]  + sharedMem[threadIdx.x * 16 + 5]  + sharedMem[threadIdx.x * 16 + 6]  + sharedMem[threadIdx.x * 16 + 7] +
                 sharedMem[threadIdx.x * 16 + 8]  + sharedMem[threadIdx.x * 16 + 9]  + sharedMem[threadIdx.x * 16 + 10] + sharedMem[threadIdx.x * 16 + 11] +
                 sharedMem[threadIdx.x * 16 + 12] + sharedMem[threadIdx.x * 16 + 13] + sharedMem[threadIdx.x * 16 + 14] + sharedMem[threadIdx.x * 16 + 15];
        
        sharedMem[threadIdx.x+256] = sum0;
    }

    if (threadIdx.x == 0)
    {
        T sum  = sharedMem[256+0]  + sharedMem[256+1]  + sharedMem[256+2]  + sharedMem[256+3] +
                 sharedMem[256+4]  + sharedMem[256+5]  + sharedMem[256+6]  + sharedMem[256+7] +
                 sharedMem[256+8]  + sharedMem[256+9]  + sharedMem[256+10] + sharedMem[256+11]+
                 sharedMem[256+12] + sharedMem[256+13] + sharedMem[256+14] + sharedMem[256+15];

        output[iterIndex] = sum;
    }
}

__shared__ float sumResultF[272];
__global__ void calc_sum_kernel(float *v, float *output, int count, int countPerThread, int iterIndex, bool useAbs = false)
{
    calc_sum<float>(v, output, sumResultF, count, countPerThread, iterIndex, useAbs);
}

__shared__ double sumResultD[272];
__global__ void calc_sum_kernel(double *v, double *output, int count, int countPerThread, int iterIndex, bool useAbs = false)
{
    calc_sum<double>(v, output, sumResultD, count, countPerThread, iterIndex, useAbs);
}

template<typename T>
__device__ void calc_max(T *v, T *output, T *sharedMem, int count, int countPerThread, int iterIndex, bool useAbs = false)
{
    sharedMem[threadIdx.x] = 0;

    for (int i = 0; i < countPerThread; ++i)
    {
        int index = i * blockDim.x + threadIdx.x;
        if (index < count)
        {
            sharedMem[threadIdx.x] = max(v[index], sharedMem[threadIdx.x]);
        }
    }

    __syncthreads();

    if (threadIdx.x < 16)
    {
        T max0 = max(max(sharedMem[threadIdx.x * 16 + 0] ,sharedMem[threadIdx.x * 16 + 1] ), max(sharedMem[threadIdx.x * 16 + 2] , sharedMem[threadIdx.x * 16 + 3] ));
        T max1 = max(max(sharedMem[threadIdx.x * 16 + 4] ,sharedMem[threadIdx.x * 16 + 5] ), max(sharedMem[threadIdx.x * 16 + 6] , sharedMem[threadIdx.x * 16 + 7] ));
        T max2 = max(max(sharedMem[threadIdx.x * 16 + 8] ,sharedMem[threadIdx.x * 16 + 9] ), max(sharedMem[threadIdx.x * 16 + 10], sharedMem[threadIdx.x * 16 + 11]));
        T max3 = max(max(sharedMem[threadIdx.x * 16 + 12],sharedMem[threadIdx.x * 16 + 13]), max(sharedMem[threadIdx.x * 16 + 14], sharedMem[threadIdx.x * 16 + 15]));
        
        sharedMem[threadIdx.x+256] = max(max(max0, max1), max(max2, max3));
    }

    if (threadIdx.x == 0)
    {
        T max0 = max(max(sharedMem[256+0] ,sharedMem[256+1] ), max(sharedMem[256+2] , sharedMem[256+3] ));
        T max1 = max(max(sharedMem[256+4] ,sharedMem[256+5] ), max(sharedMem[256+6] , sharedMem[256+7] ));
        T max2 = max(max(sharedMem[256+8] ,sharedMem[256+9] ), max(sharedMem[256+10], sharedMem[256+11]));
        T max3 = max(max(sharedMem[256+12],sharedMem[256+13]), max(sharedMem[256+14], sharedMem[256+15]));

        output[iterIndex] = max(max(max0, max1), max(max2, max3));
    }
}

__shared__ double maxResultD[272];
__global__ void calc_max_kernel(double *v, double *output, int count, int countPerThread, int iterIndex, bool useAbs = false)
{
    calc_max<double>(v, output, maxResultD, count, countPerThread, iterIndex, useAbs);
}
__shared__ float maxResultF[272];
__global__ void calc_max_kernel(float *v, float *output, int count, int countPerThread, int iterIndex, bool useAbs = false)
{
    calc_max<float>(v, output, maxResultF, count, countPerThread, iterIndex, useAbs);
}

__global__ void update_residual_kernel(float *r, float *b, float *x, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        r[index] = b[index] - calc_poisson_value<float>(x, i, j, k, ni, nj, nk);
    }
}

__global__ void update_residual_kernel(double *r, double *b, double *x, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        r[index] = b[index] - calc_poisson_value<double>(x, i, j, k, ni, nj, nk);
    }
}

__global__ void update_residual_cg_kernel(double *r, double *aMulDir, double *alpha, int rIndex, int dIndex, int number)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < number)
    {
        double alphaValue = alpha[rIndex] / alpha[dIndex];
        r[index] -= alphaValue * aMulDir[index];
    }
}

__global__ void update_x_kernel(float *x, float *dir, float *alpha, int count, int rIndex, int dIndex)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < count)
    {
        x[index] += dir[index] * alpha[rIndex] / alpha[dIndex];
    }
}

__global__ void update_x_kernel(double *x, double *dir, double *alpha, int count, int rIndex, int dIndex)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < count)
    {
        x[index] += dir[index] * alpha[rIndex] / alpha[dIndex];
    }
}

__global__ void update_dir_kernel(float *dir, float *residual, float *beta, int count, int rIndex, int rPlusIndex)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < count)
    {
        dir[index] = residual[index] + dir[index] * beta[rPlusIndex] / beta[rIndex];
    }
}

__global__ void update_dir_kernel(double *dir, double *residual, double *beta, int count, int rIndex, int rPlusIndex)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < count)
    {
        dir[index] = residual[index] + dir[index] * beta[rPlusIndex] / beta[rIndex];
    }
}

__global__ void mul_kernel(float *result, float *field, float constant, int number)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < number)
    {
        result[index] = field[index] * constant;
    }
}

__global__ void mul_kernel(double *result, double *field, double constant, int number)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < number)
    {
        result[index] = field[index] * constant;
    }
}

__global__ void add_kernel(double *field0, double *field1, double coef, int number)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < number)
    {
        field0[index] += field1[index] * coef;
    }
}

extern "C" void gpu_conjugate_gradient(float *u, float *v, float *w , float *div, float *p, float *residual, float *dir, float *dotResult, int ni, int nj, int nk, int iter, float halfrdx)
{
    int blocksize = 256;
    int number = ni * nj * nk;
    int numBlocks = (number + 255)/256;
    divergence_kernel<<<numBlocks, blocksize>>>(u, v, w, div, ni, nj, nk, halfrdx);

    // we start with x = [0], so r0 == b
    size_t bufferSize = number * sizeof(float);
    //cudaMemcpy(residual, div, bufferSize, cudaMemcpyDeviceToDevice);
    update_residual_kernel<<<numBlocks, blocksize>>>(residual, div, p, ni, nj, nk);
    // first iterater, dir0 == -r0
    mul_kernel<<<numBlocks, blocksize>>>(dir, residual, 1, number);

    {
        calc_max_kernel<<<1, blocksize>>>(residual,  dotResult, number, (number + 255)/256, 2000);
    }

    int coutPerThread = (numBlocks + 255)/256;
    float *dotResidual;
    cudaMalloc(&dotResidual, bufferSize);
    float *dotDir;
    cudaMalloc(&dotDir, bufferSize);
    //float *dotTemp;
    //cudaMalloc(&dotTemp, 4 * sizeof(float));

    // r.transpose() * r
    dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, dotResidual, number);
    calc_sum_kernel<<<1, blocksize>>>(dotResidual, dotResult, numBlocks, coutPerThread, 0);

    for (size_t i = 0; i < iter; ++i)
    {
        // step-1: calculate alpha(i+1)
        {
            // dir.transpose() * A * dir
            calc_poisson_kernel<<<numBlocks, blocksize>>>(dir, dotResidual, ni, nj, nk); 
            dot_vector_kernel<<<numBlocks, blocksize>>>(dir, dotResidual, dotDir, number);
            calc_sum_kernel<<<1, blocksize>>>(dotDir, dotResult, numBlocks, coutPerThread, i*2+1);
        }

        // step-2: calculate x(i+1)
        {
            update_x_kernel<<<numBlocks, blocksize>>>(p, dir, dotResult, number, i*2, i*2+1);
        }

        // step-3: calculate r(i+1)
        {
            update_residual_kernel<<<numBlocks, blocksize>>>(residual, div, p, ni, nj, nk);
        }

        {
            calc_max_kernel<<<1, blocksize>>>(residual,  dotResult, number, (number + 255)/256, 2001+i);
        }

        // step-4: calculate beta(i+1)
        {
            dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, dotResidual, number);
            calc_sum_kernel<<<1, blocksize>>>(dotResidual, dotResult, numBlocks, coutPerThread, (i+1)*2);
        }

        // step-5: caclulate dir(i+1)
        {
            update_dir_kernel<<<numBlocks, blocksize>>>(dir, residual, dotResult, number, i*2, (i+1)*2);
        }
    }

    number = (ni + 1) * nj * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(u, p, ni + 1, nj, nk, 1, 0, 0, halfrdx);

    number = ni * (nj + 1) * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(v, p, ni, nj + 1, nk, 0, 1, 0, halfrdx);

    number = ni * nj * (nk + 1);
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(w, p, ni, nj, nk + 1, 0, 0, 1, halfrdx);

    cudaFree(dotResidual);
    cudaFree(dotDir);
}
// Conjugate Gradient solver end

// Multi-Grid Conjugate Gradient solver
__global__ void smoothing_jacobi_kernel(float *x, float *b, float *out, float alpha, float beta, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        // x == [0]
        float x_left  = x[k*nj*ni + j*ni + i - 1];
        float x_right = x[k*nj*ni + j*ni + i + 1];
        float x_front = x[k*nj*ni + (j-1)*ni + i];
        float x_back  = x[k*nj*ni + (j+1)*ni + i];
        float x_down  = x[(k-1)*nj*ni + j*ni + i];
        float x_up    = x[(k+1)*nj*ni + j*ni + i];

        out[index] = ((x_left + x_right + x_front + x_back + x_down + x_up) + alpha * b[index]) * beta;

        // normally, we will start iterater with coarseX == [0], so we simplify Jacobi to Mul.
        // x[index] = param * b[index];
    }
    __syncthreads();
}

__global__ void smoothing_jacobi_kernel(double *x, double *b, double *out, double alpha, double beta, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        double x_left  = x[k*nj*ni + j*ni + i - 1];
        double x_right = x[k*nj*ni + j*ni + i + 1];
        double x_front = x[k*nj*ni + (j-1)*ni + i];
        double x_back  = x[k*nj*ni + (j+1)*ni + i];
        double x_down  = x[(k-1)*nj*ni + j*ni + i];
        double x_up    = x[(k+1)*nj*ni + j*ni + i];

        out[index] = ((x_left + x_right + x_front + x_back + x_down + x_up) + alpha * b[index]) * beta;
    }
    __syncthreads();
}

template<typename T>
void smoothing_jacobi(T *x, T *b, T *tempX, T alpha, T beta, int ni, int nj, int nk, int iter)
{
    int blocksize = 256;
    int number = ni * nj * nk;
    int numBlocks = (number + 255)/256;

    if (iter % 2 == 1)
        iter += 1;

    T *x_in = x;
    T *x_out = tempX;
    for (int i = 0; i < iter; ++i)
    {
        smoothing_jacobi_kernel<<<numBlocks, blocksize>>>(x_in, b, x_out, alpha, beta, ni, nj, nk);

        // swap
        T *temp = x_in;
        x_in = x_out;
        x_out = temp;
    }
}

template<typename T>
void smoothing_conjugate_gradient(T *b, T *x, T *dir, T *residual, T *aMulDir, T *dotDir, T *tempResult, int offset, int ni, int nj, int nk, int blocksize, int number, int numBlocks, int coutPerThread)
{
    calc_poisson_kernel<<<numBlocks, blocksize>>>(dir, aMulDir, ni, nj, nk);
    dot_vector_kernel<<<numBlocks, blocksize>>>(dir, aMulDir, dotDir, number);
    calc_sum_kernel<<<1, blocksize>>>(dotDir, tempResult, numBlocks, coutPerThread, offset+1);

    update_x_kernel<<<numBlocks, blocksize>>>(x, dir, tempResult, number, offset, offset+1);

    update_residual_kernel<<<numBlocks, blocksize>>>(residual, b, x, ni, nj, nk);
}

template<typename T>
void updateDir(T *b, T *x, T *residual, T *dir, T *dotResidual, T *tempResult, int offset, int blocksize, int number, int numBlocks, int coutPerThread)
{
    dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, dotResidual, number);
    calc_sum_kernel<<<1, blocksize>>>(dotResidual, tempResult, numBlocks, coutPerThread, offset+2);

    update_dir_kernel<<<numBlocks, blocksize>>>(dir, residual, tempResult, number, offset, offset+2);
}

template<typename T>
__device__ T lerp_t(T a, T b, T c)
{
    return (1.0-c)*a + c*b;
}

template<typename T>
__device__ T triLerp_t(T v000, T v001, T v010, T v011, T v100, T v101,
        T v110, T v111, T a, T b, T c)
{
    return lerp(
            lerp(
                    lerp(v000, v001, a),
                    lerp(v010, v011, a),
                    b),
            lerp(
                    lerp(v100, v101, a),
                    lerp(v110, v111, a),
                    b),
            c);

}

template<typename T>
__device__ T sample_buffer(T * b, int nx, int ny, int nz, float3 pos)
{
    float3 samplepos = pos;
    int i = int(floorf(samplepos.x));
    int j = int(floorf(samplepos.y));
    int k = int(floorf(samplepos.z));
    T fx = samplepos.x - float(i);
    T fy = samplepos.y - float(j);
    T fz = samplepos.z - float(k);

    int idx000 = i + nx*j + nx*ny*k;
    int idx001 = i + nx*j + nx*ny*k + 1;
    int idx010 = i + nx*j + nx*ny*k + nx;
    int idx011 = i + nx*j + nx*ny*k + nx + 1;
    int idx100 = i + nx*j + nx*ny*k + nx*ny;
    int idx101 = i + nx*j + nx*ny*k + nx*ny + 1;
    int idx110 = i + nx*j + nx*ny*k + nx*ny + nx;
    int idx111 = i + nx*j + nx*ny*k + nx*ny + nx + 1;
    return triLerp_t(b[idx000], b[idx001],b[idx010],b[idx011],b[idx100],b[idx101],b[idx110],b[idx111], fx, fy, fz);
}

template<typename T>
__device__ void restriction(T *residual, T *coarseResidual, int ni, int nj, int nk, int ci, int cj, int ck)
{
    int coarseIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int i = coarseIndex%ci;
    int j = (coarseIndex%(ci*cj))/ci;
    int k = coarseIndex/(ci*cj);
    if (i < ci && j < cj && k < ck)
    {
        //T value = residual[(2*k+1)*ni*nj + (2*j+1)*ni + 2*i + 1];
        //T value0 = residual[(2*k+1)*ni*nj + (2*j+1)*nj + 2*i + 0];
        //T value1 = residual[(2*k+1)*ni*nj + (2*j+1)*ni + 2*i + 2];
        //T value2 = residual[(2*k+1)*ni*nj + (2*j+0)*ni + 2*i + 1];
        //T value3 = residual[(2*k+1)*ni*nj + (2*j+2)*ni + 2*i + 1];
        //T value4 = residual[(2*k+0)*ni*nj + (2*j+1)*ni + 2*i + 1];
        //T value5 = residual[(2*k+2)*ni*nj + (2*j+1)*ni + 2*i + 1];

        //coarseResidual[coarseIndex] = residual[(2*k+1)*ni*nj + (2*j+1)*ni + 2*i + 1];

        float3 point = make_float3(2*i + 0.5, 2*j + 0.5, 2*k + 0.5);
        T value0 = sample_buffer(residual, ni, nj, nk, point);

        point = make_float3(2*i + 0.5, 2*j + 0.5, 2*k + 1.5);
        T value1 = sample_buffer(residual, ni, nj, nk, point);
        
        point = make_float3(2*i + 0.5, 2*j + 1.5, 2*k + 0.5);
        T value2= sample_buffer(residual, ni, nj, nk, point);

        point = make_float3(2*i + 0.5, 2*j + 1.5, 2*k + 1.5);
        T value3 = sample_buffer(residual, ni, nj, nk, point);

        point = make_float3(2*i + 1.5, 2*j + 0.5, 2*k + 0.5);
        T value4 = sample_buffer(residual, ni, nj, nk, point);

        point = make_float3(2*i + 1.5, 2*j + 0.5, 2*k + 1.5);
        T value5 = sample_buffer(residual, ni, nj, nk, point);
        
        point = make_float3(2*i + 1.5, 2*j + 1.5, 2*k + 0.5);
        T value6 = sample_buffer(residual, ni, nj, nk, point);

        point = make_float3(2*i + 1.5, 2*j + 1.5, 2*k + 1.5);
        T value7 = sample_buffer(residual, ni, nj, nk, point);

        coarseResidual[coarseIndex] = (value0 + value1 + value2 + value3 +value4 + value5 + value6 + value7) / 8;
    }
}

__global__ void restriction_kernel(double *residual, double *coarseResidual, int ni, int nj, int nk, int ci, int cj, int ck)
{
    restriction<double>(residual, coarseResidual, ni, nj, nk, ci, cj, ck);
}

__global__ void restriction_kernel(float *residual, float *coarseResidual, int ni, int nj, int nk, int ci, int cj, int ck)
{
    restriction<float>(residual, coarseResidual, ni, nj, nk, ci, cj, ck);
}

__global__ void prolongation_kernel(double *x, double *coarseX, int ni, int nj, int nk, int ci, int cj, int ck)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 0 && i < ni-1 && j > 0 && j < nj-1 && k > 0 && k < nk-1)
    {
        float3 pos = make_float3(float(i)/2.f - 0.5, float(j)/2.f - 0.5, float(k)/2.f - 0.5);
        x[index] += sample_buffer(coarseX, ci, cj, ck, pos);
    }
}

__global__ void prolongation_kernel(float *x, float *coarseX, int ni, int nj, int nk, int ci, int cj, int ck)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 0 && i < ni && j > 0 && j < nj && k > 0 && k < nk)
    {
        float3 pos = make_float3(float(i)/2.f - 0.5, float(j)/2.f - 0.5, float(k)/2.f - 0.5);
        x[index] += sample_buffer(coarseX, ci, cj, ck, pos);
    }
}

void V_Cycle(double *b, double *x, double *residual, SCoarseLevelInfo* levels, double *temp0, double *tempResult, int levennum, int resultOffset)
{
    int blocksize = 256;
    double scale[6] = {1.0,1.0,1.0,1.0,1.0,1.0};
    
    if (0)
    {
        scale[1] = scale[3] = scale[5] = 8.0;

        cudaMemset(levels[0].x, 0, levels[0].number*sizeof(double));
        cudaMemcpy(levels[0].b, residual, levels[0].number*sizeof(double), cudaMemcpyDeviceToDevice);

        //restriction_kernel<<<(levels[1].number + blocksize - 1) / blocksize, blocksize>>>(levels[0].b, levels[1].b, levels[0].ni, levels[0].nj, levels[0].nk, levels[1].ni, levels[1].nj, levels[1].nk);

        for (int i = 0; i < levennum - 1; i++)
        {
            int numBlocks = (levels[i].number + blocksize - 1) / blocksize;
            int coarsenumBlocks = (levels[i+1].number + blocksize - 1) / blocksize;

            cudaMemset(levels[i].x, 0, levels[i].number*sizeof(double));
            if (i%2 == 1)
            {
                cudaMemset(temp0, 0, levels[0].number*sizeof(double));
                smoothing_jacobi<double>(levels[i].x, levels[i].b, temp0, levels[i].alpha*scale[i], levels[i].beta, levels[i].ni, levels[i].nj, levels[i].nk, 32);

                update_residual_kernel<<<numBlocks, blocksize>>>(levels[i].r, levels[i].b, levels[i].x, levels[i].ni, levels[i].nj, levels[i].nk);
            }
            else
            {
                cudaMemcpy(levels[i].r, levels[i].b, levels[i].number*sizeof(double), cudaMemcpyDeviceToDevice);
            }
            restriction_kernel<<<coarsenumBlocks, blocksize>>>(levels[i].r, levels[i+1].b, levels[i].ni, levels[i].nj, levels[i].nk, levels[i+1].ni, levels[i+1].nj, levels[i+1].nk);
        }
    }
    else
    {
        //scale[1] = scale[2] = scale[3] = scale[4] = scale[5] = 8.0;
        scale[1] = 8.0;
        cudaMemcpy(levels[0].b, residual, levels[0].number*sizeof(double), cudaMemcpyDeviceToDevice);

        for (int i = 0; i < levennum - 1; i++)
        {
            int numBlocks = (levels[i].number + blocksize - 1) / blocksize;
            int coarsenumBlocks = (levels[i+1].number + blocksize - 1) / blocksize;

            cudaMemset(temp0, 0, levels[0].number*sizeof(double));
            cudaMemset(levels[i].x, 0, levels[i].number*sizeof(double));
            smoothing_jacobi<double>(levels[i].x, levels[i].b, temp0, levels[i].alpha*scale[i], levels[i].beta, levels[i].ni, levels[i].nj, levels[i].nk, 32);

            update_residual_kernel<<<numBlocks, blocksize>>>(levels[i].r, levels[i].b, levels[i].x, levels[i].ni, levels[i].nj, levels[i].nk);
            restriction_kernel<<<coarsenumBlocks, blocksize>>>(levels[i].r, levels[i+1].b, levels[i].ni, levels[i].nj, levels[i].nk, levels[i+1].ni, levels[i+1].nj, levels[i+1].nk);
        }
    }

    cudaMemset(temp0, 0, levels[0].number*sizeof(double));
    cudaMemset(levels[levennum-1].x, 0, levels[levennum-1].number*sizeof(double));
    smoothing_jacobi<double>(levels[levennum-1].x, levels[levennum-1].b, temp0, levels[levennum-1].alpha*scale[levennum-1], levels[levennum-1].beta, levels[levennum-1].ni, levels[levennum-1].nj, levels[levennum-1].nk, 32);

    for (int i = levennum-2; i >= 0; --i)
    {
        int numBlocks = (levels[i].number + blocksize - 1) / blocksize;

        prolongation_kernel<<<numBlocks, blocksize>>>(levels[i].x, levels[i+1].x, levels[i].ni, levels[i].nj, levels[i].nk, levels[i+1].ni, levels[i+1].nj, levels[i+1].nk);

        cudaMemset(temp0, 0, levels[0].number*sizeof(double));
        smoothing_jacobi<double>(levels[i].x, levels[i].b, temp0, levels[i].alpha*scale[i], levels[i].beta, levels[i].ni, levels[i].nj, levels[i].nk, 4);
    }

    int numBlocks = (levels[0].number + blocksize - 1)/blocksize;
    add_kernel<<<numBlocks, blocksize>>>(x, levels[0].x, 1.0, levels[0].number);
    update_residual_kernel<<<numBlocks, blocksize>>>(residual, b, x, levels[0].ni, levels[0].nj, levels[0].nk);
}
// 2-level grid
void V_Cycle(double *b, double *x, double *residual, SCoarseLevelInfo* levels, double *temp0, double *tempResult, int resultOffset)
{
    int blocksize = 256;
    int numBlocks = (levels[0].number + 255)/256;
    int coarsenumBlocks = (levels[1].number + 255)/256;

    double scale = 1.0;

    cudaMemset(levels[0].x, 0, levels[0].number*sizeof(double));

    if (1)
    {
        scale = 8.0;
        cudaMemset(temp0, 0, levels[0].number*sizeof(double));
        smoothing_jacobi<double>(levels[0].x, residual, temp0, levels[0].alpha, levels[0].beta, levels[0].ni, levels[0].nj, levels[0].nk, 32);
    
        cudaMemset(levels[0].r, 0, levels[0].number*sizeof(double));
        update_residual_kernel<<<numBlocks, blocksize>>>(levels[0].r, residual, levels[0].x, levels[0].ni, levels[0].nj, levels[0].nk);
        restriction_kernel<<<coarsenumBlocks, blocksize>>>(levels[0].r, levels[1].b, levels[0].ni, levels[0].nj, levels[0].nk, levels[1].ni, levels[1].nj, levels[1].nk);
    }
    else
    {
        scale = 8.0;

        restriction_kernel<<<coarsenumBlocks, blocksize>>>(residual, levels[1].b, levels[0].ni, levels[0].nj, levels[0].nk, levels[1].ni, levels[1].nj, levels[1].nk);
    }

    cudaMemset(temp0, 0, levels[0].number*sizeof(double));
    cudaMemset(levels[1].x, 0, levels[1].number*sizeof(double));
    smoothing_jacobi<double>(levels[1].x, levels[1].b, temp0, levels[1].alpha*scale, levels[1].beta, levels[1].ni, levels[1].nj, levels[1].nk, 32);

    {
        update_residual_kernel<<<coarsenumBlocks, blocksize>>>(levels[1].r, levels[1].b, levels[1].x, levels[1].ni, levels[1].nj, levels[1].nk);
    
        restriction_kernel<<<(levels[2].number + 255)/256, blocksize>>>(levels[1].r, levels[2].b, levels[1].ni, levels[1].nj, levels[1].nk, levels[2].ni, levels[2].nj, levels[2].nk);
    
        cudaMemset(temp0, 0, levels[0].number*sizeof(double));
        cudaMemset(levels[2].x, 0, levels[2].number*sizeof(double));
        smoothing_jacobi<double>(levels[2].x, levels[2].b, temp0, levels[2].alpha, levels[2].beta, levels[2].ni, levels[2].nj, levels[2].nk, 32);
        
        cudaMemset(temp0, 0, levels[0].number*sizeof(double));
        prolongation_kernel<<<coarsenumBlocks, blocksize>>>(levels[1].x, levels[2].x, levels[1].ni, levels[1].nj, levels[1].nk, levels[2].ni, levels[2].nj, levels[2].nk);
        
        smoothing_jacobi<double>(levels[1].x, levels[1].b, temp0, levels[1].alpha*scale, levels[1].beta, levels[1].ni, levels[1].nj, levels[1].nk, 4);
    }
    
    prolongation_kernel<<<numBlocks, blocksize>>>(levels[0].x, levels[1].x, levels[0].ni, levels[0].nj, levels[0].nk, levels[1].ni, levels[1].nj, levels[1].nk);
    
    cudaMemset(temp0, 0, levels[0].number*sizeof(double));
    smoothing_jacobi<double>(levels[0].x, residual, temp0, levels[0].alpha, levels[0].beta, levels[0].ni, levels[0].nj, levels[0].nk, 4);

    add_kernel<<<numBlocks, blocksize>>>(x, levels[0].x, 1.0, levels[0].number);
    update_residual_kernel<<<numBlocks, blocksize>>>(residual, b, x, levels[0].ni, levels[0].nj, levels[0].nk);
}

extern "C" void gpu_multi_grid_conjugate_gradient(float *u, float *v, float *w , double *div, double *p, double *dir, double *residual, double *temp0, double *temp1, double *tempResult,
            SCoarseLevelInfo* levels, int levelNum, int iter, double halfrdx)
{
    int blocksize = 256;
    int ni = levels[0].ni;
    int nj = levels[0].nj;
    int nk = levels[0].nk;
    int number = levels[0].number;
    int numBlocks = (number + 255)/256;
    divergence_kernel<<<numBlocks, blocksize>>>(u, v, w, div, ni, nj, nk, halfrdx);

    int coutPerThread = (numBlocks + 255)/256;
    cudaMemset(p, 0, number*sizeof(double));
    update_residual_kernel<<<numBlocks, blocksize>>>(residual, div, p, ni, nj, nk);
    mul_kernel<<<numBlocks, blocksize>>>(dir, residual, 1, number);

    {
        calc_max_kernel<<<1, blocksize>>>(residual,  tempResult, number, (number + 255)/256, 2000);
    }

    // r.transpose() * r
    dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, temp0, number);
    calc_sum_kernel<<<1, blocksize>>>(temp0, tempResult, numBlocks, coutPerThread, 0);

    for (size_t i = 0; i < iter; i++)
    {
        int resultOffset = i*2;

        smoothing_conjugate_gradient<double>(div, p, dir, residual, temp0, temp1, tempResult, resultOffset, ni, nj, nk, blocksize, number, numBlocks, coutPerThread);

        //V_Cycle(div, p, residual, levels, temp0, tempResult, i);
        V_Cycle(div, p, residual, levels, temp0, tempResult, levelNum, i);

        {
            calc_max_kernel<<<1, blocksize>>>(residual, tempResult, number, (number + 255)/256, 2001+i);
        }

        updateDir(div, p, residual, dir, temp0, tempResult, resultOffset, blocksize, number, numBlocks, coutPerThread);
    }
    
    number = (ni + 1) * nj * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(u, p, ni + 1, nj, nk, 1, 0, 0, halfrdx);

    number = ni * (nj + 1) * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(v, p, ni, nj + 1, nk, 0, 1, 0, halfrdx);

    number = ni * nj * (nk + 1);
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(w, p, ni, nj, nk + 1, 0, 0, 1, halfrdx);
}
// Multi-Grid Conjugate Gradient solver end

// jacobi iteration for projection
__global__ void jacobi_kernel(float *p, float *div, float *outP, int ni, int nj, int nk, float alpha, float beta)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>0 && i<ni-1&& j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        float p_left = p[k*nj*ni + j*ni + i - 1];
        float p_right = p[k*nj*ni + j*ni + i + 1];
        float p_front = p[k*nj*ni + (j-1)*ni + i];
        float p_back = p[k*nj*ni + (j+1)*ni + i];
        float p_down = p[(k-1)*nj*ni + j*ni + i];
        float p_up = p[(k+1)*nj*ni + j*ni + i];

        outP[index] = (p_left + p_right + p_front + p_back + p_down + p_up + alpha * div[index]) * beta;
    }
    __syncthreads();
}

extern "C" void gpu_projection_jacobi(float *u, float *v, float *w , float *div, float *p, float *p_temp, float *debugParam, int ni, int nj, int nk, int iter, float halfrdx, float alpha, float beta)
{
    int blocksize = 256;
    int number = ni * nj * nk;
    int numBlocks = (number + 255)/256;
    divergence_kernel<<<numBlocks, blocksize>>>(u, v, w, div, ni, nj, nk, halfrdx);

    int coutPerThread = (numBlocks + 255)/256;
    float *residual;
    cudaMalloc(&residual, number * sizeof(float));
    float *dotResidual;
    cudaMalloc(&dotResidual, number * sizeof(float));
    {
        update_residual_kernel<<<numBlocks, blocksize>>>(residual, div, p, ni, nj, nk);
        dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, dotResidual, number);
        calc_sum_kernel<<<1, blocksize>>>(dotResidual, debugParam, numBlocks, coutPerThread, 0);
        calc_max_kernel<<<1, blocksize>>>(residual, debugParam, number, (number + 255)/256, 2000);
    }

    float *p_in = p;
    float *p_out = p_temp;
    for (size_t i = 0; i < iter; i++)
    {
        jacobi_kernel<<<numBlocks, blocksize>>>(p_in, div, p_out, ni, nj, nk, alpha, beta);

        {
            update_residual_kernel<<<numBlocks, blocksize>>>(residual, div, p_out, ni, nj, nk);
            dot_vector_kernel<<<numBlocks, blocksize>>>(residual, residual, dotResidual, number);
            calc_sum_kernel<<<1, blocksize>>>(dotResidual, debugParam, numBlocks, coutPerThread, i+1);
            calc_max_kernel<<<1, blocksize>>>(residual, debugParam, number, (number + 255)/256, 2001+i);
        }

        // swap
        float *temp = p_in;
        p_in = p_out;
        p_out = temp;
    }
    if (p_out == p_temp)
    {
        cudaMemcpy(p, p_temp, number * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    number = (ni + 1) * nj * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(u, p_out, ni + 1, nj, nk, 1, 0, 0, halfrdx);

    number = ni * (nj + 1) * nk;
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(v, p_out, ni, nj + 1, nk, 0, 1, 0, halfrdx);

    number = ni * nj * (nk + 1);
    numBlocks = (number + 255)/256;
    gradient_kernel<<<numBlocks, blocksize>>>(w, p_out, ni, nj, nk + 1, 0, 0, 1, halfrdx);

    cudaFree(residual);
    cudaFree(dotResidual);
}
// jacobi iteration end