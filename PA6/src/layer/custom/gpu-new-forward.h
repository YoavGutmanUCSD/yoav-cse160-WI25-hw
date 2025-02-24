#ifndef SRC_LAYER_GPU_NEW_FORWARD_H
#define SRC_LAYER_GPU_NEW_FORWARD_H

#include "device.h"
#include "gpu.h"

class GPUInterface
{
    public:
    GPU* gpu;

    void conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K);
};

#endif
