#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }
	
void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Allocate GPU memory here
    // Create memory buffers for input and output vectors
    // 
    // Do not create your own device/context/queue. 
    // Use this->gpu->[program, kernel, queue, context]
    // GPU (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/gpu.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6%2Fsrc%2Flayer%2Fcustom%2Fgpu.cc
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in GPUInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/gpu-new-forward.h

    //@@ Copy memory to the GPU here
    // Copy input vectors to memory buffers
}


void GPUInterface::conv_forward_gpu(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    //
    // Do not create your own device/context/queue.
    // Use this->gpu->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel

    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel on the array
}


void GPUInterface::conv_forward_gpu_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Copy the output back to host

    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->gpu->[program, kernel, queue, context]

    //@@ Free the GPU memory here
    // Release OpenCL resources
}
