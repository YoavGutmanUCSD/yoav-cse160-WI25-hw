#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }
	
void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Allocate OpenCL memory here
    // Create memory buffers for input and output vectors
    // 
    // Do not create your own device/context/queue. 
    // Use this->opencl->[program, kernel, queue, context]
    // OpenCL (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6%2Fsrc%2Flayer%2Fcustom%opencl.cc
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in OpenCLInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl-new-forward.h

    //@@ Copy memory to the OpenCL here
    // Copy input vectors to memory buffers
}


void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array
}


void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Copy the output back to host

    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Free the OpenCL memory here
    // Release OpenCL resources
}
