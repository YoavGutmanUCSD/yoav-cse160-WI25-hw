#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "../opencl_utils.h"

#define NUM_KERNELS 11

int main()
{
    cl_int err;
    OpenCL ocl;

    char kernelNames[NUM_KERNELS][256] = { 
        "sawtoothKernel",
        "checkerboardKernel",
        "diagonalStripeKernel",
        "circularGradientKernel",
        "concentricRingsKernel",
        "spiralKernel",
        "crossPatternKernel",
        "horizontalGradientKernel",
        "diagonalStripesKernel",
        "zigzagKernel",
        "multiSpiralKernel"
    };

    for (int i=0; i < NUM_KERNELS; i++) {
        
        opencl_setup(&ocl, kernelNames[i], "kernel.cl", NULL);
    
        int image_width = 16;
        int image_height = 16;
        size_t bytes = image_width * image_height * sizeof(float);
    
        float *hostData = (float *)malloc(bytes);
    
        cl_mem deviceBuffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
        checkErr(err, "clCreateBuffer");
    
        err = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &deviceBuffer);
        checkErr(err, "clSetKernelArg(0)");
    
        size_t globalSize[2] = { (size_t)image_width, (size_t)image_height };
        size_t localSize[2]  = { 4, 4 };
    
        err = clEnqueueNDRangeKernel(ocl.queue,
                                     ocl.kernel,
                                     2,         // work_dim (2D)
                                     NULL,      // global_work_offset
                                     globalSize,// global_work_size
                                     localSize, // local_work_size (optional)
                                     0,
                                     NULL,
                                     NULL);
        checkErr(err, "clEnqueueNDRangeKernel");
    
        clFinish(ocl.queue);
    
        err = clEnqueueReadBuffer(ocl.queue, deviceBuffer, CL_TRUE, 0, bytes, hostData, 0, NULL, NULL);
        checkErr(err, "clEnqueueReadBuffer");
        
        printf("Kernel: %s\n", kernelNames[i]);
        printf("Result:\n");
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                printf("%.0f ", hostData[j * image_width + i]);
            }
            printf("\n");
        }
    
        free(hostData);
        clReleaseMemObject(deviceBuffer);
        opencl_cleanup(&ocl);
    }

    return 0;
}

