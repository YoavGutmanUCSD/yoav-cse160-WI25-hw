#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "../opencl_utils.h"

#define TILE_DIM   16 // Tile dimension for local memory

int main(void) {
    
    // M,N are variables here in host code
    int M = 32;
    int N = 32;


    cl_int err;
    OpenCL ocl;
    char buildOptions[300];
    sprintf(buildOptions, "-DM=%d -DN=%d", M, N);
    opencl_setup(&ocl, "matrixTranspose", "kernel.cl", buildOptions);

    // Allocate host memory and initialize input data
    size_t totalSize = M * N;
    float *h_input = (float *)malloc(sizeof(float) * totalSize);
    float *h_output = (float *)malloc(sizeof(float) * totalSize);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            h_input[i * N + j] = (float)(i * N + j);

    // Create device buffers and transfer input data
    cl_mem d_input = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * totalSize, NULL, &err);
    checkErr(err, "clCreateBuffer(d_input)");
    cl_mem d_output = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float) * totalSize, NULL, &err);
    checkErr(err, "clCreateBuffer(d_output)");
    err = clEnqueueWriteBuffer(ocl.queue, d_input, CL_TRUE, 0, sizeof(float) * totalSize, h_input, 0, NULL, NULL);
    checkErr(err, "clEnqueueWriteBuffer");

    // Set kernel arguments
    err = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_input);
    checkErr(err, "clSetKernelArg(0)");
    err = clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_output);
    checkErr(err, "clSetKernelArg(1)");

    // Define work-group sizes and enqueue the kernel
    size_t globalSize[2] = {
        ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM,
        ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM
    };
    size_t localSize[2] = { TILE_DIM, TILE_DIM };
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Read back the transposed result
    err = clEnqueueReadBuffer(ocl.queue, d_output, CL_TRUE, 0, sizeof(float) * totalSize, h_output, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Display the results
    printf("Input Matrix (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5d ", (int)h_input[i * N + j]);
        }
        printf("\n");
    }
    printf("\nTransposed Matrix (%d x %d):\n", N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%5d ", (int)h_output[i * M + j]);
        }
        printf("\n");
    }

    // Cleanup resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    opencl_cleanup(&ocl);
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}

