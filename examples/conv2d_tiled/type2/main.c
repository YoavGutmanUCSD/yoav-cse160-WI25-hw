#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "../../opencl_utils.h"

#define TILE_WIDTH 16

void conv2d(const float *in, const float *kernel, float *out, int H, int W, int K) {
    int out_H = H - K + 1;
    int out_W = W - K + 1;
    for (int i = 0; i < out_H; ++i) {
        for (int j = 0; j < out_W; ++j) {
            float sum = 0.0f;
            // Convolve the kernel over the image region starting at (i, j)
            for (int m = 0; m < K; ++m) {
                for (int n = 0; n < K; ++n) {
                    sum += in[(i + m) * W + (j + n)] * kernel[m * K + n];
                }
            }
            out[i * out_W + j] = sum;
        }
    }
}

int main (void) {

    int H=18, W=18, K=3;
    int H_OUT = H-K+1;
    int W_OUT = W-K+1;
    srand(500); 

    cl_int err;
    OpenCL ocl;
   
    char buildOptions [300];
    sprintf(buildOptions, "-DTILE_WIDTH=%d -DR=%d", TILE_WIDTH, K/2);
    
    opencl_setup(&ocl, "conv_forward_kernel", "kernel.cl", (char*)buildOptions);
    
    
    // Allocate and initialize host memory
    float *h_x = (float*) malloc(sizeof(float) * H * W);
    float *h_mask = (float*) malloc(sizeof(float) * K * K);
    float *h_y = (float*) malloc(sizeof(float) * H_OUT * W_OUT);
    
    // Fill input image with 1's
    for (int i = 0; i < H * W; i++) {
        h_x[i] = rand()%10;
    }
    // Fill convolution kernel (mask) with 3's
    for (int i = 0; i < K * K; i++) {
        h_mask[i] = rand()%10;
    }
    
    // Create device buffers
    cl_mem d_x = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * H * W, h_x, &err);
    checkErr(err, "clCreateBuffer(d_x)");
    
    cl_mem d_mask = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * K * K, h_mask, &err);
    checkErr(err, "clCreateBuffer(d_mask)");
    
    cl_mem d_y = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
                                sizeof(float) * H_OUT * W_OUT, NULL, &err);
    checkErr(err, "clCreateBuffer(d_y)");
    
    // Set kernel arguments
    err  = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), &d_mask);
    err |= clSetKernelArg(ocl.kernel, 3, sizeof(int), &H);
    err |= clSetKernelArg(ocl.kernel, 4, sizeof(int), &W);
    err |= clSetKernelArg(ocl.kernel, 5, sizeof(int), &K);
    checkErr(err, "clSetKernelArg");
    
    // Determine work-group sizes.
    // For correct tiling, the local work-group size should be:
    //    (TILE_WIDTH + 2*R, TILE_WIDTH + 2*R) where R = K/2.
    const int R = K/2; // equals 1
    size_t localWorkSize[3] = {TILE_WIDTH + 2 * R, TILE_WIDTH + 2 * R, 1}; // {18, 18, 1}
    // We have one output tile covering the full output image (16x16)
    size_t numGroups0 = 1; // group 0 (unused index m)
    size_t numGroups1 = 1; // group 1 (tile index)
    size_t numGroups2 = 1; // group 2 (batch index)
    size_t globalWorkSize[3] = {
        localWorkSize[0] * numGroups0,
        localWorkSize[1] * numGroups1,
        localWorkSize[2] * numGroups2
    }; // {18, 18, 1}
    
    // Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 3, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");
    
    // Read the result back to host memory
    err = clEnqueueReadBuffer(ocl.queue, d_y, CL_TRUE, 0, sizeof(float) * H_OUT * W_OUT,
                              h_y, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    // Print the output image.
    // For our settings, each output pixel should equal the sum over a 3x3 region:
    //    9 * (1 * 3) = 27.
    printf("Output image (%d x %d):\n", H_OUT, W_OUT);
    for (int i = 0; i < H_OUT; i++) {
        for (int j = 0; j < W_OUT; j++) {
            printf("%6.2f ", h_y[i * W_OUT + j]);
        }
        printf("\n");
    }

    // Compare with conv
    float *h_ref = (float*) malloc(sizeof(float) * H_OUT * W_OUT);
    conv2d(h_x, h_mask, h_ref, H, W, K);
    float ferr= 0;
    for (int i = 0; i < H_OUT * W_OUT; i++) {
        ferr += abs(h_ref[i] - h_y[i]);
    }
    printf("Error: %6.2f\n", ferr);

    
    // Clean up resources
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_mask);
    clReleaseMemObject(d_y);
    
    opencl_cleanup(&ocl);
        
    free(h_x);
    free(h_mask);
    free(h_y);
    free(h_ref);
    
    return 0;
}

