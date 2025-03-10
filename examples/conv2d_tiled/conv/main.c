#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "../../opencl_utils.h"


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


int main(void)
{

    int H = 32;
    int W = 32;
    int K = 3;
    int OUT_H = H-K+1;
    int OUT_W = W-K+1;
    srand(500);


    cl_int err;
    OpenCL ocl;

    opencl_setup(&ocl, "conv2d", "kernel.cl", "");
    
    // Allocate and initialize host memory
    float *h_image   = (float*) malloc(sizeof(float) * H * W);
    float *h_kernel  = (float*) malloc(sizeof(float) * K * K);
    float *h_output  = (float*) malloc(sizeof(float) * OUT_H * OUT_W);
    
    for (int i = 0; i < H * W; i++) {
        h_image[i] = rand()%10;
    }
    for (int i = 0; i < K * K; i++) {
        h_kernel[i] = rand()%10;
    }
    
    // Create device buffers
    cl_mem d_image = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * H * W, h_image, &err);
    checkErr(err, "clCreateBuffer(d_image)");
    
    cl_mem d_kernel = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * K * K, h_kernel, &err);
    checkErr(err, "clCreateBuffer(d_kernel)");
    
    cl_mem d_output = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * OUT_H * OUT_W, NULL, &err);
    checkErr(err, "clCreateBuffer(d_output)");
    
    // Set the kernel arguments
    err = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_image);
    checkErr(err, "clSetKernelArg(0)");
    err = clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_kernel);
    checkErr(err, "clSetKernelArg(1)");
    err = clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), &d_output);
    checkErr(err, "clSetKernelArg(2)");
    err = clSetKernelArg(ocl.kernel, 3, sizeof(int), &H);
    checkErr(err, "clSetKernelArg(3)");
    err = clSetKernelArg(ocl.kernel, 4, sizeof(int), &W);
    checkErr(err, "clSetKernelArg(4)");
    err = clSetKernelArg(ocl.kernel, 5, sizeof(int), &K);
    checkErr(err, "clSetKernelArg(5)");
    
    // Define the NDRange (output image dimensions)
    size_t global[2] = {OUT_W, OUT_H};
    
    // Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");
    
    // Read the result back to host memory
    err = clEnqueueReadBuffer(ocl.queue, d_output, CL_TRUE, 0,
                              sizeof(float) * OUT_H * OUT_W, h_output, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    // Print the output image
    printf("Output image (size %dx%d):\n", OUT_H, OUT_W);
    for (int i = 0; i < OUT_H; i++) {
        for (int j = 0; j < OUT_W; j++) {
            printf("%6.0f ", h_output[i * OUT_W + j]);
        }
        printf("\n");
    }

    // Compare with conv
    float *h_ref = (float*) malloc(sizeof(float) * OUT_H * OUT_W);
    conv2d(h_image, h_kernel, h_ref, H, W, K);
    float ferr= 0;
    for (int i = 0; i < OUT_H*OUT_W; i++) {
        ferr += abs(h_ref[i] - h_output[i]);
    }
    printf("Error: %6.2f\n", ferr);
    
    // Clean up resources
    clReleaseMemObject(d_image);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    
    opencl_cleanup(&ocl);
    
    free(h_image);
    free(h_kernel);
    free(h_output);
    free(h_ref);
    
    return 0;
}

