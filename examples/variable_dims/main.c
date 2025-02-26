#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

// Error-check helper function
static void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

#define TILE_DIM   16 // Tile dimension for local memory

// OpenCL kernel source: Transposes a matrix using local memory.
// The macros M and N (matrix dimensions) are defined at build time.
const char *kernelSource =
"__kernel void matrixTranspose(__global const float *input, __global float *output) {\n"
"    const int TILE_DIM = " "16" ";\n"
"    __local float tile[TILE_DIM][TILE_DIM+1];\n"
"\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int lx = get_local_id(0);\n"
"    int ly = get_local_id(1);\n"
"\n"
"    // Load data from global to local memory if within bounds\n"
"    if (x < N && y < M) {\n" // M,N are constants (macros) inside the kernel. They are variables in host code, passed as build options.
"        tile[ly][lx] = input[y * N + x];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Calculate transposed coordinates\n"
"    x = get_group_id(1) * TILE_DIM + lx;\n"
"    y = get_group_id(0) * TILE_DIM + ly;\n"
"\n"
"    // Write transposed data back to global memory if within bounds\n"
"    if (x < M && y < N) {\n"
"        output[y * M + x] = tile[lx][ly];\n"
"    }\n"
"}\n";

int main(void) {
    
    // M,N are variables here in host code
    int M = 32;
    int N = 32;


    cl_int err;

    // 1. Get the first available platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkErr(err, "clGetPlatformIDs");

    // 2. Get a GPU/CPU device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    checkErr(err, "clGetDeviceIDs");

    // 3. Create an OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");

    // 4. Create and build the program with build options for M and N
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");
    
    char buildOptions[300];
    sprintf(buildOptions, "-DM=%d -DN=%d", M, N);
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL); // M,N are passed as macros during build
    
    if (err != CL_SUCCESS) {
        // Print build log if there is a compilation error
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        checkErr(err, "clBuildProgram");
    }

    // 5. Create the kernel
    cl_kernel kernel = clCreateKernel(program, "matrixTranspose", &err);
    checkErr(err, "clCreateKernel");

    // 6. Allocate host memory and initialize input data
    size_t totalSize = M * N;
    float *h_input = (float *)malloc(sizeof(float) * totalSize);
    float *h_output = (float *)malloc(sizeof(float) * totalSize);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            h_input[i * N + j] = (float)(i * N + j);

    // 7. Create device buffers and transfer input data
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * totalSize, NULL, &err);
    checkErr(err, "clCreateBuffer(d_input)");
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * totalSize, NULL, &err);
    checkErr(err, "clCreateBuffer(d_output)");
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, sizeof(float) * totalSize, h_input, 0, NULL, NULL);
    checkErr(err, "clEnqueueWriteBuffer");

    // 8. Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    checkErr(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    checkErr(err, "clSetKernelArg(1)");

    // 9. Define work-group sizes and enqueue the kernel
    size_t globalSize[2] = {
        ((N + TILE_DIM - 1) / TILE_DIM) * TILE_DIM,
        ((M + TILE_DIM - 1) / TILE_DIM) * TILE_DIM
    };
    size_t localSize[2] = { TILE_DIM, TILE_DIM };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    // 10. Read back the transposed result
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * totalSize, h_output, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // 11. Display the results
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

    // 12. Cleanup resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}

