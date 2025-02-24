#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

//
// Simple OpenCL error-check helper (optional).
//
static void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // ---------------------------------------------------------
    // 1. Setup: Choose an OpenCL platform and device
    // ---------------------------------------------------------
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(err, "clGetPlatformIDs");

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkErr(err, "clGetPlatformIDs(2)");

    // Pick the first platform
    cl_platform_id platform = platforms[0];
    free(platforms);

    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // Fallback to CPU if no GPU found
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    }
    checkErr(err, "clGetDeviceIDs");

    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    // Again fallback
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    }
    checkErr(err, "clGetDeviceIDs(2)");

    // Just pick the first device
    cl_device_id device = devices[0];
    free(devices);

    // ---------------------------------------------------------
    // 2. Create an OpenCL context and command queue
    // ---------------------------------------------------------
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");

    // ---------------------------------------------------------
    // 3. Create the program from source
    // ---------------------------------------------------------
    // The kernel source can also be read from file "Test.cl".
    // For this self-contained snippet, we embed the kernel code directly:
    const char *kernelSource =

    //    "__kernel void Test(__global float* X)\n"
    //    "{\n"
    //    "   int x = get_global_id(0);\n"
    //    "   int y = get_global_id(1);\n"
    //    "   int stride = get_global_size(0);\n"
    //    "   if (x < y) {\n"
    //    "       X[y * stride + x] = 1.0f;\n"
    //    "   } else {\n"
    //    "       X[y * stride + x] = 0.0f;\n"
    //    "   }\n"
    //    "}\n";
    
    "__kernel void Test(__global float* X)\n"
    "{\n"
    "int x = get_global_id(0);\n"
    "int y = get_global_id(1);\n"
    "int stride = get_global_size(0);\n"
    "if (get_local_id(0) >= get_local_id(1)) {\n"
        "X[y * stride + x] = 1.0f;\n"
    "} else {\n"
        "X[y * stride + x] = 0.0f;\n"
    "}\n"
    "}\n";


    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");

    // ---------------------------------------------------------
    // 4. Build the program
    // ---------------------------------------------------------
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log if there's an error
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // ---------------------------------------------------------
    // 5. Create the kernel
    // ---------------------------------------------------------
    cl_kernel kernel = clCreateKernel(program, "Test", &err);
    checkErr(err, "clCreateKernel");

    // ---------------------------------------------------------
    // 6. Allocate and initialize memory buffers
    // ---------------------------------------------------------
    int image_width = 16;
    int image_height = 16;
    size_t bytes = image_width * image_height * sizeof(float);

    // This will hold your final results on the host
    float *hostData = (float *)malloc(bytes);

    // Create device buffer (equivalent to cudaMalloc)
    cl_mem deviceBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    checkErr(err, "clCreateBuffer");

    // ---------------------------------------------------------
    // 7. Set the kernel arguments
    // ---------------------------------------------------------
    // We have only one argument: the buffer
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceBuffer);
    checkErr(err, "clSetKernelArg(0)");

    // ---------------------------------------------------------
    // 8. Enqueue the kernel
    // ---------------------------------------------------------
    // We'll mimic dim3 gridDim(4,4) and blockDim(4,4) -> globalSize = 16x16
    // localSize = 4x4 (optional, can also let OpenCL decide).
    size_t globalSize[2] = { (size_t)image_width, (size_t)image_height };
    size_t localSize[2]  = { 4, 4 };

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,         // work_dim (2D)
                                 NULL,      // global_work_offset
                                 globalSize,// global_work_size
                                 localSize, // local_work_size (optional)
                                 0,
                                 NULL,
                                 NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Wait for the kernel to complete
    clFinish(queue);

    // ---------------------------------------------------------
    // 9. Read results back to host
    // ---------------------------------------------------------
    err = clEnqueueReadBuffer(queue, deviceBuffer, CL_TRUE, 0, bytes, hostData, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Optional: Print some results to verify
    // (For a 16x16, you could print a small portion or all of it.)
    printf("Result array:\n");
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            printf("%.0f ", hostData[j * image_width + i]);
        }
        printf("\n");
    }

    // ---------------------------------------------------------
    // 10. Clean up
    // ---------------------------------------------------------
    free(hostData);
    clReleaseMemObject(deviceBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

