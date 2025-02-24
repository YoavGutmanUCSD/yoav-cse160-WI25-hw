#include <stdio.h>
#include <iostream>

#include "gpu.h"

// TODO: HACK
#include "device.c"
#include "kernel.c"

#include "kernel.h"
#include "device.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void GPU::setup()
{
// Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    cl_int err;

    cl_device_id device_id;    // device ID

    // Find platforms and devices
    OclPlatformProp *platforms = nullptr;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the device subject to the OC_DEVICE_TYPE.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, nullptr, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "conv_forward_kernel", &err);
    CHECK_ERR(err, "clCreateKernel");
}

void GPU::teardown()
{
    clReleaseProgram(this->program);
    clReleaseKernel(this->kernel);
    clReleaseCommandQueue(this->queue);
    clReleaseContext(this->context);
}
