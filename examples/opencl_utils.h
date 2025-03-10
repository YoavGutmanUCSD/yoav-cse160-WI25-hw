#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// Error-check helper function
static void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

// Function to read kernel file
char* loadKernelSource(const char *filename, size_t *kernelSize)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Seek to end to determine file size
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);

    char *source = (char*)malloc(length + 1);
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source.\n");
        exit(EXIT_FAILURE);
    }

    size_t readLength = fread(source, 1, length, fp);
    source[readLength] = '\0'; // Null-terminate the string
    if (kernelSize)
        *kernelSize = readLength;

    fclose(fp);
    return source;
}

typedef struct {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device; 
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel; 
} OpenCL;

void opencl_setup(OpenCL *ocl, char*kernelName, char*kernelPath, char*buildOptions)
{

    cl_int err;
    // 1. Get the first available platform
    err = clGetPlatformIDs(1, &ocl->platform, NULL);
    checkErr(err, "clGetPlatformIDs");
    
    // 2. Get the first available device (CPU/GPU)
    err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_DEFAULT, 1, &ocl->device, NULL);
    checkErr(err, "clGetDeviceIDs");
    
    // 3. Create a context and command queue
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");
    
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    checkErr(err, "clCreateCommandQueue");
    
    // 4. Create and build the program
    size_t kernelSize;
    const char *kernelSource = loadKernelSource(kernelPath, &kernelSize);
    ocl->program = clCreateProgramWithSource(ocl->context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(ocl->program, 1, &ocl->device, buildOptions, NULL, NULL);
    if (err != CL_SUCCESS) {
        // If there is a build error, print the build log
        size_t log_size;
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // 5. Create the kernel
    ocl->kernel = clCreateKernel(ocl->program, kernelName, &err);
    checkErr(err, "clCreateKernel");
}

void opencl_cleanup(OpenCL *ocl){
    
    // 13. Clean up resources
    clReleaseKernel(ocl->kernel);
    clReleaseProgram(ocl->program);
    clReleaseCommandQueue(ocl->queue);
    clReleaseContext(ocl->context);
    
}

