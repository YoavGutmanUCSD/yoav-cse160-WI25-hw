#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel_general.cl"
#define DEBUG_MODE 1
#if DEBUG_MODE == 0
#define PRINT printf
#else
#define PRINT if(0)
#endif

void OpenCLMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

	unsigned int tileSize = 8;

	// Find platforms and devices
	OclPlatformProp *platforms = NULL;
	cl_uint num_platforms;

	err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
	CHECK_ERR(err, "OclFindPlatforms");

	// Get the ID for the specified kind of device type.
	err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
	CHECK_ERR(err, "OclGetDeviceWithFallback");

	// Create a context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	CHECK_ERR(err, "clCreateContext");

	// Create a command queue
	queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context
			, 1
			, (const char **)&kernel_source
			, NULL
			, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
	device_a = clCreateBuffer(
			context
			, CL_MEM_READ_ONLY
			, input0->shape[0] * input0->shape[1] * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_a\n");
	device_b = clCreateBuffer(
			context
			, CL_MEM_READ_ONLY
			, input1->shape[0] * input1->shape[1] * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_b\n");
	device_c = clCreateBuffer(
			context
			, CL_MEM_WRITE_ONLY
			, result->shape[0] * result->shape[1] * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_c\n");

    //@@ Copy memory to the GPU here
	clEnqueueWriteBuffer(
			queue
			, device_a
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, input0->shape[0] * input0->shape[1] * sizeof(int)
			, input0->data
			, 0
			, NULL
			, NULL
			);
	PRINT("enqueue write to device_a\n");
	clEnqueueWriteBuffer(
			queue
			, device_b
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, input1->shape[0] * input1->shape[1] * sizeof(int)
			, input1->data
			, 0
			, NULL
			, NULL
			);
	PRINT("enqueue write to device_b\n");

    // Set the arguments to our compute kernel
    // __global const int *A, __global const int *B, __global int *C,
    // const unsigned int numARows, const unsigned int numAColumns,
    // const unsigned int numBRows, const unsigned int numBColumns,
    // const unsigned int numCRows, const unsigned int numCColumns
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &input1->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 5");
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &input1->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 6");
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &result->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 7");
    err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &result->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 8");
    /*err |= clSetKernelArg(kernel, 9, sizeof(unsigned int), &tileSize);*/
    /*CHECK_ERR(err, "clSetKernelArg 9");*/

    // @@ define local and global work sizes
	size_t local_work_size[2] = {tileSize,tileSize};
	// A^{LxN}^T * B^{LxM} = C^{NxM}
	size_t global_work_size[2] = {
		((input0->shape[0]+(tileSize-1))/tileSize)*tileSize,
		((input1->shape[1]+(tileSize-1))/tileSize)*tileSize
	};
	PRINT("set sizes\n");

    //@@ Launch the GPU Kernel here
	clEnqueueNDRangeKernel(
			queue
			, kernel
			, 2 // work dimension, i think 2d so 2
			, NULL
			, global_work_size
			, local_work_size
			, 0
			, NULL
			, NULL);
	PRINT("launched kernel\n");

    //@@ Copy the GPU memory back to the CPU here
	clEnqueueReadBuffer(
			queue
			, device_c
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, result->shape[0] * result->shape[1] * sizeof(int)
			, result->data
			, 0
			, NULL
			, NULL
			);
	PRINT("read data\n");

    //@@ Free the GPU memory here
	clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(kernel_source);
	PRINT("release memory\n");
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix
	rows = host_a.shape[0];
	cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    OpenCLMatrixMultiply(&host_a, &host_b, &host_c);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
