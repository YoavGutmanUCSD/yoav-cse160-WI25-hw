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

#define WILL_DEBUG 1
#define PRINT if(WILL_DEBUG) printf
	
void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

	// B = batch size, 4th dimension. lots of images stacked on top of each other in one batch
	// M = output features, like channels
	// C = input features
	// H = height of input image. # of rows
	// W = width of input image. # of cols.
	// K = size of mask

	// C H W K cannot be done in parallel. B and M can be.
	// Actually, B H W can be done in parallel. C B M shouldn't be.
	// 
	// 
	// 

    //@@ Allocate OpenCL memory here
    // Create memory buffers for input and output vectors
    // 
    // Do not create your own device/context/queue. 
    // Use this->opencl->[program, kernel, queue, context]
    // OpenCL (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.cc  
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in OpenCLInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl-new-forward.h
	cl_int err;
	*device_x = clCreateBuffer(
			this->opencl->context
			, CL_MEM_READ_ONLY
			, W * H * B * C * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_x (image)\n");
	*device_k = clCreateBuffer(
			this->opencl->context
			, CL_MEM_READ_ONLY
			, K * K * C * M * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_k (mask)\n");
	*device_y = clCreateBuffer(
			this->opencl->context
			, CL_MEM_WRITE_ONLY
			, (W - K+1) * (H - K+1) * B * M * sizeof(int)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_y (output)\n");

    //@@ Copy memory to the OpenCL here
    // Copy input vectors to memory buffers
	clEnqueueWriteBuffer(
			this->opencl->queue
			, *device_x
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, W * H * B * C * sizeof(int)
			, host_x
			, 0
			, NULL
			, NULL
			);
	PRINT("enqueue write to device_x\n");
	clEnqueueWriteBuffer(
			this->opencl->queue
			, *device_k
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, K * K * C * M * sizeof(int)
			, host_k
			, 0
			, NULL
			, NULL
			);
	PRINT("enqueue write to device_k\n");
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
	cl_int err;
    err = clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg 5");
    err |= clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg 6");
    err |= clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg 7");
    err |= clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg 8");
	PRINT("set kernel args\n");

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array
	size_t local_work_size[3] = {1,1,1};
	size_t just_one = 1;
	size_t global_work_size[3] = 
	    { W
		, H 
		, B
	    };

	printf("before\n");
	fflush(stdout);
	err = clEnqueueNDRangeKernel(
			this->opencl->queue
			, this->opencl->kernel
			, 1 // work dimension, it's 3d this time around
			, NULL
			/*, global_work_size*/
			, &just_one
			/*, local_work_size*/
			/*, &just_one*/
			, NULL
			, 0
			, NULL
			, NULL);
    CHECK_ERR(err, "kernel error");
	PRINT("launched kernel\n");
	fflush(stdout);
}


void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    //@@ Copy the output back to host
    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]
	clEnqueueReadBuffer(
			this->opencl->queue
			, device_y
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, (W - K+1) * (H - K+1) * B * M * sizeof(int)
			, host_y
			, 0
			, NULL
			, NULL
			);
	PRINT("read data\n");

	for(int i = 0; i < 100; i++){
		printf("%f ", host_y[i]);
	}
	printf("\n");

    //@@ Free the OpenCL memory here
    // Release OpenCL resources
	clReleaseMemObject(device_k);
	clReleaseMemObject(device_x);
	clReleaseMemObject(device_y);
	/*clReleaseKernel(this->opencl->kernel);*/
	/*clReleaseCommandQueue(this->opencl->queue);*/
	/*clReleaseContext(this->opencl->context);*/
	/*free(kernel_source);*/
}
