#include <cmath>
#include <iostream>
#include <vector>

#include <clblast.h>

#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define CHECK_ERR(err, msg)                            \
    if (err != CL_SUCCESS)                             \
    {                                                  \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                            \
    }

#define will_debug 1
#define PRINT if(will_debug) printf

void OpenCLInterface::conv_forward_gemm_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, cl_mem *device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //@@ Allocate GPU memory here (don't forget batch sizes!)
	size_t input_size  = B * C * H * W; // indexed in this order
	size_t kernel_size = M * C * K * K;
	size_t unroll_size = B * C * K * K * (H-K+1) * (W-K+1);
	size_t output_size = B * M * (H-K+1) * (W-K+1);
	cl_int err;
	*device_x = clCreateBuffer(
			this->opencl->context
			, CL_MEM_READ_ONLY
			, input_size * sizeof(float)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_x (image)\n");
	*device_k = clCreateBuffer(
			this->opencl->context
			, CL_MEM_READ_ONLY
			, kernel_size * sizeof(float)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_k (mask)\n");

	*device_y = clCreateBuffer(
			this->opencl->context
			, CL_MEM_WRITE_ONLY
			, output_size * sizeof(float)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_y (final output)\n");

	*device_x_unroll = clCreateBuffer(
			this->opencl->context
			, CL_MEM_READ_WRITE
			, unroll_size * sizeof(float)
			, NULL // host_ptr points to already allocated memory. we have none
			, &err
			);
	PRINT("allocated device_x_unroll (intermediate output)\n");

    //@@ Copy memory to the GPU here
	clEnqueueWriteBuffer(
			this->opencl->queue
			, *device_x
			, CL_TRUE // maybe CL_FALSE possible -- try
			, 0
			, input_size * sizeof(float)
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
			, kernel_size * sizeof(float)
			, host_k
			, 0
			, NULL
			, NULL
			);
	PRINT("enqueue write to device_k\n");
}

void OpenCLInterface::conv_forward_gemm_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const cl_mem device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //@@ ====== Start im2col =====

    // @@ define local and global work sizes
	size_t local_work_size[3] = {0,0,0};
	size_t global_work_size[3] = {1,1,1};

	cl_int err;
    err = clSetKernelArg(this->opencl->im2col_kernel,  0, sizeof(cl_mem), &device_x_unroll);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 2, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 3, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 4, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 5, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg 5");
    err |= clSetKernelArg(this->opencl->im2col_kernel, 6, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg 6");

    //@@ Launch the im2col kernel here
	err = clEnqueueNDRangeKernel(
			this->opencl->queue
			, this->opencl->im2col_kernel
			, 3 // work dimension, it's 3d this time around
			, NULL
			, global_work_size
			/*, &just_one*/
			/*, local_work_size*/
			/*, &just_one*/
			, NULL // locwrksize
			, 0
			, NULL
			, NULL);

    //@@ ====== End im2col =====

    //@@ ====== Start gemm =====

    // @@ Call clblast::GemmBatched here
	auto unroll_offsets = std::vector<size_t>();
	auto output_offsets = std::vector<size_t>();
	std::vector<float> alphas(B, 1);
	std::vector<float> betas(B, 0);
	std::vector<size_t> zeroes(B, 0);
	size_t single_unrolled_size = (W-K+1) * (H-K+1) * C * K * K;
	size_t single_output_size = (W-K+1) * (H-K+1);
	for(int i = 0; i < B; i++)
	{
		unroll_offsets.push_back(i * single_unrolled_size);
		output_offsets.push_back(i * single_output_size);
	}
	// Mx(K*K)∙(K*K)x((W-K+1)(H-K+1)) = Mx((W-K+1)(H-K+1)), which is desired.
	// can't make K*K the columns of B.
	// solution: transpose both and do B∙A. then do another transpose. 
	// this would be FUN. but maybe for LATER.

	clblast::GemmBatched<float>(
			clblast::Layout::kRowMajor   // const Layout layout
			, clblast::Transpose::kNo    // const Transpose a_transpose
			, clblast::Transpose::kNo    // const Transpose b_transpose
			, M                          // const size_t m		rows in A. 
			, (W-K+1)*(H-K+1)            // const size_t n		columns in B. 
			, K*K*C                      // const size_t k		rows in B and columns in A.
			, alphas.data()              // const T *alphas
			, device_k                   // const cl_mem a_buffer
			, zeroes.data()              // const size_t *a_offsets
			, K*K*C                      // const size_t a_ld
			, device_x_unroll            // const cl_mem b_buffer
			, unroll_offsets.data()      // const size_t *b_offsets
			, (W-K+1)*(H-K+1)            // const size_t b_ld
			, betas.data()               // const T *betas
			, device_y                   // cl_mem c_buffer
			, output_offsets.data()      // const size_t *c_offsets
			, (W-K+1)*(H-K+1)            // const size_t c_ld
			, B                          // const size_t batch_count
			, &this->opencl->queue       // cl_command_queue* queue
			, NULL                       // cl_event* event
			);
/*GemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,*/
/*                       const size_t m, const size_t n, const size_t k,*/
/*                       const T *alphas,*/
/*                       const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,*/
/*                       const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,*/
/*                       const T *betas,*/
/*                       cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,*/
/*                       const size_t batch_count,*/
/*                       cl_command_queue* queue, cl_event* event)*/

    //@@ ====== End gemm =====
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, cl_mem device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //@@ Copy the output back to host

    //@@ Free the GPU memory here
}
