#ifndef SRC_LAYER_GPU_H
#define SRC_LAYER_GPU_H

#define KERNEL_PATH "src/layer/custom/new-forward-kernel.cl"

#include "kernel.h"
#include "device.h"

class GPU
{
    public:
        cl_program program;        // program
        cl_kernel kernel;          // kernel
        cl_command_queue queue;    // command queue
        cl_context context;        // context

        void setup();
        void teardown();
};

#endif