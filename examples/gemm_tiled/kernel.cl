// OpenCL kernel source: Transposes a matrix using local memory.
// The macros M and N (matrix dimensions) are defined at build time.

__kernel void matrixTranspose(__global const float *input, __global float *output) {
    const int TILE_DIM =  16 ;
    __local float tile[TILE_DIM][TILE_DIM+1];

    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Load data from global to local memory if within bounds
    if (x < N && y < M) { // M,N are constants (macros) inside the kernel. They are variables in host code, passed as build options.
        tile[ly][lx] = input[y * N + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate transposed coordinates
    x = get_group_id(1) * TILE_DIM + lx;
    y = get_group_id(0) * TILE_DIM + ly;

    // Write transposed data back to global memory if within bounds
    if (x < M && y < N) {
        output[y * M + x] = tile[lx][ly];
    }
}