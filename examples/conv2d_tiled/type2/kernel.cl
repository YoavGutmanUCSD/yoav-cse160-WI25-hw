// Kernel source for tiled convolution (conv_forward_kernel)
// Note: The kernel expects three extra group dimensions:
//   - group 0: output feature map index (unused here)
//   - group 1: tile/block index (used to compute output tile offset)
//   - group 2: batch index (unused here)
// TILE_WIDTH is the tile size, and R is the mask radius, given as build options

__kernel void conv_forward_kernel(__global float *y, 
                                  __global const float *x, 
                                  __constant float *mask, 
                                  const int H, 
                                  const int W, 
                                  const int K) 
{
    const int H_out = H - K + 1; 
    const int W_out = W - K + 1; 
    __local float input_s[TILE_WIDTH + 2 * R][TILE_WIDTH + 2 * R]; 
 
    int tx = get_local_id(0); 
    int ty = get_local_id(1);  
    int block_y = get_group_id(1); 
 
    // Compute W_grid as the number of tiles along the output width.
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    
    int col = (block_y % W_grid) * TILE_WIDTH + ty; 
    int row = (block_y / W_grid) * TILE_WIDTH + tx; 
 
    #define    y2d(i1,i0) y   [(i1) * (W_out) + (i0)] 
    #define    x2d(i1,i0) x   [(i1) * (W)     + (i0)] 
    #define MASK2d(i1,i0) mask[(i1) * (K)     + (i0)] 
 
    // Load the input tile (and its halo) into local memory 
    if (col < W && row < H) { 
        input_s[tx][ty] = x2d(row, col); 
    } 
    barrier(CLK_LOCAL_MEM_FENCE); 
 
    // Only threads with indices less than TILE_WIDTH produce output 
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && row < H_out && col < W_out) { 
        float accum = 0.0f; 
        for (int p = 0; p < K; p++) { 
            for (int q = 0; q < K; q++) { 
                accum += input_s[tx + p][ty + q] * MASK2d(p, q); 
            } 
        } 
        y2d(row, col) = accum; 
    } 
    
    #undef y2d 
    #undef x2d 
    #undef MASK2d 
}

