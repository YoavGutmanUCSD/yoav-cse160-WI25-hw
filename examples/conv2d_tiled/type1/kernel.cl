// The build option -DTILE_WIDTH sets the output tile dimension. R sets K/2
// For a convolution kernel of size K (assumed odd), we have R = K/2.

__kernel void conv_forward_kernel(__global float *y,
                                  __global const float *x,
                                  __constant float *mask,
                                  const int H,
                                  const int W,
                                  const int K)
{
    // Each work-group computes one output tile.
    // Compute the tile (group) indices.
    int tile_x = get_group_id(0);
    int tile_y = get_group_id(1);
    
    // Local work-item indices (within the output tile).
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    // Global coordinates for the output pixel computed by this thread.
    int out_col = tile_x * TILE_WIDTH + local_x;
    int out_row = tile_y * TILE_WIDTH + local_y;
    
    // The output image dimensions for valid convolution.
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // The full input tile required (core plus halo) has dimensions:
    //   TILE_WIDTH + K - 1  =  TILE_WIDTH + 2*R.
    const int localDim = TILE_WIDTH + K - 1;  // = TILE_WIDTH + 2*R

    // The top-left global coordinate of the input tile to load.
    // For valid convolution, the output pixel at (tile_y*TILE_WIDTH, tile_x*TILE_WIDTH)
    // uses the input region starting at that same coordinate.
    int in_row_start = tile_y * TILE_WIDTH;
    int in_col_start = tile_x * TILE_WIDTH;
    
    // Declare a local memory array to hold the input tile (core plus halo).
    __local float localTile[TILE_WIDTH + 2*R][TILE_WIDTH + 2*R];

    // There are TILE_WIDTH*TILE_WIDTH work-items per group.
    int numThreads = TILE_WIDTH * TILE_WIDTH;
    int threadId = local_y * TILE_WIDTH + local_x;
    int totalElements = localDim * localDim;

    // Each thread loads multiple elements from the required input tile into local memory.
    // The elements are loaded in a strided loop.
    for (int idx = threadId; idx < totalElements; idx += numThreads) {
        int i = idx / localDim;
        int j = idx % localDim;
        int global_row = in_row_start + i;
        int global_col = in_col_start + j;
        if (global_row < H && global_col < W) {
            localTile[i][j] = x[global_row * W + global_col];
        } else {
            localTile[i][j] = 0.0f; // Boundary padding.
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread computes its output convolution if within valid output bounds.
    if (out_row < H_out && out_col < W_out) {
        float accum = 0.0f;
        // For this strategy, the convolution window for the output pixel
        // corresponds to a KxK block in localTile starting at (local_y, local_x).
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                accum += localTile[local_y + p][local_x + q] * mask[p * K + q];
            }
        }
        y[out_row * W_out + out_col] = accum;
    }
}

