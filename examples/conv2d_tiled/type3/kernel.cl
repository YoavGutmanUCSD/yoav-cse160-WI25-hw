// TILE_WIDTH is provided as a build option (e.g., -DTILE_WIDTH=16)

__kernel void conv_forward_kernel(__global float *y,
                                  __global const float *x,
                                  __constant float *mask,
                                  const int H,
                                  const int W,
                                  const int K,
                                  const unsigned int W_grid,
                                  const unsigned int H_grid)
{
    // Each work-group corresponds to one output tile.
    // We use a 2D NDRange so that:
    //   group id (0) is the tile (column) index and group id (1) is the tile (row) index.
    int tile_col = get_group_id(0); // tile index in X
    int tile_row = get_group_id(1); // tile index in Y
    
    int local_row = get_local_id(0); // within-tile row index
    int local_col = get_local_id(1); // within-tile column index
    
    // The output coordinate computed by this work-item:
    int out_row = tile_row * TILE_WIDTH + local_row;
    int out_col = tile_col * TILE_WIDTH + local_col;
    
    // The output image dimensions for valid convolution.
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
    // Local memory holds only the "core" input corresponding to the output tile.
    __local float tile[TILE_WIDTH][TILE_WIDTH];
    
    // Each work-item loads its corresponding input element.
    // (We assume that the input element for the output pixel is at the same coordinates.)
    if (out_row < H && out_col < W) {
        tile[local_row][local_col] = x[out_row * W + out_col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute the convolution only if the work-item corresponds to a valid output pixel.
    if (out_row < H_out && out_col < W_out) {
        float accum = 0.0f;
        // The starting coordinate for this tile in the input image.
        int tileStartRow = tile_row * TILE_WIDTH;
        int tileStartCol = tile_col * TILE_WIDTH;
        
        // For each position in the KxK convolution window:
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int cur_row = out_row + p;
                int cur_col = out_col + q;
                float val;
                // If the required input element lies within the tile's "core",
                // use the local memory; otherwise, load it from global memory.
                if ((cur_row >= tileStartRow) && (cur_row < tileStartRow + TILE_WIDTH) &&
                    (cur_col >= tileStartCol) && (cur_col < tileStartCol + TILE_WIDTH))
                {
                    val = tile[cur_row - tileStartRow][cur_col - tileStartCol];
                } else {
                    // For halo elements, read directly from global memory.
                    if (cur_row < H && cur_col < W)
                        val = x[cur_row * W + cur_col];
                    else
                        val = 0.0f; // Boundary check (should not occur in valid convolution)
                }
                accum += val * mask[p * K + q];
            }
        }
        // Write the computed convolution result to the output image.
        y[out_row * W_out + out_col] = accum;
    }
}

