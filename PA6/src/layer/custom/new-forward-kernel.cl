#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}



__kernel void conv_forward_kernel(__global float *y, __global float *x, __constant float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define BLOCK_SIZE (TILE_WIDTH + KERNEL_SZ - 1)

	//@@ Insert code to implement convolution here
	// variable definitions
	int maskRadius = K / 2;
	int rowLoc = get_local_id(1); 
	int colLoc = get_local_id(0); 
	int rowIn = get_global_id(1);
	int colIn = get_global_id(0);
	/*int  h = rowIn - maskRadius; // row in output (if aligned with x4d)*/
	/*int  w = colIn - maskRadius; // col in output (if aligned with x4d)*/
	__local float tileMem[TILE_WIDTH][TILE_WIDTH];
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int b = get_global_id(2); // batch

	for(int c = 0; c < C; c++)     // sum over all input feature maps (channels)
	{
		if(rowIn + maskRadius >= H || colIn + maskRadius >= W)
		{
			tileMem[rowLoc][colLoc] = 0.0f;
			/*printf("(rowIn, colIn) => (%d, %d). maskRadius is %d\n", rowIn, colIn, maskRadius);*/
		}
		else 
			tileMem[rowLoc][colLoc] = x4d(b, c, rowIn+maskRadius, colIn+maskRadius);

		barrier(CLK_LOCAL_MEM_FENCE);

		if(rowIn < H_out && rowIn >= 0 && colIn < W_out && colIn >= 0)
			for(int m = 0; m < M; m++)          // for each output feature map
			{
				// local tileMem is [C][H][W]
				// why not parallelize H and W?
				// you can do B too

				// load data if you're in range [not done]
				/*if(rowOut < height && colOut < width)*/
				/*{*/
				/*	// tileMem[c][y][x] = x4d(b, c, h, w)*/
				/**/
				/*}*/
				/*else */
				/*{*/
				/**/
				/*}*/

				// y4d(b, m, h, w) += k4d(m, c, p, q) * x4d(b, c, h+p, w+q) // normally
				// y4d(b,m,h,w) += k4d(m, c, p, q) * tileMem[c][h+p][w+q] // with tile

				{
					/*y4d(b, m, h, w) = 0.0f;*/
					/*y4d(b, m, rowIn, colIn) = 0.0f;*/
					// here
					for(int p = 0; p < K; p++) // KxK filter
						for(int q = 0; q < K; q++)
						{
							if (rowLoc + p < maskRadius || p + rowLoc >= TILE_WIDTH + maskRadius
									|| colLoc + q < maskRadius || colLoc + q >= TILE_WIDTH + maskRadius)
								y4d(b, m, rowIn, colIn)
									+= x4d(b, c, rowIn + p , colIn + q) 
									* k4d(m, c, p, q);
							else
								y4d(b, m, rowIn, colIn)
									+= tileMem[rowLoc + p - maskRadius][colLoc + q - maskRadius] 
									* k4d(m, c, p, q);
						}
				}
			}
		barrier(CLK_LOCAL_MEM_FENCE);
	}


	// lecture naive algorithm
	/*
	for (int b = 0; b < B; b++)             // for each image in batch
		for(int m = 0; m < M; m++)          // for each output feature map
			for(int h = 0; h < H_out; h++)  // for each output element
				for(int w = 0; w < W_out; w++) 
				{
					y[b, m, h, w] = 0.0f;
					for(int c = 0; c < C; c++)     // sum over all input feature maps (channels)
						for(int p = 0; p < K; p++) // KxK filter
							for(int q = 0; q < K; q++)
								y[b, m, h, w] += x[b, c, h + p, w + q] * k[m,c,p,q];
				}
	*/
#undef y4d
#undef x4d
#undef k4d
}
