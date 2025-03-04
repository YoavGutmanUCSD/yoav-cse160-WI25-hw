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
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;
	int maskRadius = K / 2;
	int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

	int hTileLoc = get_local_id(1);
	int wTileLoc = get_local_id(0);
	int hTileInd = get_group_id(1) / W_grid;
	int wTileInd = get_group_id(1) % W_grid;

	int h = hTileInd*TILE_WIDTH + hTileLoc;
	int w = wTileInd*TILE_WIDTH + wTileLoc;
	int b = get_global_id(2); // batch
	int m = get_group_id(0); // output channel

	__local float tileMem[TILE_WIDTH][TILE_WIDTH];

	float accum = 0.0f;

	for(int c = 0; c < C; c++)     // sum over all input feature maps (channels)
	{
		// load tiles
		if(h+maskRadius < H && w+maskRadius < W)
			tileMem[hTileLoc][wTileLoc] = x4d(b,c,h+maskRadius,w+maskRadius);
		barrier(CLK_LOCAL_MEM_FENCE);

		// compute
		if(h < H_out && w < W_out)
		{
			for(int p = 0; p < K; p++)
				for(int q = 0; q < K; q++)
				{
					if(hTileLoc+p < TILE_WIDTH && wTileLoc+q < TILE_WIDTH
							&& hTileLoc+p >= maskRadius && wTileLoc+q >= maskRadius)
						accum
							+= tileMem[hTileLoc+p-maskRadius][wTileLoc+q-maskRadius] 
							* k4d(m,c,p,q);
					else
						accum
							+= x4d(b, c, h+p, w+q)
							* k4d(m,c,p,q);
				}
			y4d(b, m, h, w) = accum;
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
