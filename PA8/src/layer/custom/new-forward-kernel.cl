#define TILE_WIDTH 16
#define KERNEL_SZ 7

// default implementation
__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  // `unrolled` is a (B, H_unroll, W_unroll) tensor
  // `unrolled` is a (B, C*K*K*(H-K+1), W-K+1) tensor
#define x_unroll_3d(i2, i1, i0) unrolled[(i2 * H_unroll + i1) * W_unroll + i0]
#define unroll_mask_dim(c,p,q) (c * K*K + p*K + q)
#define unroll_mat_dim(h,w) (h*W_out+w)
// usage: x_unroll_3d(b, unroll_col(c,p,q), unroll_row(h,w))

/*#define x_unroll_3d(i2, i1, i0) unrolled[(i2*C*K*K + i1)*(H-K+1)*(W-K+1)+i0]*/

  //@@ Define your im2col operations here.
  int W_out = (W-K+1);
  int H_out = (H-K+1);
  int H_unroll = K*K*C_in;
  int W_unroll = H_out * W_out;
  int maskRadius = K/2;
  int b = get_global_id(2);
  int cin = get_global_id(0);
  int row_i = get_global_id(1) / W;
  int col_i = get_global_id(1) % W;

  /*for(int b = 0; b < B; b++)*/
  /*for(int cin = 0; cin < C_in; cin++)*/
  /*for(int row_i = 0; row_i < H; row_i++)*/
  /*for(int col_i = 0; col_i < W; col_i++)*/

  for(int p = 0; p < K; p++)
  for(int q = 0; q < K; q++)
  {
    int row_o = row_i - maskRadius + p;
    int col_o = col_i - maskRadius + q;
    _Bool row_o_in_bounds = row_o >= 0 && row_o < H-K+1;
    _Bool col_o_in_bounds = col_o >= 0 && col_o < W-K+1;
    if(row_o_in_bounds && col_o_in_bounds)
    {
      /*int row_u = unroll_mat_dim(row_o, col_o);*/
      int row_u = col_o * H_out + row_o;
      int col_u = unroll_mask_dim(cin, p, q);
      x_unroll_3d(b, col_u, row_u) = x4d(b, cin, row_i+p, col_i+q);
    }
  }

	/* sequential pseudocode
for b in 0..<B:
    for c_in in 0..<C:
        for row_i in 0..<H:
            for col_i in 0..<W:
                for mask_offset_row in 0..<K:
                    for mask_offset_col in 0..<K:
                        # indices in the output of the convolution whose receptive fields include (row_i, col_i)
                        row_o = ??
                        col_o = ??
                        row_o_in_bounds = 0 <= row_o and row_o < H - K + 1
                        col_o_in_bounds = 0 <= col_o and col_o < W - K + 1
                        if row_o_in_bounds and col_o_in_bounds:
                            # indices in x_unroll to write to
                            col_u = ??
                            row_u = ??

                            x_unroll[b, col_u, row_u] = x[b, c_in, row_i, col_i]
     */

#undef x4d
#undef x_unroll_3d
}
