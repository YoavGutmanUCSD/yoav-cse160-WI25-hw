
// OpenCL kernel for 2D convolution (valid padding)
__kernel void conv2d(__global const float* image, 
                     __global const float* ker, 
                     __global float* output, 
                     const int H, const int W, const int K) 
{ 
    int outW = W - K + 1; 
    int outH = H - K + 1; 
    int col = get_global_id(0); 
    int row = get_global_id(1); 
    if(col < outW && row < outH) { 
        float sum = 0.0f; 
        for (int i = 0; i < K; i++) { 
            for (int j = 0; j < K; j++) { 
                sum += image[(row + i)*W + (col + j)] * ker[i*K + j]; 
            } 
        } 
        output[row*outW + col] = sum; 
    } 
}

