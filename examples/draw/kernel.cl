
__kernel void sawtoothKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int stride = get_global_size(0);
    if (get_local_id(0) >= get_local_id(1)) {
        X[y * stride + x] = 1.0f;
    } else {
        X[y * stride + x] = 0.0f;
    }
}

__kernel void checkerboardKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int stride = get_global_size(0);
    if ((x + y) % 2 == 0) {
        X[y * stride + x] = 1.0f;
    } else {
        X[y * stride + x] = 0.0f;
    }
}

__kernel void diagonalStripeKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int stride = get_global_size(0);
    if (abs(x - y) < 2) {
        X[y * stride + x] = 1.0f;
    } else {
        X[y * stride + x] = 0.0f;
    }
}

__kernel void circularGradientKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float dist = sqrt(dx*dx + dy*dy);
    float maxDist = sqrt(cx*cx + cy*cy);
    float val = 1.0f - (dist / maxDist);
    if(val < 0.0f) val = 0.0f;
    X[y * width + x] = val;
}

__kernel void concentricRingsKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float dist = sqrt(dx*dx + dy*dy);
    float ringWidth = 1.0f;
    int ring = (int)(dist / ringWidth);
    if (ring % 2 == 0)
        X[y * width + x] = 1.0f;
    else
        X[y * width + x] = 0.0f;
}

__kernel void spiralKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float radius = sqrt(dx*dx + dy*dy);
    float angle = atan2(dy, dx);
    float value = sin(10.0f * log(radius + 1.0f) + angle);
    X[y * width + x] = (value > 0.0f) ? 1.0f : 0.0f;
}

__kernel void crossPatternKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int cx = width / 2;
    int cy = height / 2;
    int threshold = 2;
    if (abs(x - cx) < threshold || abs(y - cy) < threshold)
        X[y * width + x] = 1.0f;
    else
        X[y * width + x] = 0.0f;
}

__kernel void horizontalGradientKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    float val = (float)x / (float)(width - 1);
    X[y * width + x] = val;
}

__kernel void diagonalStripesKernel(__global float* X) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    // Create stripes along the main diagonal
    int diff = abs(x - y);
    if (diff % 4 < 2)
        X[y * width + x] = 1.0f;
    else
        X[y * width + x] = 0.0f;
}


__kernel void zigzagKernel(__global float* X) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    // Create a zigzag effect by comparing mod values of x and y
    if ((x % 8) < (y % 8))
        X[y * width + x] = 1.0f;
    else
        X[y * width + x] = 0.0f;
}

__kernel void multiSpiralKernel(__global float* X) {
    const float PI = 3.14159265f;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float r = sqrt(dx*dx + dy*dy);
    float angle = atan2(dy, dx);
    // Combine multiple spiral arms using angle and radius
    if (sin(5.0f * angle + r/3.0f) > 0.0f)
        X[y * width + x] = 1.0f;
    else
        X[y * width + x] = 0.0f;
}