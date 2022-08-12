#include "../real.cuh"

const int TILE_DIM = 32;
const int N = 128;

void __global__ reduce(real* d_x, real* d_y,  N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];

    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset  >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        // d_y[bid] = s_y[0];
        atomcAdd(&d_y[0], s_y[0]);
    }
}

real reduce(const real* d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;
    
    real h_y[1] = {0};
    real* d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(s_y, h_y, sizeof(real), cudaMemcpyHostToDevice);

    reduce<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}