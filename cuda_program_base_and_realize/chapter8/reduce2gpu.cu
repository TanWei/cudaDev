#include "../real.cuh"

const int TILE_DIM = 32;
const int N = 128;

void __global__ reduce_global(real* d_x, real* d_y)
{
    /**
     * @brief 
     * 每个线程块处理blockDim.x个数据。
     * 该核函数仅仅将—个长度为10^8的数组d_x归约到＿个长度为10^8／128的
     * 数组d_y。为了计算整个数组元素的和，我们将数组d_y从设备复制到主机,并在
     * 主机继续对数组d_y归约,得到最终的结果。这样做不是很高效,但我们暂时先这
     * 样做。
     */

    // 4 * 8
    const int tid = threadIdx.x;  // 3
    real* x = d_x + bloxkDIm.x * blockIdx.x; // 8 * 1
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) // offset 4
    {
        if (tid < offset) // 3 < 4
        {
            x[tid] +=  x[tid + offset]; // x[3] = x[3 + 4];
        }
        __syncthreads(); // x[0 1 2 3] + x[4 5 6 7]完成
    }

    if (tid == 0)
    {
        dy[blockIdx.x] = x[0]; // threadblock的结果放在自己blockIdx对应的位置内
    }
}


void __global__ reduce_shared(real* d_x, real* d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
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
        d_y[bid] = s_y[0];
    }
}

//bank冲突
void __global__ void transpose1(const real* A, const real* B, const int N) // N 128
{
    //__shared__ real S[TILE_DIM][TILE_DIM]; // TILE_DIM值为32
    __shared__ real S[TILE_DIM][TILE_DIM + 1]; //解决上面一行的共享bank冲突
    int bx = blockIdx.x * TILE_DIM;
    int by = blockidx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1]; // 11行 无bank冲突
    }
    __syncthreads();
    int nx2 = bx + threadIdx.y;
    int ny2 = by = threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 +* N + ny2] = S[threadIdx.x][threadIdx.y]; // 19行 y方向bank冲突
    }
}