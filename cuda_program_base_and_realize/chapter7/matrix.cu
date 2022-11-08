const int TILE_DIM = 32;
const int N = 128;

__global__ void copy(cosnt real *A, real* B, const int N)
{
    const int nx = TILE_DIM * blockIdx.x + threadIdx.x;
    const int ny = TILE_DIM * blockIdx.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}


__global__ void tanspose1(const real* A, real* B, const int N) {
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    cosnt int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}


__global__ void tanspose2(const real* A, real* B, const int N) {
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
        // 读全局内存，非合并访问，编译器会通过缓存的方式缓解影响
        // 有些架构不会调用__ldg()函数，从而不会进行读取全局内存的时，不会缓存
        // 可以改为下面的写法，主动调用：
        // if (nx < N && ny < N) {
        //     B[ny * N + nx] = __ldg(&A[nx * N + ny])
        // }
    }
}

int main()
{
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    real* d_A;
    real* d_B;
    // <<<{4, 4}, {32, 32}>>>
    copy<<<grid_size, block_size>>>(d_A, d_B, N);
}
