__global__ void add(const double* x, const double* y, double* z, int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double)*N;
    double *x, *y, *z;
    CHECK(
        cudaMalloc((void**)&x, M)
    );
    CHECK(
        cudaMalloc((void**)&y, M)
    );
    CHECK(
        cudaMalloc((void**)&z, M)
    );

    for (int n=0; n<N; ++n)
    {
        x[n] = -i;
        y[n] = x[n] * x[n];
    }

    const int block_size = 128;
    const int grid_size = (N+128-1) /block_size;

    add<<<grid_size, block_size>>>(x, y, z);
    CHECK(
        cudaDeviceSynchronize();
    );
    //check(z, N);
    CHECK(
        cudaFree(d_x)
    );
    CHECK(
        cudaFree(d_y)
    );
    CHECK(
        cudaFree(d_z);
    );
    
    return0;
}

__global__ void add(const double* x, const double* y, double* z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N )
    {
        return;
    }
    z[n] = x[n] + y[n];
    if (n < 10)
    {
        printf("from block:%d, thread:%d\n", blockIdx.x, threadIdx.x);
    }
    if (n==10)
    {
        printf("========================");
    }
    if (n > N - 10)
    {
        printf("from block:%d, thread:%d\n", blockIdx.x, threadIdx.x);
    }
}