#include "error.cuh"
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double* x, const double* y, double* z, int N);

void check(const double* z, const int N);

//#define CHECK(call) call

int main()
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double* h_x = (double*) malloc(M);
    double* h_y = (double*) malloc(M);
    double* h_z = (double*) malloc(M);
    for (int n=0; n < N; n++)
    {
        h_x[n] = a;
        h_y[n] = b;
    }
    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void**)&d_x, M));
    CHECK(cudaMalloc((void**)&d_y, M));
    CHECK(cudaMalloc((void**)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
    const int block_size = 128;
    const int grid_size = N / block_size;
    printf("block_size:%d, grid_size:%d\n", block_size, grid_size);

    //add<<<grid_size, block_size>>>(d_x, d_y, d_z, N); 
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CHECK(cudaGetLastError()); //捕捉下面一句话之前最后一个错误
    CHECK(cudaDeviceSynchronize()); //主机和设备同步

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
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

void check(const double* z, const int N)
{
    bool has_error = false;
    for (int n=0; n<N; n++)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
            break;
        }
    }
    printf("%s", has_error ? "has errot" : "no error");
}
