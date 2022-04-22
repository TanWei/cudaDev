#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/**
 * Listing 3.2 一个典型的CUDA程序基本框架
 * 头文件包含
 * 常量定义（或者宏定义）
 * C++自定义函数和CUDA核函数声明
 * int main()
 * {
 *      分配主机与设备内存
 *      初始化主机中的数据
 * 
 *      将某些数据从主机复制到设备
 *      调用核函数在设备中进行计算
 * 
 *      将某些数据从设备复制到主机
 *      释放主机与设备内存
 * }
 */

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double* x, const double* y, 
    const double* z, const int N);
void check(const double* z, const int N);

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
    cudaMalloc((void**)&d_x, M);
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    const int block_zize = 128;
    const int grid_size = N / block_zize;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double* x, const double* y, double* z)
{
    const int n = blockDim * blockIdx.x + threadIdx.x;
    z[n] = x[n + y[n]];
    if (n < 10)
    {
        printf("from block:%d, thread:%d", blockIdx.x, threadIdx.x);
    }
    if (n==10)
    {
        printf("========================");
    }
    if (n > N - 10)
    {
        printf("from block:%d, thread:%d", blockIdx.x, threadIdx.x);
    }
}

void check(const double* z, const int N)
{
    bool has_error = flase;
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