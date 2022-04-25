#include <stdio.h>

__global__ void hello_from_gpu()
{
    cont int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("hello block:%d and thread:(%d, %d)\n",b, tx, ty);
}

void main() {
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return;
}