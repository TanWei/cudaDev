#include "book.h"

#define N 10

__global__ void add(int* pa, int* pb, int* pc)
{
    if (blockIdx.x >= N)
    {
        return;
    }
    // N个block, 1个Thread
    pc[blockIdx.x]  = pa[blockIdx.x] + pb[blockIdx.x];
}

int main(){
    int a[N], b[N], c[N];

    int* dev_a;
    int* dev_b;
    int* dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(int)*N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(int)*N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)*N));
    for (int i=0; i<N; i++)
    {
        a[i] = -i;
        b[i] = a[i] * a[i];
    }
    HANDLE_ERROR(
        cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice)
    );
    HANDLE_ERROR(
        cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice)
    );

    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR(
        cudaDeviceSynchronize() 
    );
    

    HANDLE_ERROR(
        cudaMemcpy(c, dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost)
    );
    for (int i=0; i<N; i++)
    {
        printf("c value : %d \n", c[i]);
    }
    // c value : 0 
    // c value : 0
    // c value : 2 
    // c value : 6 
    // c value : 12 
    // c value : 20 
    // c value : 30 
    // c value : 42 
    // c value : 56 
    // c value : 72 
}
