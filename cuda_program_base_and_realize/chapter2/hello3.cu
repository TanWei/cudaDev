#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello from gpu\n");
}

int main()
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
	return 0;
}
