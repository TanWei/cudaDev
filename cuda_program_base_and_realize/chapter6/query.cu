#include "../error.cuh"
#include <stdio.h>

int main(int argc, char * argv[])
{
    int device_id = 0;
    if(argc > 1)
    {
        device_id = atoi(argv[1]);
    }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_id));
    // 1.block是resident在SM上的，一个SM可能有一个或多个resident blocks，需要具体根据资源占用分析。
    printf("device id:                                  %d\n", device_id);
    printf("device name:                                %s\n", prop.name);
    printf("compute capability:                         %d.%d\n", prop.major, prop.minor);
    printf("amount of global memory:                    %g GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("amount of constant memory:                  %g KB\n", prop.totalConstMem / 1024.0);
    printf("Maximum grid size:                          %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                         %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("num of SMs:                                 %d\n", prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block:  %g KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:     %g kB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:      %d k\n", prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:         %d k\n", prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:        %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:           %d\n", prop.maxThreadsPerMultiProcessor);
    return 0;
}