#pragma once
#include <stdio.h>

#define CHECK(call)                                                         \
do {                                                                        \
    const cudaError_t error_code = call;                                    \
    if (error_code != cudaSuccess)                                          \
    {                                                                       \
        printf("CUDA Error:\n");                                            \
        printf("    FILE:        %s\n", __FILE__);                          \
        printf("    Line:        %d\n", __LINE__);                          \
        printf("    Error code:  %d\n", error_code);                        \
        printf("    Error text:  %s\n", cudaGetErrorString(error_code));    \
    }                                                                       \
} while(0)                                                                  \