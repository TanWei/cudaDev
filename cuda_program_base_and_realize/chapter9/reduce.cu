#include "../real.cuh"

const int TILE_DIM = 32;
const int N = 128;

void __global__ reduce(real* d_x, real* d_y,  N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];

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
        // d_y[bid] = s_y[0];
        atomcAdd(&d_y[0], s_y[0]);
    }
}

real reduce(const real* d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;
    
    real h_y[1] = {0};
    real* d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(
        cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice)
    );

    reduce<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}
// https://stackoverflow.com/questions/24942073/dynamic-shared-memory-in-cuda
// 当使用带有CUDA的动态共享内存时,只有一个指针传递给内核,它以字节为单位定义请求/分配区域的开始:

// extern __shared__ char array[];
// 没有办法以不同的方式处理它.但是,这并不妨碍您拥有两个用户大小的数组.这是一个有效的例子:

// $ cat t501.cu
// #include <stdio.h>

// __global__ void my_kernel(unsigned arr1_sz, unsigned arr2_sz){

//   extern __shared__ char array[];

//   double *my_ddata = (double *)array;
//   char *my_cdata = arr1_sz*sizeof(double) + array;

//   for (int i = 0; i < arr1_sz; i++) my_ddata[i] = (double) i*1.1f;
//   for (int i = 0; i < arr2_sz; i++) my_cdata[i] = (char) i;

//   printf("at offset %d, arr1: %lf, arr2: %d\n", 10, my_ddata[10], (int)my_cdata[10]);
// }

// int main(){
//   unsigned double_array_size = 256;
//   unsigned char_array_size = 128;
//   unsigned shared_mem_size = (double_array_size*sizeof(double)) + (char_array_size*sizeof(char));
//   my_kernel<<<1,1, shared_mem_size>>>(256, 128);
//   cudaDeviceSynchronize();
//   return 0;
// }


// $ nvcc -arch=sm_20 -o t501 t501.cu
// $ cuda-memcheck ./t501
// ========= CUDA-MEMCHECK
// at offset 10, arr1: 11.000000, arr2: 10
// ========= ERROR SUMMARY: 0 errors
// $
// 如果你有一个混合数据类型数组的随机排列,你需要手动对齐你的数组起始点(并请求足够的共享内存)或者使用alignment指令(并确保请求足够的共享内存),或者使用结构有助于对齐.