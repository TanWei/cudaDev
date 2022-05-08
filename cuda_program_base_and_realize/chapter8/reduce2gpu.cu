#include "../real.cuh"
void __global__ reduce_global(real* d_x, real* d_y)
{
    /**
     * @brief 
     * 每个线程块处理blockDim.x个数据。
     * 该核函数仅仅将—个长度为10^8的数组d_x归约到＿个长度为10^8／128的
     * 数组d_y。为了计算整个数组元素的和，我们将数组d_y从设备复制到主机,并在
     * 主机继续对数组d_y归约,得到最终的结果。这样做不是很高效,但我们暂时先这
     * 样做。
     */

    // 4 * 8
    const int tid = threadIdx.x;  // 3
    real* x = d_x + bloxkDIm.x * blockIdx.x; // 8 * 1
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) // offset 4
    {
        if (tid < offset) // 3 < 4
        {
            x[tid] +=  x[tid + offset]; // x[3] = x[3 + 4];
        }
        __syncthreads(); // x[0 1 2 3] + x[4 5 6 7]完成
    }

    if (tid == 0)
    {
        dy[blockIdx.x] = x[0]; // threadblock的结果放在自己blockIdx对应的位置内
    }
}