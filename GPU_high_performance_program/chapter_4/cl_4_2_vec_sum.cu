#inlcude "../common/book.h"

#define N 10

__global__ void add(int* a, int* b, int* c)
{
    if tid = blockIdx.x;
    if (tid<N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void add2(int* a, int* b, int* c)
{
    int tid = threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];

    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR( 
        cudaMalloc( (void**)dev_a, N * sizeof(int) ) 
    );
    HANDLE_ERROR( cudaMalloc( (void**)dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)dev_c, N * sizeof(int) ) );

    for (int i=0; i<N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(
        cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) 
        );
    HANDLE_ERROR(
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) 
        );

    // grid楼  block房间 Thread工作人员 , GRID,BLOCK,THREAD是软件概念,而非硬件的概念。
    // add<<<blocks, threads>>> 
    // https://zhuanlan.zhihu.com/p/353401382
    // https://blog.csdn.net/fengtian12345/article/details/80546799
    // 实际设计中，CUDA将这种对应关系规定为：
    // 1） Grid(不止一个)分配到Device上运行；
    // 2） Block分配到SM上运行；
    // 3） Thread分配到Core上运行。

    //__global__返回值必为void，非void返回值类型没有意义
    //add<<<N, 1>>>(dev_a, dev_b, dev_c); 
    add2<<<1, N>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(
        cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
        );

    for (int i=0; i<N; )
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);


    
    return 0;
}