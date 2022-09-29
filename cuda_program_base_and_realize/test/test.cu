namespace cns{
    const int N = 128;
    __global__ copy(float* A, float* B)
    {
        int nx = blockIdx.x * 32 + threadIdx.x;
        int ny = blockIdx.y * 32 + threadIdx.y;
    }
}

namespace rns{
    const int N = 128;
    __global__ reduce(float* A, float* B)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        __shared__ float s_v[128];
        // s_a
    }
}


int main()
{
    // float
    cns::copy<<<{4, 4}, {32, 32}>>>(A, );
} 