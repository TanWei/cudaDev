//version 1
double __device__ add_device(const double x, const double y) {
    return x+y;
}

__global__ void add1(const double* x, const double* y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n]  = add_device(x[n], y[n]);
    }
}