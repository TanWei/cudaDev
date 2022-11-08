//version 1 : device func with return val
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

// version 2 : device func with pointer
void __device__ add2_device(const double x, const double y, double *z) {
    *z = x + y;
}

void __global__ add2(const double* x, const double* y, double* z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add2_device(x[n], y[n], &z[n]);
    }
}

// version 3 : dev func with ref

void __device__ add3_device(const double x, const double y, double &z) 
{
    z = x + y;
}

void __global__ add3(const double* x, const double* y, double* z, const int N)
{
    const int n = blockDim.x*blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n], y[n], z[n]);
    }
}