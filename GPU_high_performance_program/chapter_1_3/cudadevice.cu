#include "../common/book.h"

int main (void)
{
    cudaDeviceProp prop;
    int cout;
    HANDLE_ERROR( cudaGetDeviceCount(&count));
    for (int i=0; i<count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        
    }

    cudaDeviceProp prop_my;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop_my.major = 1; // 设备计算功能集的主版本号
    prop.minor = 3; //设备计算功能集的次版本号
    HANDLE_ERROR( cudaChooseDevice(&dev, &prop) );
    prinf("fu he de gpu id is: %d", dev);
    HANDLE_ERROR( cudaSetDevice(dev) );
}