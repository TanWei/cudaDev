# cudaDev

Grid Block Thread

硬件限制Block Thread最大值为65535，
且THread <  maxThreadsPerBlock

二维索引空间转换为线性空间的标准方法：
blockDim：每个Block里的线程数量
int tid = threadIdx.x + blockIdx.x * blockDim.x

