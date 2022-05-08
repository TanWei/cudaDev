#include "../real.cuh"
namespace reduce1cpu {
    /**
     * @brief 
     * 单精度的浮点数只用6、7位精确的有效数字。
     * 将sum值加到3000多万以后，再加1.23相加其值就不再相加了
     * 更安全的求和算法：kahan求和算法
     * @param x 
     * @param N 
     * @return real 
     */
    real reduce(const real* x, const int N) {
        real sum = 0.0;
        for (int n=0; n < N; n++)
        {
            sum += x[n];
        }
        return sum;
    }
    
}