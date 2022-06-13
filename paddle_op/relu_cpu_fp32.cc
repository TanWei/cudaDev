#include "paddle/extension.h"
#include <vector>
#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> ReluCPUForaward(const paddle::Tensor& x) {
    CHECK_INPUT(x);

    auto out = paddle::empty_like(x);

    auto x_numel = x.numel();
    auto* x_data = x.data<float>();
    auto* out_data = out.data<float>();
    for (int64_t i=0; i<x_numel; ++i)
    {
        out_data[i] = std::max(static_cast<float>(0.), x_data[i]);
    }

    return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out){
    CHECK_INPUT(x);
    CHECK_INPUT(out);
    CHECK_INPUT(grad_out);

    auto grad_x = paddle::empty_like(x);

    auto out_numel = out.numel();
    auto* out_data = out.data<float>();
    auto* grad_out_data = grad_out.data<float>();
    auto* grad_x_data = grad_x.data<float>();

    for (int64_t i=0; i<out_numel; i++) {
        grad_x_data[i] = 
            grad_out_data[i] * (out_data[i] > static_cast<float>(0) ? 1. : 0.);

    }
    return {grad_x};
}