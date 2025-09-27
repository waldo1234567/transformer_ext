#include <torch/extension.h>
#include <ATen/ATen.h>

void gelu_scaled_forward_wrapper(at::Tensor x, float scale);
void gelu_scaled_backward_wrapper(const at::Tensor& Xpre, const at::Tensor& grad_out, at::Tensor&dX, float scale);

std::vector<at::Tensor> matmul_gelu_forward(at::Tensor A, at::Tensor B, float scale){
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported (MxN times NxK)");
    at::Tensor B_cast = B;
    if(B.dtype() != A.dtype()){
        B_cast = B.to(A.dtype());
    }
    at::Tensor B_t = B_cast.transpose(0,1).contiguous();
    //cublass come here
    auto Cpre = at::mm(A, B_cast);
    auto C = Cpre.clone();
    gelu_scaled_forward_wrapper(C, scale);
    return{C, Cpre};
}

std::vector<at::Tensor> matmul_gelu_backward(
    at::Tensor A, at::Tensor B, at::Tensor Cpre, at::Tensor grad_out, float scale
){
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda() && Cpre.device().is_cuda() && grad_out.device().is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have same dtype");
    TORCH_CHECK(A.scalar_type() == Cpre.scalar_type() || A.scalar_type() == grad_out.scalar_type(),
            "Expected A/B and Cpre/grad_out to share dtype (consider casting)");

    const auto target_opts = A.options(); // device + dtype
    at::Tensor Cpre_cast = Cpre;
    at::Tensor grad_out_cast = grad_out;
    if (Cpre.scalar_type() != A.scalar_type()) {
        Cpre_cast = Cpre.to(target_opts);
    }
    if (grad_out.scalar_type() != A.scalar_type()) {
        grad_out_cast = grad_out.to(target_opts);
    }

    auto dC = at::empty_like(Cpre_cast, target_opts);
    gelu_scaled_backward_wrapper(Cpre_cast, grad_out_cast, dC, scale);

    auto gradA = at::mm(dC, B.t());

    auto gradB = at::mm(A.t(), dC);

    return {gradA, gradB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &matmul_gelu_forward, "MatMul + GELU + scale forward (CUDA)");
    m.def("backward", &matmul_gelu_backward, "MatMul + GELU + scale backward (CUDA)");
}

