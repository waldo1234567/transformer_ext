#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>


inline int GET_BLOCKS(const int n, const int block) {
    return (n + block - 1) / block;
}
 
#define kAlpha 0.7978845608f   // sqrt(2/pi)
#define kBeta 0.044715f
//forward
template<typename scalar_t>
__global__ void gelu_scaled_kernel(scalar_t* __restrict__ X, const int64_t size, const float scale){
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if(i >= size) return;
    float x = static_cast<float>(X[i]);
    float t = kAlpha * (x + kBeta * x * x * x);
    float gelu = 0.5f * x *(1.0f + tanhf(t));
    X[i] = static_cast<scalar_t>(scale * gelu);
}

template<typename scalar_t>
__global__ void gelu_scaled_backward_kernel(
    const scalar_t* __restrict__ Xpre,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ dX,
    const int64_t size,
    const float scale

){
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = static_cast<float>(Xpre[i]);
    float t = kAlpha * (x + kBeta * x * x * x);
    float tanh_t =  tanhf(t);
    float left = 0.5f * (1.0f + tanh_t);
    const float dt_dx = kAlpha * (1.0f + 3.0f * kBeta * x * x);
    const float right = 0.5f * x * (1.0f - tanh_t * tanh_t) * dt_dx;
    const float gelu_deriv = left + right;
    const float gout = static_cast<float>(grad_out[i]);
    dX[i] = static_cast<scalar_t>(gout * scale * gelu_deriv);
}

void gelu_scaled_forward_wrapper(at::Tensor x, float scale){
    TORCH_CHECK(x.device().is_cuda(), "X must be CUDA tensor");
    const int threads = 256;
    const int64_t size = x.numel();    
    const int blocks = GET_BLOCKS(size, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gelu_scaled_kernel", ([&]{
        using scalar_t = scalar_t;
        gelu_scaled_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), size, scale
        );
    }));
    C10_CUDA_CHECK(cudaGetLastError());
}


void gelu_scaled_backward_wrapper(
    const at::Tensor& Xpre, const at::Tensor& grad_out,
    at::Tensor& dX, float scale
){
    TORCH_CHECK(Xpre.device().is_cuda(), "Xpre must be a cuda Tensor")
    TORCH_CHECK(grad_out.device().is_cuda(), "Xpre must be a cuda Tensor")
    const int threads = 256;
    const int64_t size = Xpre.numel();
    const int blocks = GET_BLOCKS(size, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Xpre.scalar_type(), "gelu_scaled_kernel_backward", ([&]{
        gelu_scaled_backward_kernel<scalar_t><<<blocks, threads, 0 , stream>>>(
            Xpre.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            dX.data_ptr<scalar_t>(),
            size,
            scale
        );
    }));

    C10_CUDA_CHECK(cudaGetLastError());
}
