from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension,CUDAExtension
import torch
import math
from torch.utils.cpp_extension import load
import os, time

this_dir = os.path.dirname(__file__)
sources = [os.path.join(this_dir, "custom_extension.cpp"),
           os.path.join(this_dir, "custom_cuda.cu")
           ]
ext = load(name="custom_extension", sources=sources, verbose=True)
print("Built:", ext)

class MatmulGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,A,B,scale:float):
        C,Cpre = ext.forward(A,B, float(scale))
        ctx.save_for_backward(A,B,Cpre)
        ctx.scale = float(scale)
        return C
    
    @staticmethod
    def backward(ctx, grad_out):
        A,B,Cpre = ctx.saved_tensors
        scale = ctx.scale
        grads = ext.backward(A,B,Cpre, grad_out.contiguous(), float(scale))
        grad_A, grad_B = grads[0], grads[1]
        return grad_A, grad_B,None

def matmul_gelu(A,B,scale=1.0):
    return MatmulGELU.apply(A,B,float(scale))

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)
    M, N, K = 16, 32, 20
    A = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, K, device=device, dtype=torch.float32, requires_grad=True)
    scale = 1.0 / math.sqrt(N)

    # Reference using PyTorch op (autograd)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)
    Cpre_ref = A_ref.mm(B_ref)
    C_ref = 0.5 * Cpre_ref * (1 + torch.tanh(0.7978845608 * (Cpre_ref + 0.044715 * Cpre_ref**3)))
    C_ref = scale * C_ref
    loss_ref = C_ref.sum()
    loss_ref.backward()
    gradA_ref = A_ref.grad.clone()
    gradB_ref = B_ref.grad.clone()

    # Extension path
    A_ext = A.detach().clone().requires_grad_(True)
    B_ext = B.detach().clone().requires_grad_(True)
    C_ext = matmul_gelu(A_ext, B_ext, scale)
    loss_ext = C_ext.sum()
    loss_ext.backward()
    gradA_ext = A_ext.grad.clone()
    gradB_ext = B_ext.grad.clone()

    print("Forward max abs diff:", (C_ref - C_ext).abs().max().item())
    print("gradA max abs diff:", (gradA_ref - gradA_ext).abs().max().item())
    print("gradB max abs diff:", (gradB_ref - gradB_ext).abs().max().item())

    # Tolerances: should be small (1e-5 .. 1e-4)
    return (C_ref - C_ext).abs().max().item(), (gradA_ref - gradA_ext).abs().max().item(), (gradB_ref - gradB_ext).abs().max().item()

if __name__ == "__main__":
    c_diff, gA_diff, gB_diff = test()
    print("Done. diffs:", c_diff, gA_diff, gB_diff)