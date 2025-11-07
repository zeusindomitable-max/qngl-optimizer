# qngl/optimizer.py
import torch

class QNGLOptimizer:
    def __init__(self, params, lr=0.1, reg=1e-6):
        self.params = list(params)
        self.lr = lr
        self.reg = reg

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_complex():
                g_real = torch.outer(grad.real, grad.real)
                g_imag = torch.outer(grad.imag, grad.imag)
                F = g_real + g_imag
            else:
                F = torch.outer(grad, grad)
            F = F + self.reg * torch.eye(F.shape[0], device=grad.device)
            try:
                invF = torch.linalg.inv(F)
                delta = torch.matmul(invF, grad)
            except:
                delta = grad  # fallback
            p.data -= self.lr * delta

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
