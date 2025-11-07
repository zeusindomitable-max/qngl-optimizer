# qngl/metric.py
import torch

def fubini_study_metric(alpha, psi_funcs, x_data):
    Psi = sum(a * psi(x) for a, psi in zip(alpha, psi_funcs))
    norm = torch.abs(Psi).mean()
    Psi = Psi / torch.sqrt(norm)

    g = torch.zeros(len(alpha), len(alpha), dtype=torch.complex64)
    for i in range(len(alpha)):
        for j in range(i, len(alpha)):
            ip = torch.vdot(psi_funcs[i](x_data), psi_funcs[j](x_data))
            proj = torch.vdot(psi_funcs[i](x_data), Psi) * torch.vdot(Psi, psi_funcs[j](x_data))
            g[i,j] = ip - proj
            g[j,i] = g[i,j].conj()
    return torch.real(g)
