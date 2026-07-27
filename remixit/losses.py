"""PIT + negative SI-SNR 損失(N音源対応)。"""
from itertools import permutations

import torch


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR [dB]。est, ref: [..., T]"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps) * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def pit_neg_si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """N音源 PIT の negative SI-SNR(バッチ平均)。est, ref: [B, N, T](同数の N)。

    N=2(話者のみ)・N=3(話者2+雑音、DEC-016)のどちらでも使う。
    順列総当たり(N=3でも6通り)なのでNが小さい前提で十分軽い。
    """
    assert est.size(1) == ref.size(1), "est と ref の音源数は一致している必要がある"
    n = est.size(1)
    mat = si_snr(est.unsqueeze(2), ref.unsqueeze(1))  # [B, N, N] (est_slot, ref_slot)
    best = None
    for perm in permutations(range(n)):
        idx = torch.tensor(perm, device=est.device)
        val = mat[:, torch.arange(n, device=est.device), idx].mean(dim=1)  # [B]
        best = val if best is None else torch.maximum(best, val)
    return -best.mean()
