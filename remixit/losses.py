"""PIT + negative SI-SNR 損失(2話者)。"""
import torch


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR [dB]。est, ref: [..., T]"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps) * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def pit_neg_si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """2話者 PIT の negative SI-SNR(バッチ平均)。est, ref: [B, 2, T]"""
    assert est.size(1) == 2 and ref.size(1) == 2
    # 各ペアの SI-SNR 行列 [B, est_slot, ref_slot]
    mat = si_snr(est.unsqueeze(2), ref.unsqueeze(1))
    straight = (mat[:, 0, 0] + mat[:, 1, 1]) / 2
    crossed = (mat[:, 0, 1] + mat[:, 1, 0]) / 2
    best = torch.maximum(straight, crossed)
    return -best.mean()
