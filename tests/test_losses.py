import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remixit.losses import pit_neg_si_snr, si_snr


def test_pit_neg_si_snr_perfect_match_is_very_negative():
    torch.manual_seed(0)
    b, t = 4, 8000
    ref = torch.randn(b, 2, t)
    loss = pit_neg_si_snr(ref, ref)
    assert loss.item() < -50  # 完全一致は SI-SNR が非常に大きい(負損失も非常に小さい)


def test_pit_neg_si_snr_two_source_matches_permutation_by_hand():
    torch.manual_seed(0)
    b, t = 3, 8000
    ref = torch.randn(b, 2, t)
    est = ref[:, [1, 0]]  # 意図的に入れ替えた並び
    loss = pit_neg_si_snr(est, ref)
    assert loss.item() < -50  # PIT が入れ替わりを吸収して完全一致を検出できる


def test_pit_neg_si_snr_three_source_perfect_match():
    torch.manual_seed(0)
    b, t = 2, 8000
    ref = torch.randn(b, 3, t)
    perm = [2, 0, 1]
    est = ref[:, perm]
    loss = pit_neg_si_snr(est, ref)
    assert loss.item() < -50


def test_pit_neg_si_snr_three_source_worse_than_perfect():
    torch.manual_seed(0)
    b, t = 2, 8000
    ref = torch.randn(b, 3, t)
    noisy_est = ref + 0.5 * torch.randn_like(ref)
    perfect_loss = pit_neg_si_snr(ref, ref)
    noisy_loss = pit_neg_si_snr(noisy_est, ref)
    assert noisy_loss.item() > perfect_loss.item()
