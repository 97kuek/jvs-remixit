import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remixit.remix import remix, remix_with_noise, sample_derangement, split_teacher_estimate


def test_derangement_has_no_fixed_points():
    g = torch.Generator().manual_seed(0)
    for b in (2, 3, 4, 8, 16):
        for _ in range(50):
            perm = sample_derangement(b, generator=g)
            assert not (perm == torch.arange(b)).any()
            assert sorted(perm.tolist()) == list(range(b))  # 置換であること


def test_derangement_rejects_batch_of_one():
    with pytest.raises(AssertionError):
        sample_derangement(1)


def test_remix_arithmetic():
    torch.manual_seed(0)
    b, t = 4, 8000
    est = torch.randn(b, 2, t)
    perm = sample_derangement(b, generator=torch.Generator().manual_seed(1))
    remixed, targets = remix(est, perm)

    assert remixed.shape == (b, t)
    assert targets.shape == (b, 2, t)
    # m̃ = ターゲットの和
    assert torch.allclose(remixed, targets.sum(dim=1))
    # slot0 は元のまま、slot1 は置換されている
    assert torch.equal(targets[:, 0], est[:, 0])
    assert torch.equal(targets[:, 1], est[:, 1][perm])
    # 置換により各バッチ要素は「元の教師推定和」と一致しない
    orig_sum = est.sum(dim=1)
    assert not torch.allclose(remixed, orig_sum)


def test_split_teacher_estimate_two_output_computes_residual():
    torch.manual_seed(0)
    b, t = 4, 8000
    speaker_est = torch.randn(b, 2, t)
    noise = torch.randn(b, t)
    mix = speaker_est.sum(dim=1) + noise
    got_speaker, got_noise = split_teacher_estimate(speaker_est, mix)
    assert torch.equal(got_speaker, speaker_est)
    assert torch.allclose(got_noise, noise, atol=1e-5)


def test_split_teacher_estimate_three_output_uses_third_slot_directly():
    b, t = 4, 8000
    t_est = torch.randn(b, 3, t)
    mix = torch.randn(b, t)  # 3出力時は mix を使わない(残差を計算しない)
    speaker_est, noise_est = split_teacher_estimate(t_est, mix)
    assert torch.equal(speaker_est, t_est[:, :2])
    assert torch.equal(noise_est, t_est[:, 2])


def test_remix_with_noise_keeps_noise_unpermuted():
    torch.manual_seed(0)
    b, t = 4, 8000
    speaker_est = torch.randn(b, 2, t)
    noise_est = torch.randn(b, t)
    perm = sample_derangement(b, generator=torch.Generator().manual_seed(1))
    remixed, targets = remix_with_noise(speaker_est, noise_est, perm)

    assert remixed.shape == (b, t)
    assert targets.shape == (b, 3, t)
    assert torch.allclose(remixed, targets.sum(dim=1))
    # slot0=s1(そのまま)、slot1=s2(入れ替え)、slot2=雑音(入れ替えない)
    assert torch.equal(targets[:, 0], speaker_est[:, 0])
    assert torch.equal(targets[:, 1], speaker_est[:, 1][perm])
    assert torch.equal(targets[:, 2], noise_est)  # 雑音は自分の録音のまま
