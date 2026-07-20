import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remixit.remix import remix, sample_derangement


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
