import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import remixit.datasets as ds_mod
from remixit.datasets import MixtureOnlyDataset, SupervisedDataset

SR = 8000


@pytest.fixture
def toy_data(tmp_path):
    """3件のミニ jvs2mix(mix = s1 + s2)。1件だけセグメント長より短い。"""
    rng = np.random.default_rng(0)
    lens = [2 * SR, 3 * SR, SR // 2]  # 2s, 3s, 0.5s
    for sub in ("mix", "s1", "s2"):
        (tmp_path / sub).mkdir()
    for i, n in enumerate(lens):
        s1 = 0.1 * rng.standard_normal(n).astype("float32")
        s2 = 0.1 * rng.standard_normal(n).astype("float32")
        for sub, sig in (("mix", s1 + s2), ("s1", s1), ("s2", s2)):
            sf.write(tmp_path / sub / f"{i:06d}.wav", sig, SR, subtype="FLOAT")
    return tmp_path


def test_mixture_only_never_touches_references(toy_data, monkeypatch):
    opened = []
    orig_read, orig_info = sf.read, sf.info

    def spy_read(path, *a, **k):
        opened.append(str(path))
        return orig_read(path, *a, **k)

    def spy_info(path, *a, **k):
        opened.append(str(path))
        return orig_info(path, *a, **k)

    monkeypatch.setattr(ds_mod.sf, "read", spy_read)
    monkeypatch.setattr(ds_mod.sf, "info", spy_info)

    dset = MixtureOnlyDataset(toy_data, segment_sec=1.0, sample_rate=SR)
    for i in range(len(dset)):
        _ = dset[i]
    assert opened, "spy did not capture any file access"
    bad = [p for p in opened if "/s1/" in p or "/s2/" in p]
    assert not bad, f"RemixIT dataset accessed reference files: {bad}"


def test_segment_shape_and_padding(toy_data):
    dset = MixtureOnlyDataset(toy_data, segment_sec=1.0, sample_rate=SR)
    assert len(dset) == 3
    for i in range(3):
        assert dset[i].shape == (SR,)
    # 0.5s のファイルは後半ゼロパディング
    short = dset[2]
    assert torch.all(short[SR // 2:] == 0)


def test_supervised_crops_are_aligned(toy_data):
    torch.manual_seed(0)
    dset = SupervisedDataset(toy_data, segment_sec=1.0, sample_rate=SR)
    for i in range(len(dset)):
        mix, refs = dset[i]
        assert mix.shape == (SR,) and refs.shape == (2, SR)
        # 同一区間の切り出しなら mix = s1 + s2 が保たれる
        assert torch.allclose(mix, refs.sum(dim=0), atol=1e-6)
