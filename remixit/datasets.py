"""jvs2mix データセット。

- MixtureOnlyDataset: RemixIT (E2/E3) 用。**mix/ しか参照しない**(正解 s1/s2 に
  アクセスしないことをクラスのレベルで保証する。docs/05_architecture.md)
- SupervisedDataset: 教師あり上限 (E1) 用。mix/s1/s2 を同一区間で切り出す。

いずれも固定長セグメント(ランダムクロップ、短ければゼロパディング)を返す。
乱数は torch のグローバル/ワーカー seed に従う(DataLoader の worker seeding と整合)。
"""
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


def _random_crop_indices(n: int, seg: int) -> tuple[int, int]:
    if n <= seg:
        return 0, n
    off = int(torch.randint(0, n - seg + 1, (1,)).item())
    return off, off + seg


def _load_segment(path: Path, start: int, stop: int, seg: int) -> torch.Tensor:
    wav, _ = sf.read(path, start=start, stop=stop, dtype="float32")
    out = torch.from_numpy(wav)
    if len(out) < seg:
        out = torch.nn.functional.pad(out, (0, seg - len(out)))
    return out


class MixtureOnlyDataset(Dataset):
    """混合音のみを返す(RemixIT 用)。s1/s2 には一切アクセスしない。"""

    def __init__(self, data_dir: str, segment_sec: float, sample_rate: int):
        self.mix_paths = sorted((Path(data_dir) / "mix").glob("*.wav"))
        assert self.mix_paths, f"no wav files under {data_dir}/mix"
        self.seg = int(segment_sec * sample_rate)

    def __len__(self):
        return len(self.mix_paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = self.mix_paths[i]
        n = sf.info(path).frames
        start, stop = _random_crop_indices(n, self.seg)
        return _load_segment(path, start, stop, self.seg)


class SupervisedDataset(Dataset):
    """mix と正解 (s1, s2) を同一区間で返す(E1 用)。"""

    def __init__(self, data_dir: str, segment_sec: float, sample_rate: int):
        self.data_dir = Path(data_dir)
        self.mix_paths = sorted((self.data_dir / "mix").glob("*.wav"))
        assert self.mix_paths, f"no wav files under {data_dir}/mix"
        self.seg = int(segment_sec * sample_rate)

    def __len__(self):
        return len(self.mix_paths)

    def __getitem__(self, i: int):
        name = self.mix_paths[i].name
        n = sf.info(self.mix_paths[i]).frames
        start, stop = _random_crop_indices(n, self.seg)
        mix = _load_segment(self.data_dir / "mix" / name, start, stop, self.seg)
        refs = torch.stack(
            [_load_segment(self.data_dir / s / name, start, stop, self.seg) for s in ("s1", "s2")]
        )
        return mix, refs
