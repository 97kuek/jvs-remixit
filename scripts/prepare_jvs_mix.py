#!/usr/bin/env python
"""JVS から2話者シミュ混合 (jvs2mix) を生成する。

段階1: 16kHz JVS 発話を 8kHz にリサンプルしてキャッシュ(data/jvs_8k/)
段階2: 話者 disjoint な filelist から別話者ペアをサンプルし、
       SNR 一様 [snr_range] dB・min 長 fully-overlapped の混合を生成

config に noise: セクションがある場合(DEC-010)、WHAM! 準拠で雑音を付与する:
  - 雑音ファイルは split ごとに disjoint(WHAM! の tr/cv/tt に対応)
  - 大きい方の話者に対する雑音 SNR を一様 [noise.snr_range] dB でスケール
  - 雑音が短い場合はタイル、長い場合はランダム区間を切り出し

出力: {out_dir}/{split}/{mix,s1,s2[,noise]}/{id}.wav + {out_dir}/{split}/metadata.csv
s1/s2(正解)は評価と教師あり上限 (E1) 用。RemixIT (E2/E3) は mix/ のみ参照する。

使い方:
    python scripts/prepare_jvs_mix.py --config scripts/conf/data_jvs2mix.yaml
    python scripts/prepare_jvs_mix.py --config ... --splits test --num 20   # smoke
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

SPK_RE = re.compile(r"(jvs\d+)")
SRC_SR = 16000

SPLIT_SEED_OFFSET = {"train": 0, "valid": 1, "test": 2}


def build_8k_cache(filelist: Path, cache_dir: Path, sr: int) -> dict:
    """filelist の全 wav を sr へリサンプルして cache_dir に保存。

    Returns: {cached_path(str): num_samples(int)}
    """
    index_path = cache_dir / f"index_{filelist.stem}.json"
    if index_path.exists():
        return json.loads(index_path.read_text())

    resampler = torchaudio.transforms.Resample(SRC_SR, sr)
    index = {}
    files = [line.strip() for line in filelist.read_text().splitlines() if line.strip()]
    for i, src in enumerate(files):
        src = Path(src)
        spk = SPK_RE.search(str(src)).group(1)
        dst = cache_dir / spk / src.name
        if not dst.exists():
            wav, in_sr = sf.read(src, dtype="float32")
            assert in_sr == SRC_SR, f"{src}: expected {SRC_SR}Hz, got {in_sr}"
            wav8k = resampler(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, wav8k, sr, subtype="PCM_16")
            n_samples = len(wav8k)
        else:
            n_samples = sf.info(dst).frames
        index[str(dst)] = n_samples
        if (i + 1) % 2000 == 0:
            print(f"  resampled {i + 1}/{len(files)}")
    index_path.write_text(json.dumps(index))
    return index


def load_noise(path: str, n: int, sr: int, resamplers: dict, rng) -> np.ndarray:
    """雑音を読み、モノラル化・リサンプルし、長さ n に合わせる。"""
    noise, in_sr = sf.read(path, dtype="float32")
    if noise.ndim > 1:
        noise = noise[:, 0]  # WHAM! はステレオ。1ch タスクの慣例に従い ch0 を使用
    if in_sr != sr:
        if in_sr not in resamplers:
            resamplers[in_sr] = torchaudio.transforms.Resample(in_sr, sr)
        noise = resamplers[in_sr](torch.from_numpy(noise).unsqueeze(0)).squeeze(0).numpy()
    if len(noise) < n:  # 短ければタイル
        noise = np.tile(noise, n // len(noise) + 1)
    off = rng.integers(len(noise) - n + 1)
    return noise[off : off + n]


def generate_split(split: str, index: dict, cfg: dict, n_mix: int, seed: int) -> None:
    sr = cfg["sample_rate"]
    min_len = int(cfg["min_len_sec"] * sr)
    lo, hi = cfg["snr_range"]

    noise_cfg = cfg.get("noise")
    if noise_cfg:
        noise_files = sorted(str(p) for p in Path(noise_cfg["dirs"][split]).glob("*.wav"))
        assert noise_files, f"no noise wavs in {noise_cfg['dirs'][split]}"
        n_lo, n_hi = noise_cfg["snr_range"]
        resamplers: dict = {}
        # 雑音専用の独立 rng: 話者ペア・SNR の乱数列をクリーン版と一致させ、
        # 「同一混合の雑音あり/なし」比較を可能にするため(共有 rng だと列がずれる)
        rng_noise = np.random.default_rng(seed + 1000)

    by_spk = defaultdict(list)
    for path, n in index.items():
        if n >= min_len:
            by_spk[SPK_RE.search(path).group(1)].append(path)
    speakers = sorted(by_spk)
    assert len(speakers) >= 2, f"{split}: need >=2 speakers, got {len(speakers)}"
    print(f"[{split}] {len(speakers)} speakers, "
          f"{sum(len(v) for v in by_spk.values())} usable utterances -> {n_mix} mixtures")

    out = Path(cfg["out_dir"]) / split
    subs = ("mix", "s1", "s2", "noise") if noise_cfg else ("mix", "s1", "s2")
    for sub in subs:
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_mix):
        spk1, spk2 = rng.choice(speakers, size=2, replace=False)
        p1 = by_spk[spk1][rng.integers(len(by_spk[spk1]))]
        p2 = by_spk[spk2][rng.integers(len(by_spk[spk2]))]
        s1, _ = sf.read(p1, dtype="float32")
        s2, _ = sf.read(p2, dtype="float32")

        n = min(len(s1), len(s2))  # min 長 fully-overlapped
        s1, s2 = s1[:n], s2[:n]

        # s1/s2 パワー比が snr[dB] になるよう s2 をスケール
        snr = rng.uniform(lo, hi)
        p_s1 = np.mean(s1**2) + 1e-10
        p_s2 = np.mean(s2**2) + 1e-10
        s2 = s2 * np.sqrt(p_s1 / p_s2 / 10 ** (snr / 10))

        if noise_cfg:
            np_path = noise_files[rng_noise.integers(len(noise_files))]
            noise = load_noise(np_path, n, sr, resamplers, rng_noise)
            # WHAM! 準拠: 大きい方の話者パワーに対する雑音 SNR
            noise_snr = rng_noise.uniform(n_lo, n_hi)
            p_louder = max(np.mean(s1**2), np.mean(s2**2)) + 1e-10
            p_noise = np.mean(noise**2) + 1e-10
            noise = noise * np.sqrt(p_louder / p_noise / 10 ** (noise_snr / 10))
        else:
            noise = np.zeros_like(s1)

        mix = s1 + s2 + noise

        # クリップ回避: 全信号を同率スケール(SNR と mix=Σsources を保存)
        peak = max(np.abs(mix).max(), np.abs(s1).max(), np.abs(s2).max(), np.abs(noise).max())
        if peak > 0.9:
            g = 0.9 / peak
            s1, s2, noise, mix = s1 * g, s2 * g, noise * g, mix * g

        mix_id = f"{i:06d}"
        sigs = [("mix", mix), ("s1", s1), ("s2", s2)]
        if noise_cfg:
            sigs.append(("noise", noise))
        for sub, sig in sigs:
            sf.write(out / sub / f"{mix_id}.wav", sig, sr, subtype="PCM_16")
        row = [mix_id, p1, p2, f"{snr:.2f}", n]
        if noise_cfg:
            row += [np_path, f"{noise_snr:.2f}"]
        rows.append(row)

    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["id", "s1_src", "s2_src", "snr_db", "num_samples"]
        if noise_cfg:
            header += ["noise_src", "noise_snr_db"]
        w.writerow(header)
        w.writerows(rows)
    dur_h = sum(int(r[4]) for r in rows) / sr / 3600
    print(f"[{split}] done: {n_mix} mixtures ({dur_h:.1f} h) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--num", type=int, default=None,
                    help="混合数の上書き(smoke テスト用)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = Path(cfg["out_dir"]).parent / "jvs_8k"

    for split in args.splits:
        filelist = Path(cfg["filelists"][split])
        print(f"[{split}] building 8kHz cache from {filelist}")
        index = build_8k_cache(filelist, cache_dir, cfg["sample_rate"])
        n_mix = args.num or cfg["num_mixtures"][split]
        generate_split(split, index, cfg, n_mix, cfg["seed"] + SPLIT_SEED_OFFSET[split])


if __name__ == "__main__":
    main()
