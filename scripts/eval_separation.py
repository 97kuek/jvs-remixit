#!/usr/bin/env python
"""2話者分離の評価 (PIT + SI-SNR / SI-SNRi)。

E0(教師ゼロショット)と E1-E3(学習済み生徒)の評価を共用する。
ESPnet の SeparateSpeech で推論し、正解 (s1/s2) と比較する。

使い方(E0 の例):
    conda run -n tflocoformer python scripts/eval_separation.py \
        --train_config <exp_dir>/config.yaml \
        --model_file <exp_dir>/valid.loss.ave_5best.pth \
        --data_dir data/jvs2mix_8k/test \
        --out exp/e0_wsj0_teacher \
        --tag wsj0_2mix_teacher
"""
import argparse
import csv
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def si_snr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    e = est - s
    return float(10 * np.log10((np.dot(s, s) + eps) / (np.dot(e, e) + eps)))


def pit_si_snr(ests, refs):
    """全置換のうち平均 SI-SNR 最大のものを返す。"""
    best = None
    for perm in permutations(range(len(refs))):
        vals = [si_snr(ests[i], refs[j]) for i, j in enumerate(perm)]
        mean = sum(vals) / len(vals)
        if best is None or mean > best[0]:
            best = (mean, vals, perm)
    return best  # (mean, per-source, perm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_config", help="ESPnet config.yaml(教師の評価時)")
    ap.add_argument("--model_file", help="ESPnet チェックポイント .pth(教師の評価時)")
    ap.add_argument("--student_ckpt", help="自前形式チェックポイント(train_*.py の出力、生徒の評価時)")
    ap.add_argument("--data_dir", required=True, help="{mix,s1,s2}/ を含むディレクトリ")
    ap.add_argument("--out", required=True, help="結果の出力ディレクトリ")
    ap.add_argument("--tag", default="model", help="結果ファイルの識別名")
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="評価ファイル数の上限(smoke用)")
    args = ap.parse_args()

    if args.student_ckpt:
        import torch
        from remixit.training import load_student_checkpoint

        model, _ = load_student_checkpoint(args.student_ckpt, args.device)
        model.eval()
        model_id = args.student_ckpt

        def separate(mix_batch, fs):
            with torch.no_grad():
                est = model(torch.from_numpy(mix_batch).to(args.device))
            return [est[:, i].cpu().numpy() for i in range(est.size(1))]

    else:
        assert args.train_config and args.model_file, \
            "--student_ckpt か (--train_config + --model_file) のどちらかを指定"
        import remixit.espnet_compat  # noqa: F401  (tflocoformer をタスクへ実行時登録)
        from espnet2.bin.enh_inference import SeparateSpeech
        model_id = args.model_file
        separate = SeparateSpeech(
            train_config=args.train_config,
            model_file=args.model_file,
            normalize_output_wav=False,
            device=args.device,
        )

    data = Path(args.data_dir)
    ids = sorted(p.stem for p in (data / "mix").glob("*.wav"))
    if args.limit:
        ids = ids[: args.limit]
    assert ids, f"no wav files under {data}/mix"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, mid in enumerate(ids):
        mix, sr = sf.read(data / "mix" / f"{mid}.wav", dtype="float32")
        refs = [sf.read(data / s / f"{mid}.wav", dtype="float32")[0] for s in ("s1", "s2")]
        ests = separate(mix[None, :], fs=sr)
        ests = [e[0] for e in ests]
        assert len(ests) == 2, f"expected 2 sources, got {len(ests)}"

        mean_snr, per_src, _ = pit_si_snr(ests, refs)
        # 入力そのまま(mix)の SI-SNR を引いて改善量に
        mix_snr = sum(si_snr(mix, r) for r in refs) / 2
        rows.append([mid, f"{mean_snr:.2f}", f"{mix_snr:.2f}", f"{mean_snr - mix_snr:.2f}"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)}")

    with open(out / f"results_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "si_snr", "input_si_snr", "si_snr_i"])
        w.writerows(rows)

    arr = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    summary = {
        "tag": args.tag,
        "model_file": model_id,
        "data_dir": str(data),
        "n": len(rows),
        "si_snr_mean": round(float(arr[:, 0].mean()), 2),
        "input_si_snr_mean": round(float(arr[:, 1].mean()), 2),
        "si_snr_i_mean": round(float(arr[:, 2].mean()), 2),
        "si_snr_i_median": round(float(np.median(arr[:, 2])), 2),
        "si_snr_i_std": round(float(arr[:, 2].std()), 2),
    }
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
