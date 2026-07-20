#!/usr/bin/env python
"""E1: 教師あり上限(オラクル)。mix + 正解 (s1, s2) で PIT SI-SNR 学習。

使い方:
    CUDA_VISIBLE_DEVICES=0 conda run -n tflocoformer python scripts/train_supervised.py \
        --config scripts/conf/train_supervised.yaml --out_dir exp/e1_supervised
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remixit.datasets import SupervisedDataset
from remixit.losses import pit_neg_si_snr
from remixit.separator import build_student
from remixit.training import init_wandb, save_checkpoint, warmup_cosine_scheduler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default=None, help="config の data_dir を上書き")
    ap.add_argument("--run_name", default="e1_supervised")
    ap.add_argument("--smoke", action="store_true", help="数ステップだけ実行して終了")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)

    tr_set = SupervisedDataset(Path(cfg["data_dir"]) / "train", cfg["segment_sec"], cfg["sample_rate"])
    cv_set = SupervisedDataset(Path(cfg["data_dir"]) / "valid", cfg["segment_sec"], cfg["sample_rate"])
    tr = DataLoader(tr_set, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], drop_last=True)
    cv = DataLoader(cv_set, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model = build_student(cfg["model_config"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(tr) // cfg["grad_accum"]
    sched = warmup_cosine_scheduler(opt, cfg["warmup_steps"], total_steps, cfg["lr_min_ratio"])
    wb = init_wandb(args.run_name, cfg, enabled=not args.no_wandb and not args.smoke)

    best, patience = float("inf"), 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for step, (mix, refs) in enumerate(tr):
            mix, refs = mix.to(device), refs.to(device)
            loss = pit_neg_si_snr(model(mix), refs) / cfg["grad_accum"]
            loss.backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                opt.zero_grad()
                sched.step()
            if step % 50 == 0:
                lr = sched.get_last_lr()[0]
                print(f"epoch {epoch} step {step}/{len(tr)} loss {loss.item() * cfg['grad_accum']:.2f} lr {lr:.2e}")
                if wb:
                    wb.log({"train/loss": loss.item() * cfg["grad_accum"], "lr": lr})
            if args.smoke and step >= 3:
                print("smoke OK")
                return

        model.eval()
        torch.manual_seed(0)  # valid のクロップを毎エポック同一に
        with torch.no_grad():
            vloss = sum(pit_neg_si_snr(model(m.to(device)), r.to(device)).item() for m, r in cv) / len(cv)
        print(f"epoch {epoch} valid_loss {vloss:.2f}")
        if wb:
            wb.log({"valid/loss": vloss, "epoch": epoch})
        if vloss < best:
            best, patience = vloss, 0
            save_checkpoint(out / "best.pth", model, cfg["model_config"], epoch, vloss)
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best {best:.2f})")
                break
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
