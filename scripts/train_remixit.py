#!/usr/bin/env python
"""E2/E3: RemixIT 自己学習(2話者分離版)。

各ステップ:
  1. ドメイン内混合音 m のバッチをサンプル(正解 s1/s2 には一切触れない)
  2. 教師推定 (s̃1, s̃2) = f_T(m)  [no_grad]
  3. リミキシング: m̃ = s̃1 + P s̃2(P はバッチ内 derangement)
  4. 生徒学習: PIT-SI-SNR( f_S(m̃), (s̃1, P s̃2) )
  5. sequential の場合、update_every_epochs ごとに教師 ← 生徒のコピー

検証損失は「教師ターゲットとの一致度」であり真の分離品質ではない(発散検知用)。
品質評価は学習後に eval_separation.py(正解あり test)で行う。

使い方:
    CUDA_VISIBLE_DEVICES=0 conda run -n tflocoformer python scripts/train_remixit.py \
        --config scripts/conf/train_remixit.yaml --out_dir exp/e2_remixit_static
"""
import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remixit.datasets import MixtureOnlyDataset
from remixit.losses import pit_neg_si_snr
from remixit.remix import remix, sample_derangement
from remixit.separator import build_student, load_pretrained
from remixit.training import init_wandb, save_checkpoint, warmup_cosine_scheduler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    assert cfg["batch_size"] >= 2, "RemixIT は batch_size >= 2 が必須"
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    run_name = args.run_name or f"remixit_{cfg['teacher_update']}"

    tr_set = MixtureOnlyDataset(Path(cfg["data_dir"]) / "train", cfg["segment_sec"], cfg["sample_rate"])
    cv_set = MixtureOnlyDataset(Path(cfg["data_dir"]) / "valid", cfg["segment_sec"], cfg["sample_rate"])
    tr = DataLoader(tr_set, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], drop_last=True)
    cv = DataLoader(cv_set, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                    drop_last=True)

    teacher = load_pretrained(cfg["teacher_exp_dir"], device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_student(cfg["model_config"], device)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(tr) // cfg["grad_accum"]
    sched = warmup_cosine_scheduler(opt, cfg["warmup_steps"], total_steps, cfg["lr_min_ratio"])
    wb = init_wandb(run_name, cfg, enabled=not args.no_wandb and not args.smoke)

    perm_gen = torch.Generator().manual_seed(cfg["seed"])

    def remixit_loss(mix_batch):
        with torch.no_grad():
            t_est = teacher(mix_batch)
        perm = sample_derangement(mix_batch.size(0), generator=perm_gen)
        remixed, targets = remix(t_est, perm)
        return pit_neg_si_snr(student(remixed), targets)

    best = float("inf")
    patience = 0
    n_updates = 0
    for epoch in range(cfg["epochs"]):
        # E3: sequential 教師更新(RemixIT 論文の 20 エポック毎プロトコル)
        if (cfg["teacher_update"] == "sequential" and epoch > 0
                and epoch % cfg["update_every_epochs"] == 0):
            teacher = copy.deepcopy(student).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            n_updates += 1
            print(f"epoch {epoch}: teacher <- student (update #{n_updates})")
            # 教師が変わると検証損失の基準も変わるため、早期終了のカウンタをリセットする
            best, patience = float("inf"), 0

        student.train()
        for step, mix in enumerate(tr):
            loss = remixit_loss(mix.to(device)) / cfg["grad_accum"]
            loss.backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["grad_clip"])
                opt.step()
                opt.zero_grad()
                sched.step()
            if step % 50 == 0:
                lr = sched.get_last_lr()[0]
                print(f"epoch {epoch} step {step}/{len(tr)} loss {loss.item() * cfg['grad_accum']:.2f} lr {lr:.2e}")
                if wb:
                    wb.log({"train/loss": loss.item() * cfg["grad_accum"], "lr": lr,
                            "teacher_updates": n_updates})
            if args.smoke and step >= 3:
                print("smoke OK")
                return

        student.eval()
        torch.manual_seed(0)  # 検証のクロップを毎エポック同一に
        cv_perm_gen = torch.Generator().manual_seed(0)  # 検証の置換も固定
        with torch.no_grad():
            vloss = 0.0
            for mix in cv:
                mix = mix.to(device)
                t_est = teacher(mix)
                perm = sample_derangement(mix.size(0), generator=cv_perm_gen)
                remixed, targets = remix(t_est, perm)
                vloss += pit_neg_si_snr(student(remixed), targets).item()
            vloss /= len(cv)
        print(f"epoch {epoch} valid_loss(vs teacher) {vloss:.2f}")
        if wb:
            wb.log({"valid/loss_vs_teacher": vloss, "epoch": epoch})
        # 教師更新で検証損失の基準が変わるため、best/patience は直近の教師更新以降でのみ意味を持つ
        # (教師更新時にリセット済み)。最新エポックも常に保存し、品質判断は test 評価で行う。
        save_checkpoint(out / "last.pth", student, cfg["model_config"], epoch, vloss)
        if vloss < best:
            best, patience = vloss, 0
            save_checkpoint(out / "best.pth", student, cfg["model_config"], epoch, vloss)
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best {best:.2f} since last teacher update)")
                break
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
