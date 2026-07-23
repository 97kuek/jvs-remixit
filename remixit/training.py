"""学習の共通部品(E1/E2/E3 共用)。

- warmup + cosine の学習率スケジュール(jchat-sep の教訓: 最初から自動化し手動変更しない)
- チェックポイント保存/読み込み(自前形式: model state + model config パス)
- W&B 初期化(DEC-011)
"""
import math
from pathlib import Path

import torch


def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def save_checkpoint(path: Path, model, model_cfg: str, epoch: int, valid_loss: float, **extra):
    """extra には optimizer / scheduler の state_dict、best、patience などを渡し、
    途中から再開できるようにする(すべて任意)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "model_cfg": model_cfg,
            "epoch": epoch, "valid_loss": valid_loss}
    ckpt.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # 保存中に落ちても既存ファイルを壊さない


def load_student_checkpoint(path: str, device: str = "cpu"):
    from remixit.separator import build_student

    ckpt = torch.load(path, map_location=device)
    model = build_student(ckpt["model_cfg"], device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def init_wandb(run_name: str, config: dict, enabled: bool = True):
    if not enabled:
        return None
    import wandb

    return wandb.init(project="jvs-remixit", name=run_name, config=config)
