"""TF-Locoformer 分離モデルのラッパ(教師・生徒共用)。

ESPnet の encoder(STFT) → separator → decoder(iSTFT) を波形 in/out で包む。
- 教師: MERL の exp ディレクトリ(config.yaml + ckpt)から構築
- 生徒: 自前の model config(scripts/conf/model_*.yaml)からスクラッチ構築
"""
from pathlib import Path

import torch
import yaml

import remixit.espnet_compat  # noqa: F401  (tflocoformer をタスクへ実行時登録)
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.separator.tflocoformer_separator import TFLocoformerSeparator
from espnet2.tasks.enh import EnhancementTask


class SeparationModel(torch.nn.Module):
    """波形 [B, T] -> 分離波形 [B, num_spk, T]。"""

    def __init__(self, encoder, separator, decoder):
        super().__init__()
        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        ilens = torch.full((mix.size(0),), mix.size(1), dtype=torch.long, device=mix.device)
        feats, f_lens = self.encoder(mix, ilens)
        feats, _, _ = self.separator(feats, f_lens)
        waves = [self.decoder(f, ilens)[0] for f in feats]
        est = torch.stack(waves, dim=1)
        return est[..., : mix.size(1)]


def load_pretrained(exp_dir: str, device: str = "cpu") -> SeparationModel:
    """ESPnet の exp ディレクトリ(config.yaml + ckpt)から構築。"""
    exp = Path(exp_dir)
    enh_model, _ = EnhancementTask.build_model_from_file(
        exp / "config.yaml", exp / "valid.loss.ave_5best.pth", device
    )
    model = SeparationModel(enh_model.encoder, enh_model.separator, enh_model.decoder)
    return model.to(device)


def build_student(model_cfg_path: str, device: str = "cpu") -> SeparationModel:
    """自前 config から生徒モデルをスクラッチ構築。"""
    cfg = yaml.safe_load(Path(model_cfg_path).read_text())
    stft = cfg["stft"]
    encoder = STFTEncoder(n_fft=stft["n_fft"], hop_length=stft["hop_length"])
    decoder = STFTDecoder(n_fft=stft["n_fft"], hop_length=stft["hop_length"])
    separator = TFLocoformerSeparator(input_dim=-1, **cfg["separator"])
    return SeparationModel(encoder, separator, decoder).to(device)
