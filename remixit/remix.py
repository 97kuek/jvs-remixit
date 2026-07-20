"""バッチ内リミキシング — RemixIT の核(2話者分離版)。

教師の推定 (s̃1, s̃2) のうち片方のスロットをバッチ内で置換して再混合する:
    m̃_b = s̃1_b + s̃2_{perm(b)}
生徒は m̃ を入力に、(s̃1, s̃2_perm) を疑似ターゲットとして PIT 損失で学習する。

置換 perm は不動点なし(derangement)でサンプルする。不動点があると
その要素は元の混合の教師推定和 = ほぼ元の混合となり、学習信号が弱まるため。
"""
import torch


def sample_derangement(batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """不動点のないランダム置換を返す。batch_size >= 2 必須(RemixIT の制約)。"""
    assert batch_size >= 2, "RemixIT のバッチ内置換には batch_size >= 2 が必要"
    idx = torch.arange(batch_size)
    while True:
        perm = torch.randperm(batch_size, generator=generator)
        if not (perm == idx).any():
            return perm


def remix(est_sources: torch.Tensor, perm: torch.Tensor):
    """教師推定から擬似混合と疑似ターゲットを作る。

    Args:
        est_sources: 教師の推定 [B, 2, T](slot0 = s̃1, slot1 = s̃2)
        perm: バッチ内置換 [B](sample_derangement の出力)
    Returns:
        remixed_mix: m̃ = s̃1 + s̃2[perm]  [B, T]
        pseudo_targets: (s̃1, s̃2[perm]) を積んだ [B, 2, T]
    """
    assert est_sources.dim() == 3 and est_sources.size(1) == 2
    s1 = est_sources[:, 0]
    s2p = est_sources[:, 1][perm]
    pseudo_targets = torch.stack([s1, s2p], dim=1)
    return s1 + s2p, pseudo_targets
