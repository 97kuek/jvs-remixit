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
    """教師推定から擬似混合と疑似ターゲットを作る(話者2つのみ、雑音を明示的に扱わない版)。

    Args:
        est_sources: 教師の推定 [B, 2, T](slot0 = s̃1, slot1 = s̃2)
        perm: バッチ内置換 [B](sample_derangement の出力)
    Returns:
        remixed_mix: m̃ = s̃1 + s̃2[perm]  [B, T]
        pseudo_targets: (s̃1, s̃2[perm]) を積んだ [B, 2, T]

    注意(DEC-016): 環境雑音のある混合では、この関数だと s̃2 を別録音のものに
    入れ替えるとき「元の録音の雑音」と「入れ替え元の録音の雑音」が両方混ざり、
    実際にはあり得ない二重雑音の疑似混合になってしまう。雑音のある設定では
    remix_with_noise() を使うこと。
    """
    assert est_sources.dim() == 3 and est_sources.size(1) == 2
    s1 = est_sources[:, 0]
    s2p = est_sources[:, 1][perm]
    pseudo_targets = torch.stack([s1, s2p], dim=1)
    return s1 + s2p, pseudo_targets


def split_teacher_estimate(t_est: torch.Tensor, mix: torch.Tensor):
    """教師の推定から (話者2つ, 雑音1つ) を取り出す(DEC-016)。

    教師が話者2出力のみ(MERL公開モデル、または更新前)の場合は、
    雑音を残差 ñ = mix − Σ話者推定 として計算する。
    教師が既に3出力(E3で生徒からコピーされた後)の場合は、
    3つ目のスロットをそのまま雑音推定として使う(残差の再計算はしない)。

    Args:
        t_est: 教師の推定 [B, 2, T] または [B, 3, T]
        mix: 元の混合音 [B, T](2出力の場合の残差計算に使う)
    Returns:
        speaker_est: 話者2つの推定 [B, 2, T]
        noise_est: 雑音の推定 [B, T]
    """
    if t_est.size(1) == 2:
        speaker_est = t_est
        noise_est = mix - t_est.sum(dim=1)
    else:
        assert t_est.size(1) == 3
        speaker_est = t_est[:, :2]
        noise_est = t_est[:, 2]
    return speaker_est, noise_est


def remix_with_noise(speaker_est: torch.Tensor, noise_est: torch.Tensor, perm: torch.Tensor):
    """雑音を明示的に扱うリミキシング(DEC-016)。

    教師は話者2つしか出力しないため、雑音推定は残差 ñ = m − (s̃1+s̃2) として
    呼び出し側で計算して渡す。話者2のスロットだけをバッチ内で入れ替え、
    雑音は自分の録音のものをそのまま使う(入れ替えない)。これにより疑似混合の
    雑音は常に1つの録音由来のみになり、実際の混合音(雑音は1つ)に近い分布になる:

        m̃_b = s̃1_b + s̃2_{perm(b)} + ñ_b

    Args:
        speaker_est: 教師の話者推定 [B, 2, T](slot0 = s̃1, slot1 = s̃2)
        noise_est: 雑音の推定(残差) [B, T]。話者間で入れ替えない
        perm: バッチ内置換 [B](sample_derangement の出力、話者2のみに適用)
    Returns:
        remixed_mix: m̃  [B, T]
        pseudo_targets: (s̃1, s̃2[perm], ñ) を積んだ [B, 3, T]
    """
    assert speaker_est.dim() == 3 and speaker_est.size(1) == 2
    assert noise_est.dim() == 2
    s1 = speaker_est[:, 0]
    s2p = speaker_est[:, 1][perm]
    pseudo_targets = torch.stack([s1, s2p, noise_est], dim=1)
    return s1 + s2p + noise_est, pseudo_targets
