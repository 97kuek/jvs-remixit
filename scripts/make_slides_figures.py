#!/usr/bin/env python
"""発表資料用の図をPythonで生成する(2026-07-30時点で確定している数値のみ使用)。

出力先: docs/figures/*.png(パワーポイントへの貼り付け用、300dpi)
数値の出典はすべて exp/*/summary_*.json および docs/07_results.md。
"""
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 日本語フォント(Noto Sans CJK JP)を明示的に指定する
jp_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(jp_font_path)
jp_font = fm.FontProperties(fname=jp_font_path).get_name()
matplotlib.rcParams["font.family"] = jp_font
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# dataviz skill の色(カテゴリカル、固定順)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#8a8a86"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"

matplotlib.rcParams.update({
    "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def bar_with_labels(ax, x, values, colors, width=0.6, fmt="{:+.2f}"):
    bars = ax.bar(x, values, width=width, color=colors, zorder=3)
    vmin, vmax = min(values + [0]), max(values + [0])
    span = max(vmax - vmin, 1e-6)
    pad = span * 0.06
    for b, v in zip(bars, values):
        va = "bottom" if v >= 0 else "top"
        offset = pad if v >= 0 else -pad
        ax.text(b.get_x() + b.get_width() / 2, v + offset, fmt.format(v),
                ha="center", va=va, fontsize=11, color=TEXT)
    ax.axhline(0, color="#d8d7d2", linewidth=1, zorder=1)
    ax.grid(axis="y", color="#eeeeec", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    # ラベル分の余白を上下に確保する(値ラベルが軸目盛やタイトルと衝突しないように)
    ax.set_ylim(vmin - span * 0.22, vmax + span * 0.22)
    return bars


# ── 図1: E0 クリーン vs 雑音つき(3教師) ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
teachers = ["WSJ0-2mix\n(英語, クリーン学習)", "Libri2mix\n(英語, クリーン学習)", "WHAMR!\n(英語, 雑音+残響あり学習)"]
clean = [19.66, 25.37, None]
noisy = [0.57, 3.18, 9.09]
x = range(len(teachers))
w = 0.35
bars_clean = ax.bar([i - w / 2 for i in x if clean[i] is not None],
                     [v for v in clean if v is not None], width=w, color=BLUE,
                     label="クリーンな日本語混合", zorder=3)
bars_noisy = ax.bar([i + w / 2 for i in x], noisy, width=w, color=ORANGE,
                     label="雑音つき日本語混合", zorder=3)
for i, v in enumerate(clean):
    if v is not None:
        ax.text(i - w / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=11, color=TEXT)
for i, v in enumerate(noisy):
    ax.text(i + w / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=11, color=TEXT)
ax.set_xticks(list(x))
ax.set_xticklabels(teachers, fontsize=11)
ax.set_ylabel("SI-SNR 改善量 [dB]", fontsize=12)
ax.axhline(0, color="#d8d7d2", linewidth=1, zorder=1)
ax.grid(axis="y", color="#eeeeec", linewidth=1, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig1_e0_clean_vs_noisy.png")
plt.close(fig)

# ── 図2: 主結果のはしご(E0 → E2 → E1) ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
labels = ["入力\n(無加工)", "E0\n教師そのまま\n(WHAMR!版)", "E2\nRemixIT\n(正解なし自己学習)", "E1\n教師あり学習\n(上限)"]
values = [0.0, 9.09, -1.61, 13.23]
colors = [GRAY, BLUE, ORANGE, AQUA]
bar_with_labels(ax, range(len(labels)), values, colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("SI-SNR 改善量 [dB]", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig2_main_ladder.png")
plt.close(fig)

# ── 図3: 診断の過程で観測された実データ品質のピーク値 ──────────────
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
labels = ["E2 (元の構成)\n最終値\n[1000件]",
          "E2 (雑音分離後)\n初期のピーク\n[300件]",
          "E3 (教師逐次更新)\n初期のピーク\n[300件]",
          "E2 (バッチ拡張)\nエポック9時点\n[300件, 進行中]"]
values = [-1.61, 0.14, 0.38, 0.66]
colors = [GRAY, ORANGE, YELLOW, BLUE]
bar_with_labels(ax, range(len(labels)), values, colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("SI-SNR 改善量 [dB]", fontsize=12)
ax.text(0.5, 0.02, "※右端(バッチ拡張)は学習継続中の途中経過。最終値ではない",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=TEXT_MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig3_diagnosis_progress.png")
plt.close(fig)

# ── 図4: 「教師との一致度」と「実品質」の乖離(E2, 雑音分離後) ──────
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
stages = ["学習序盤\n(エポック1-2)", "学習後半\n(エポック20+)"]
real_quality = [0.14, -0.84]
bar_with_labels(ax, range(len(stages)), real_quality, [ORANGE, GRAY], width=0.45, fmt="{:+.2f} dB")
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=12)
ax.set_ylabel("SI-SNR 改善量 [dB]", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_internal_vs_real_divergence.png")
plt.close(fig)

print("生成した図:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p)
