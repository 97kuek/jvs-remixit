# 論文ノート: TF-Locoformer (Saijo et al., 2024)

> TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement
> arXiv:2408.03440 / IWAENC 2024。公式実装: https://github.com/merlresearch/tf-locoformer

## 一言でいうと

TF領域デュアルパスモデルから**RNNを排除**し、Transformer + 畳み込みによる局所モデリング(ConvSwiGLU)で TF-GridNet 級以上の SoTA 性能を出す音声分離・強調モデル。

## 背景

- TF領域デュアルパス(周波数方向と時間方向のモデリングを交互に行う)が現在の主流(TF-GridNet がSoTA)
- ただし TF-GridNet は RNN(BLSTM)依存 → 学習の並列化ができず遅い、スケールしにくい
- Transformer は大域モデリングは得意だが**局所モデリングが弱く**、TF領域デュアルパスでは単体では勝てなかった
- → self-attention に大域、畳み込みに局所を担当させる設計

## アーキテクチャ

- 入力: STFT の実部虚部 X ∈ R^{2×T×F} → Conv2D+gLN で D次元特徴 Z ∈ R^{D×T×F}
- **周波数モデリング → 時間モデリング** を B 回交互に実行(デュアルパス)
- 各ブロック(マカロン構造):
  1. Z ← Z + ConvSwiGLU(Z)/2
  2. Z ← Z + MHSA(Norm(Z))   ※RoPE(回転位置エンコーディング)使用
  3. Z ← Z + ConvSwiGLU(Z)/2
- **ConvSwiGLU**: FFNの線形層を Conv1D/Deconv1D に置換 + SwiGLU ゲート活性化
- **RMSGroupNorm**: 各TFビンのD次元ベクトルをG群に分けて正規化(話者などの概念のdisentangleを促す)。RMSNormより一貫して微改善
- 出力: Deconv2D → 複素スペクトルマッピング(RI成分直接推定)→ iSTFT

## モデルサイズ

| 設定 | D | B | C | K | H | G | params |
|---|---|---|---|---|---|---|---|
| Small | 96 | 4 | 256 | 4 | 4 | 4 | 5.0M |
| Medium | 128 | 6 | 384 | 4 | 4 | 4 | 15.0M |
| Large | 128 | 9 | 384 | 4 | 4 | 4 | 22.5M |

## 学習設定(再現時の参考)

- ESPnet-SE パイプライン。STFT窓 16ms / hop 8ms(WHAMR!のみ窓32ms, K=8, C半減)
- AdamW (wd=1e-2), warmup 4000ステップで lr 0→1e-3、val loss 3エポック停滞で×0.5、early stopping 10エポック
- batch 4、入力4秒、勾配クリップ L2=5、入力を標準偏差で正規化
- 分離: PIT + negative SI-SNR loss / 強調: 時間領域L1 + マルチ解像度TF領域L1

## 主要結果

- WSJ0-2mix: Medium 23.6dB / Large 24.2dB SI-SNRi(TF-GridNet 23.5dB と同等以上)、DM併用で25.1dB
- Libri2Mix: Medium 22.1dB(SoTA)
- WHAMR!(残響+雑音): Small 17.4dB で TF-GridNet(17.1dB)超え。**TF領域モデルは残響に強い**
- DNS2020(強調): Medium で SI-SNR 23.3dB, PESQ 3.72(SoTA)
- Ablation: カーネルK=3,4が最適(K=1=純Transformerは大幅劣化)→ 局所モデリングが本質的に重要

## この研究プロジェクトへの示唆

- 公式コードが公開されている(ESPnet依存だがモデル部分は流用可能)→ 実装リスク低
- 強調タスク(DNS)でも実証済み → RemixIT(強調の枠組み)との組み合わせは自然
- 計算コスト: Medium を DNS 2700h でフル学習するのは研究室規模では厳しい可能性 → Small またはさらに縮小した設定 + 小規模コーパスが現実的
- RemixIT では教師が M=2(speech+noise)出力を持つ必要がある → TF-Locoformer の出力ヘッドを N=2 にすれば適合。mixture consistency の追加を検討
