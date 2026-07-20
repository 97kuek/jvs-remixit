# 論文ノート: RemixIT (Tzinis et al., 2022)

> RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing
> arXiv:2202.08862v3 / IEEE JSTSP

## 一言でいうと

**クリーン音声もノイズ単体波形も一切使わずに**、ドメイン内のノイジー混合音声だけで音声強調(speech enhancement)モデルを自己学習(self-training)させる手法。教師(teacher)-生徒(student)フレームワーク + 「ブートストラップ・リミキシング」が肝。

## 問題設定

- 音声強調 = 混合音 m = s + n から音声 s を復元するタスク
- 教師あり学習はクリーン音声 D_s とノイズ D_n のペアが必要 → 実環境ではドメインミスマッチに弱い
- MixIT は分離済みのドメイン内ノイズ録音が必要という制約がある
- **RemixIT はドメイン内の「混合音データセット D_m」だけでよい**(最も制約が緩い)

## 手法の流れ (Algorithm 1)

1. **教師の事前学習**: OODデータ(ドメイン外)で教師モデル f_T を学習(教師あり or MixIT どちらでも可)
2. ドメイン内の混合音バッチ m ~ D_m に対して教師が推定: (s̃, ñ) = f_T(m)
3. **ブートストラップ・リミキシング**: バッチ内でノイズ推定 ñ を**ランダムに置換(permute)**し、音声推定と足し直す: m̃ = s̃ + P ñ
4. 生徒 f_S は m̃ を入力に、教師の推定 (s̃, Pñ) を疑似ターゲットとして通常の教師あり損失で学習
5. **教師の継続更新**: K ステップごとに教師を生徒の重みで更新
   - Sequential: θ_T ← θ_S(20エポック毎に置き換え)← 基本デフォルト
   - EMA: θ_T ← γθ_S + (1−γ)θ_T, γ=0.01(低リソース適応時はこちら)

## なぜ効くか(理論)

- L_RemixIT = 教師ありloss + 教師誤差ノルム(生徒に対し定数) − 2⟨生徒誤差, 教師誤差⟩
- リミキシングにより同じ教師音声推定 s̃* が多様なノイズと混ざる → 生徒誤差と教師誤差の相関項が期待値でゼロに近づく(Theorem II.1: B→∞ で ∇L_RemixIT ≈ ∇L_Supervised)
- 教師がかなり悪い推定(SI-SNR < 5dB)を出しても生徒は改善できる(Fig. 5)

## 実験セットアップ(参考にすべき点)

- モデル: Sudo rm -rf (U=8 ConvBlocks, 0.56M〜0.79M params)。**手法はアーキテクチャ非依存**と明言
- 出力は M=2 スロット(speech + noise)。mixture consistency layer で s̃+ñ=m を強制
- 損失: negative SI-SDR、Adam、lr=1e-3(6エポック毎に半減)、batch B=2、16kHz
- データセット: DNS2020 / LibriFSD50K (LFSD) / WHAM! / VCTK-DEMAND
- 評価: SI-SDR(i), PESQ, STOI

## 主要結果

- DNSテスト: 教師(OOD MixIT, 14.8dB) → 生徒 16.0dB SI-SDR。全MixIT系ベースラインを上回る
- 半教師あり設定(LFSDで教師あり事前学習→DNS適応): 17.6 → 18.0dB
- ゼロショットドメイン適応: EMA教師更新で+0.8dB程度の適応改善
- ドメイン内ノイズ録音が一部使えるなら Eq.17 の拡張(確率 p_n で実ノイズと教師推定ノイズを混用)でさらに向上

## この研究プロジェクトへの示唆

- RemixIT は「アーキテクチャ非依存」→ Sudo rm -rf を TF-Locoformer に置き換える提案は手法の趣旨に合致
- 元論文は時間領域の軽量モデルで実験 → TF領域SoTAアーキテクチャで検証する価値がある(新規性の芽)
- 教師の事前学習コストが高い → 事前学習済みチェックポイントの活用や小規模コーパスでのスケールダウンが現実的
- 教師更新プロトコル(sequential vs EMA)は適応データ量で選ぶ
