# proken-A: RemixIT × TF-Locoformer による話者分離の言語・雑音適応

- プロジェクト研究A(発表: 2026-08-10)
- 英語のクリーンな環境で学習された分離モデル(教師)を、正解ラベルなしの「雑音つき日本語混合音」(JVS + WHAM! noiseで作った模擬データ)だけを使ってTF-Locoformer(生徒)に適応させる、RemixIT自己学習の数値による検証

- 研究の目的・動機: [docs/00_research_motivation.md](docs/00_research_motivation.md)
- 意思決定ログ: [docs/03_decision_log.md](docs/03_decision_log.md)
- 実験計画: [docs/04_experiment_plan.md](docs/04_experiment_plan.md)
- システム設計: [docs/05_architecture.md](docs/05_architecture.md)
- コードの変更ログ: [docs/06_change_log.md](docs/06_change_log.md)
- 実験結果・進み具合: [docs/07_results.md](docs/07_results.md)

## 現在の状況(2026-07-20時点)

- E0(教師を追加学習なしで評価)完了。クリーンな日本語混合では英語教師だけでSI-SNR改善量25dB相当まで出てしまい適応の余地がなかったため、適応先をWHAM! noiseつきの日本語混合に難しくした(詳細: docs/03_decision_log.md DEC-010)
- 雑音の下では英語教師は大きく性能が落ちる(WSJ0-2mix 0.6dB / Libri2mix 3.2dB / WHAMR! 9.1dB)。教師はWHAMR!版に決定した
- E1(教師あり学習の上限)・E2(RemixIT、教師固定)を雑音版データで学習中

## セットアップ

- conda環境`tflocoformer`(ESPnet 202402 / PyTorch 2.1.0+cu118)を使う

```bash
# 1. 教師のチェックポイント(MERL公開, WSJ0-2mix/Libri2mix/WHAMR!/DNS)を取得
bash scripts/download_teacher.sh

# 2. WHAM! noise(適応先の雑音の素材, 17GB)を取得
bash scripts/download_wham_noise.sh

# 3. 混合音のデータを生成
#    クリーン版(対照条件)
conda run -n tflocoformer python scripts/prepare_jvs_mix.py \
    --config scripts/conf/data_jvs2mix.yaml
#    雑音版(適応先の本体)
conda run -n tflocoformer python scripts/prepare_jvs_mix.py \
    --config scripts/conf/data_jvs2mix_noisy.yaml

# 4. テストの実行
conda run -n tflocoformer python -m pytest tests/ -q
```

## 実験(詳細: docs/04_experiment_plan.md)

| 番号 | 内容 | スクリプト | 状態 |
|---|---|---|---|
| E0 | 英語教師の追加学習なしでの評価(クリーン・雑音) | `scripts/eval_separation.py` | 完了 |
| E1 | 教師あり学習の上限 | `scripts/train_supervised.py` | 学習中 |
| E2 | RemixIT(教師固定) | `scripts/train_remixit.py` | 学習中 |
| E3 | RemixIT(教師を逐次更新) | `scripts/train_remixit.py`(`teacher_update: sequential`) | 未着手 |

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n tflocoformer python scripts/train_supervised.py \
    --config scripts/conf/train_supervised.yaml --out_dir exp/e1_supervised

CUDA_VISIBLE_DEVICES=1 conda run -n tflocoformer python scripts/train_remixit.py \
    --config scripts/conf/train_remixit.yaml --out_dir exp/e2_remixit_static
```

- GPUを使う前に必ず`nvidia-smi`で他の人が使っていないか確認すること(詳細: [CLAUDE.md](CLAUDE.md))

## リポジトリの構成

```
docs/               研究のドキュメント(動機・意思決定ログ・実験計画・設計・変更ログ・結果)
remixit/            RemixIT用に使い回すコード(混ぜ直し・データセット・損失・モデル・学習の部品)
scripts/            データ準備・学習・評価のスクリプト(実行の単位)
scripts/conf/       各スクリプトの設定(YAML)
tests/              pytest(remixit/のテスト)
data/ exp/          NASへのシンボリックリンク(生成したデータ・学習の成果物、gitの管理対象外)
```
