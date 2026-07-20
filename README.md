# proken-A: RemixIT × TF-Locoformer によるクロスリンガル・ノイズ耐性話者分離

> プロジェクト研究A(発表: 2026-08-10)。
> 英語クリーン環境で学習された分離モデル(教師)を、正解ラベルなしの「雑音つき日本語混合音」
> (JVS + WHAM! noise シミュ)だけで TF-Locoformer(生徒)に適応させる RemixIT 自己学習の定量検証。

- 研究の目的・動機: [docs/00_research_motivation.md](docs/00_research_motivation.md)
- 意思決定ログ: [docs/03_decision_log.md](docs/03_decision_log.md)
- 実験計画: [docs/04_experiment_plan.md](docs/04_experiment_plan.md)
- システム設計: [docs/05_architecture.md](docs/05_architecture.md)
- コード変更ログ: [docs/06_change_log.md](docs/06_change_log.md)
- **実験結果・進捗**: [docs/07_results.md](docs/07_results.md)

## 現在の状況(2026-07-20 時点)

- E0(教師ゼロショット評価)完了。クリーン日本語混合では英語教師だけで SI-SNRi 25dB 相当まで出てしまい
  適応の余地がなかったため、適応先を **WHAM! noise つき日本語混合** に難化(詳細: 03_decision_log DEC-010)。
- 雑音下では英語教師は大きく劣化(WSJ0-2mix 0.6dB / Libri2mix 3.2dB / **WHAMR! 9.1dB**)。
  **教師は WHAMR! 版に確定。**
- E1(教師あり上限)・E2(RemixIT, static 教師)を雑音版データで学習中。

## セットアップ

conda 環境 `tflocoformer`(ESPnet 202402 / PyTorch 2.1.0+cu118)を使用。

```bash
# 1. 教師チェックポイント(MERL 公開, WSJ0-2mix/Libri2mix/WHAMR!/DNS)取得
bash scripts/download_teacher.sh

# 2. WHAM! noise(適応先の雑音源, 17GB)取得
bash scripts/download_wham_noise.sh

# 3. 混合データ生成
#    クリーン版(対照条件)
conda run -n tflocoformer python scripts/prepare_jvs_mix.py \
    --config scripts/conf/data_jvs2mix.yaml
#    雑音版(適応先本体)
conda run -n tflocoformer python scripts/prepare_jvs_mix.py \
    --config scripts/conf/data_jvs2mix_noisy.yaml

# 4. テスト
conda run -n tflocoformer python -m pytest tests/ -q
```

## 実験(詳細: docs/04_experiment_plan.md)

| # | 内容 | スクリプト | 状態 |
|---|---|---|---|
| E0 | 英語教師のゼロショット評価(クリーン/雑音) | `scripts/eval_separation.py` | 完了 |
| E1 | 教師あり上限(オラクル) | `scripts/train_supervised.py` | 学習中 |
| E2 | RemixIT(static 教師) | `scripts/train_remixit.py` | 学習中 |
| E3 | RemixIT(sequential 教師更新) | `scripts/train_remixit.py`(`teacher_update: sequential`) | 未着手 |

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n tflocoformer python scripts/train_supervised.py \
    --config scripts/conf/train_supervised.yaml --out_dir exp/e1_supervised

CUDA_VISIBLE_DEVICES=1 conda run -n tflocoformer python scripts/train_remixit.py \
    --config scripts/conf/train_remixit.yaml --out_dir exp/e2_remixit_static
```

GPU実行前に必ず `nvidia-smi` で他ユーザーの使用状況を確認すること(詳細: [CLAUDE.md](CLAUDE.md))。

## リポジトリ構成

```
docs/               研究ドキュメント(動機・決定ログ・実験計画・設計・変更ログ・結果)
remixit/            RemixIT 共通コード(リミキシング・データセット・損失・モデル・学習部品)
scripts/            データ準備・学習・評価スクリプト(実行単位)
scripts/conf/       各スクリプトの設定(YAML)
tests/              pytest(remixit/ のユニットテスト)
data/ exp/          NAS へのシンボリックリンク(生成データ・学習成果物、git管理外)
```
