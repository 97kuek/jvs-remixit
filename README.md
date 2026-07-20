# proken-A: RemixIT × TF-Locoformer によるクロスリンガル話者分離

> プロジェクト研究A(発表: 2026-08-10)。
> 英語で学習された分離モデル(教師)を、正解ラベルなしの日本語混合音(JVS シミュ)だけで
> TF-Locoformer(生徒)に適応させる RemixIT 自己学習の定量検証。

- 研究の目的・動機: [docs/00_research_motivation.md](docs/00_research_motivation.md)
- 意思決定ログ: [docs/03_decision_log.md](docs/03_decision_log.md)
- 実験計画: [docs/04_experiment_plan.md](docs/04_experiment_plan.md)
- システム設計: [docs/05_architecture.md](docs/05_architecture.md)

## セットアップ

conda 環境 `tflocoformer`(ESPnet 202402 / PyTorch 2.1.0+cu118)を使用。

```bash
# 1. 混合データ生成(JVS 16kHz → 8kHz、2話者混合)
conda run -n tflocoformer python scripts/prepare_jvs_mix.py \
    --config scripts/conf/data_jvs2mix.yaml

# 2. 教師チェックポイント取得
bash scripts/download_teacher.sh

# 3. E0: 教師ゼロショット評価(Go/No-Go ゲート)
conda run -n tflocoformer python scripts/eval_separation.py --help
```

## 実験

| # | 内容 | スクリプト |
|---|---|---|
| E0 | 英語教師のゼロショット評価 | `scripts/eval_separation.py` |
| E1 | 教師あり上限 | `scripts/train_supervised.py` |
| E2/E3 | RemixIT(static / sequential) | `scripts/train_remixit.py` |
