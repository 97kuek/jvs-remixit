# システム設計 (v1.0, 2026-07-20)

> 実装の構成と設計判断。実験の中身は [04_experiment_plan.md](04_experiment_plan.md)、決定の経緯は [03_decision_log.md](03_decision_log.md)。

## 方針

- **ESPnet はライブラリとして使う**(DEC-004 の緩和策): モデル(`espnet2.enh.separator.tflocoformer_separator.TFLocoformerSeparator`)、STFTエンコーダ/デコーダ、SI-SNR損失などは ESPnet のクラスを import して使い、**学習ループは自作スクリプト**にする。ESPnet 本体のソースは改変しない(環境: conda `tflocoformer`, ESPnet 202402, PyTorch 2.1.0+cu118)。
- **1スクリプト=1実験段階**。データ準備 → E0評価 → E1教師あり → E2/E3 RemixIT → 推論、が `scripts/` に一対一で並ぶ。
- 設定はすべて `scripts/conf/*.yaml` に明文化し、実行時の手動上書きをしない(jchat-sep の教訓: 再現性)。
- 乱数は全スクリプトで seed 固定。データ生成は split ごとに独立 seed。

## リポジトリ構成

```
proken-A/
├── docs/                       # 研究ドキュメント(本ファイル群)
├── scripts/
│   ├── prepare_jvs_mix.py      # JVS 16kHz→8kHz + 2話者混合生成(train/valid/test)
│   ├── download_teacher.sh     # MERL 公開チェックポイント取得
│   ├── eval_separation.py      # 分離評価 (SI-SNRi/SDRi)。E0(教師ゼロショット)と E1-E3 の評価を共用
│   ├── train_supervised.py     # E1: 教師あり上限
│   ├── train_remixit.py        # E2/E3: RemixIT(static / sequential は conf で切替)
│   ├── separate.py             # 単発推論・試聴用
│   └── conf/
│       ├── data_jvs2mix.yaml   # 混合生成条件
│       ├── model_small.yaml    # TF-Locoformer 構成(生徒)
│       ├── train_supervised.yaml
│       └── train_remixit.yaml
├── remixit/                    # 再利用モジュール(スクリプトから import)
│   ├── datasets.py             # 混合データセット (supervised / mixture-only)
│   ├── separator.py            # ESPnet TF-Locoformer + STFT前後処理のラッパ(教師・生徒共用)
│   ├── remix.py                # バッチ内リミキシング(話者スロット置換)
│   └── teacher.py              # 教師の保持・更新プロトコル(static/sequential/EMA)
├── tests/                      # pytest(remix の置換検証、損失の PIT 動作、smoke)
├── exp/                        # 学習成果物(git 管理外)
└── data/                       # 生成した混合データ(git 管理外; 実体は ~/corpora または NAS)
```

## データフロー

```
~/corpora/jvs/{train,valid,test}_wavs.txt   (話者disjoint、jchat-sep 由来の資産を再利用)
        │ prepare_jvs_mix.py: 別話者ペアをサンプル → 8kHzへリサンプル
        │   → SNR一様[-5,5]dBでスケール → min長で切り出し fully-overlapped 混合
        ▼
data/jvs2mix_8k/{train,valid,test}/{mix,s1,s2}/*.wav + metadata.csv
        │
        ├─ E0: teacher(MERL ckpt) → eval_separation.py(test)
        ├─ E1: train_supervised.py(mix+s1+s2 使用)
        └─ E2/E3: train_remixit.py(**mix のみ使用**。s1/s2 は触らない)
                     └ 評価時のみ test の s1/s2 を使用
```

- 生成規模(初期値): train 20,000 / valid 1,000 / test 1,000 混合(train ≈ 22h @8kHz)。
- **RemixIT 学習コードは正解 (s1/s2) にアクセスしない**ことをデータセットクラスのレベルで保証する(mixture-only データセットは mix/ ディレクトリしか見ない)。

## モデル・学習の要点

- 生徒: TFLocoformerSeparator を **Small 構成以下**(D=96, B=4, C=256, K=4, H=4, G=4 ≈5.0M)で使用。11GB VRAM に合わせ batch 2〜4 + 勾配累積。
- 教師: MERL 公開 ckpt(8kHz, STFT 窓16ms/hop 8ms)。読み込み後は `eval()` + `no_grad` で推論のみ。sequential 更新時は生徒の state_dict をコピー。
- 損失: PIT + negative SI-SNR(ESPnet 実装を利用)。
- リミキシング: `remix.py` にて置換 P を「不動点なし(derangement)」でサンプル(自分自身と再混合すると恒等ペアになり学習信号が弱まるため)。batch ≥ 2 を assert。
- mixture consistency: 初期実装では入れない(教師推定 s̃1+s̃2 ≒ m の誤差は許容)。効果があれば E4 扱いで検証。

## テスト方針

- `tests/test_remix.py`: 置換が derangement であること、m̃ = s̃1 + Ps̃2 の数値一致
- `tests/test_datasets.py`: mixture-only データセットが s1/s2 を読まないこと(パス監視)
- `tests/test_smoke_train.py`: 8混合・1エポックの smoke(supervised / remixit 両方、CPU可)
```
