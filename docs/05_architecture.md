# システム設計 (v1.0, 2026-07-20)

- 実装の構成と設計の判断を記す。実験の中身は [04_experiment_plan.md](04_experiment_plan.md)、決定の経緯は [03_decision_log.md](03_decision_log.md) を見ること

## 方針

- ESPnetは部品として使う(DEC-004の対策): モデル(`espnet2.enh.separator.tflocoformer_separator.TFLocoformerSeparator`)、STFTの前後処理、SI-SNR損失などはESPnetのクラスを読み込んで使い、学習の流れは自分で書いたスクリプトにする。ESPnet本体のソースコードは書き換えない(環境: conda環境`tflocoformer`、ESPnet 202402、PyTorch 2.1.0+cu118)
- 1つのスクリプトが1つの実験の段階を担う。データ準備 → E0の評価 → E1の教師あり学習 → E2/E3のRemixIT → 推論、が`scripts/`に1対1で並ぶ
- 設定はすべて`scripts/conf/*.yaml`に書き、実行時に手で上書きしない(jchat-sepの教訓: 再現できなくなる)
- 乱数はすべてのスクリプトで種を固定する。データ生成は学習・検証・評価それぞれ別の種を使う

## リポジトリの構成

```
proken-A/
├── docs/                       # 研究のドキュメント(このファイル群)
├── scripts/
│   ├── prepare_jvs_mix.py      # JVSを16kHz→8kHzに変換し、2話者の混合音を作る(学習・検証・評価)
│   ├── download_teacher.sh     # MERL公開のチェックポイントを取得する
│   ├── eval_separation.py      # 分離の評価(SI-SNR/SDRの改善量)。E0(教師)とE1-E3(生徒)の評価で共用
│   ├── train_supervised.py     # E1: 教師あり学習の上限
│   ├── train_remixit.py        # E2/E3: RemixIT(教師固定か逐次更新かは設定で切り替え)
│   ├── separate.py             # 単発の推論・聞き比べ用
│   └── conf/
│       ├── data_jvs2mix.yaml   # 混合音の生成条件
│       ├── model_small.yaml    # TF-Locoformerの構成(生徒)
│       ├── train_supervised.yaml
│       └── train_remixit.yaml
├── remixit/                    # 使い回すコード(スクリプトから読み込む)
│   ├── datasets.py             # 混合音のデータセット(教師あり用・混合音のみ用)
│   ├── separator.py            # ESPnetのTF-Locoformer + STFT前後処理をまとめたもの(教師・生徒で共用)
│   ├── remix.py                # バッチ内での混ぜ直し(話者の推定を入れ替える)
│   └── training.py             # 学習率の決め方、チェックポイントの保存、W&Bの初期化
├── tests/                      # pytest(入れ替えの検証、損失の動作、簡易確認)
├── exp/                        # 学習の成果物(gitの管理対象外)
└── data/                       # 生成した混合音のデータ(gitの管理対象外。実体は~/corporaまたはNAS)
```

## データの流れ

```
~/corpora/jvs/{train,valid,test}_wavs.txt   (話者が重ならないよう分割済み。jchat-sep由来の資産を再利用)
        │ prepare_jvs_mix.py: 別の話者どうしのペアを取り出す → 8kHzに変換し直す
        │   → SNR一様[-5,5]dBでスケールする → 短い方に合わせて完全に重なる混合音を作る
        ▼
data/jvs2mix_8k/{train,valid,test}/{mix,s1,s2}/*.wav + metadata.csv
        │
        ├─ E0: 教師(MERLのチェックポイント) → eval_separation.py(評価用データ)
        ├─ E1: train_supervised.py(mix・s1・s2をすべて使う)
        └─ E2/E3: train_remixit.py(mixだけを使う。s1・s2には触れない)
                     └ 評価のときだけ、評価用データのs1・s2を使う
```

- 生成の規模(初期値): 学習用20,000 / 検証用1,000 / 評価用1,000件の混合音(学習分でおよそ22時間、8kHz)
- RemixITの学習コードが正解(s1・s2)にアクセスしないことは、データセットのクラスの作りそのもので保証する(混合音だけを使うデータセットは`mix/`ディレクトリしか見ない)

## モデル・学習の要点

- 生徒: TFLocoformerSeparatorをSmall構成以下(特徴の次元D=96、ブロック数B=4、隠れ層C=256、畳み込みの窓幅K=4、頭の数H=4、グループ数G=4、パラメータ数約5.0M)で使う。11GBのメモリに合わせ、バッチサイズ2〜4+勾配の蓄積を使う
- 教師: MERL公開のチェックポイント(8kHz、STFTの窓16ミリ秒・ずらし幅8ミリ秒)。読み込んだあとは評価モード・勾配を計算しない状態にして推論だけに使う。逐次更新のときは生徒の重みをそのままコピーする
- 損失: PIT(話者の割り当てを考慮した損失)+ 負のSI-SNR(ESPnetの実装を利用)
- 混ぜ直し: `remix.py`で、入れ替えを「自分自身とは入れ替わらない並べ替え」からランダムに選ぶ(自分自身と混ぜ直すと元の混合音とほぼ同じ組み合わせになり、学習の手がかりが弱くなるため)。バッチサイズが2以上であることを必ず確認する
- 「音声1+音声2=入力の混合音」を強制する層: 最初の実装では入れない(教師の推定の和が入力とわずかにずれることは許容する)。効果がありそうならE4として確かめる

## テストの方針

- `tests/test_remix.py`: 入れ替えが「自分自身とは入れ替わらない並べ替え」になっていること、m̃ = s̃1 + Ps̃2 の数値が合っていること
- `tests/test_datasets.py`: 混合音だけを使うデータセットがs1・s2を読み込まないこと(ファイルへのアクセスを監視して確認)
- `tests/test_smoke_train.py`: 8件の混合音・1エポックでの簡易確認(教師あり学習・RemixITの両方、GPUがなくても実行できる)
