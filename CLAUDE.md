# CLAUDE.md — AIエージェント作業ガイド

- プロジェクト研究A: RemixIT × TF-Locoformer による、言語をまたいだ話者分離の研究(発表: 2026-08-10)
- 研究の背景・決定・計画は [docs/README.md](docs/README.md) から辿ること

## ドキュメントの文体ルール

1. なるべく箇条書きにする。長い段落は避ける
2. だ・である調で書く。です・ます調は使わない
3. 一般的でない単語は使わず、平易な言葉で書く。普通に使う単語(GPU、エポック、損失など)はそのまま使ってよい
4. 造語は使わない

## 絶対に守るルール

1. GPUを使う前に必ず`nvidia-smi`で他の人が使っていないか確認する(共有サーバー、RTX 2080 Ti 11GB×3)。空いているGPUを`CUDA_VISIBLE_DEVICES`で指定する。全部埋まっていたら実行せず報告する。他に使っている人がいなければ2枚まで同時に使ってよい(研究室了承済み、DEC-012)。3枚すべての同時使用はしない
2. 容量が大きいものはローカルに置かない。ルートの記憶領域は満杯(空き200MB前後)。生成したデータ・チェックポイント・ログは`data/` `exp/`(NAS `/mnt/kiso-qnap5/kueki/proken-A/`へのシンボリックリンク)の配下に置く
3. 意思決定は [docs/03_decision_log.md](docs/03_decision_log.md) に記録する(決定・背景・選択肢・理由)
4. コードの変更は [docs/06_change_log.md](docs/06_change_log.md) に「何を・なぜ・どう変えたか」を記録する
5. RemixITの学習コード(E2/E3)は正解(s1/s2)に絶対にアクセスしない(mix/だけを使う)。設計上の要請

## 実験の記録

- 学習ジョブは W&B に記録する(project: `jvs-remixit`, entity: `97kuek-waseda-university`, DEC-011)。実行の名前は実験の番号(e1_supervised / e2_remixit_static / e3_remixit_seq)にそろえる
- 推論だけの評価(E0など)は`exp/`配下のCSV/JSONだけでよい

## 環境

- conda環境: `tflocoformer`(Python 3.10 / PyTorch 2.1.0+cu118 / ESPnet 202402)。実行例: `conda run -n tflocoformer python scripts/xxx.py`
- ESPnetは部品として使う(書き換えない)。モデルは`espnet2.enh.separator.tflocoformer_separator`
- NAS(NFS)上では`git clone`が失敗する。tarball展開 + LFSのAPIで代用する(scripts/download_teacher.shを参照)

## 主要なパス

| 用途 | パス |
|---|---|
| JVS(16kHzに変換・話者が重ならない分割済み) | `~/corpora/jvs/{train,valid,test}_wavs.txt` |
| 2話者混合(クリーン版, 対照条件) | `data/jvs2mix_8k/{train,valid,test}/{mix,s1,s2}/` |
| 2話者混合(雑音版, 適応先の本体, DEC-010) | `data/jvs2mix_noisy_8k/{train,valid,test}/{mix,s1,s2,noise}/` |
| WHAM! noise(適応先の雑音の素材) | `/mnt/kiso-qnap5/kueki/proken-A/corpora/wham_noise/{tr,cv,tt}/` |
| 教師のチェックポイント(MERL, 8kHz) | `/mnt/kiso-qnap5/kueki/proken-A/third_party/tf-locoformer/egs2/*/enh1/exp/*/valid.loss.ave_5best.pth`(採用した教師はwhamr/、DEC-010) |
| RemixITの共通コード | `remixit/`(remix.py / datasets.py / losses.py / separator.py / training.py) |
| テスト | `tests/`(pytest) |
| 実験の成果物 | `exp/`(NAS) |

## 実験の流れ(詳細: docs/04_experiment_plan.md、進み具合: docs/07_results.md)

- E0 教師を追加学習なしで評価(進めてよいかの目安: SI-SNR改善量5dB以上)→ E1 教師あり学習の上限 → E2 RemixIT(教師固定)→ E3 RemixIT(教師を逐次更新)
- E0(クリーン・雑音とも)完了。教師はWHAMR!版に決定した(雑音の下でのSI-SNR改善量9.09dBで最も良い)
- E1・E2は雑音版データ(jvs2mix_noisy_8k)で学習中

## 関連プロジェクト

- `~/projects/jchat-sep` は同じテーマの先行プロジェクト(仕切り直す前のもの)。コードはコピーしない(DEC-006)が、教訓(学習率の自動調整、バッチサイズ2以上、RemixITの検証損失の解釈)とデータの参照は可
