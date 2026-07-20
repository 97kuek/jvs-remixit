# CLAUDE.md — AIエージェント作業ガイド

プロジェクト研究A: RemixIT × TF-Locoformer によるクロスリンガル話者分離(発表: 2026-08-10)。
研究の背景・決定・計画は [docs/README.md](docs/README.md) から辿ること。

## 絶対に守るルール

1. **GPU実行前に必ず `nvidia-smi` で他ユーザーの使用状況を確認する**(共有サーバー、RTX 2080 Ti 11GB ×3)。
   空いているGPUを `CUDA_VISIBLE_DEVICES` で明示指定。全部埋まっていたら実行せず報告。
   他に使用者がいなければ **2枚まで同時使用可**(研究室了承済み、DEC-012)。3枚全部の占有はしない。
2. **大容量物はローカルに置かない**。ルートFSは満杯(空き200MB前後)。生成データ・チェックポイント・
   ログは `data/` `exp/`(→ NAS `/mnt/kiso-qnap5/kueki/proken-A/` へのシンボリックリンク)配下に置く。
3. **意思決定は [docs/03_decision_log.md](docs/03_decision_log.md) に ADR 形式で記録**(決定・背景・選択肢・理由)。
4. **コード変更は [docs/06_change_log.md](docs/06_change_log.md) に「何を・なぜ・どう変えたか」を記録**。
5. RemixIT の学習コード(E2/E3)は正解 s1/s2 に**絶対にアクセスしない**(mix/ のみ)。設計上の要請。

## 実験ログ

- 学習ジョブは **W&B** に記録する(project: `jvs-remixit`, entity: `97kuek-waseda-university`, DEC-011)。
  run 名は実験 ID(e1_supervised / e2_remixit_static / e3_remixit_seq)に揃える。
- 推論のみの評価(E0 等)は `exp/` 配下の CSV/JSON のみでよい。

## 環境

- conda 環境: `tflocoformer`(Python 3.10 / PyTorch 2.1.0+cu118 / ESPnet 202402)
  実行例: `conda run -n tflocoformer python scripts/xxx.py`
- ESPnet は**ライブラリとして使用**(改変しない)。モデルは `espnet2.enh.separator.tflocoformer_separator`
- NFS(NAS)上では `git clone` が失敗する → tarball + LFS API(scripts/download_teacher.sh 参照)

## 主要パス

| 用途 | パス |
|---|---|
| JVS(16kHz変換・話者disjoint分割済み) | `~/corpora/jvs/{train,valid,test}_wavs.txt` |
| 2話者混合(クリーン版, 対照条件) | `data/jvs2mix_8k/{train,valid,test}/{mix,s1,s2}/` |
| **2話者混合(雑音版, 適応先本体, DEC-010)** | `data/jvs2mix_noisy_8k/{train,valid,test}/{mix,s1,s2,noise}/` |
| WHAM! noise(適応先の雑音源) | `/mnt/kiso-qnap5/kueki/proken-A/corpora/wham_noise/{tr,cv,tt}/` |
| 教師チェックポイント(MERL, 8kHz) | `/mnt/kiso-qnap5/kueki/proken-A/third_party/tf-locoformer/egs2/*/enh1/exp/*/valid.loss.ave_5best.pth`(**採用教師 = whamr/**, DEC-010) |
| RemixIT 共通コード | `remixit/`(remix.py / datasets.py / losses.py / separator.py / training.py) |
| 単体テスト | `tests/`(pytest) |
| 実験成果物 | `exp/`(NAS) |

## 実験の流れ(詳細: docs/04_experiment_plan.md, 進捗: docs/07_results.md)

E0 教師ゼロショット評価(Go/No-Go: SI-SNRi ≥ 5dB)→ E1 教師あり上限 → E2 RemixIT(static) → E3 RemixIT(sequential)

- E0(クリーン・雑音とも)完了。教師は WHAMR! 版に確定(雑音下 SI-SNRi 9.09dB で最良)。
- E1・E2 は雑音版データ(jvs2mix_noisy_8k)で学習中。

## 関連プロジェクト

`~/projects/jchat-sep` は同テーマの先行プロジェクト(仕切り直し前)。**コードはコピーしない**(DEC-006)が、
教訓(lr自動スケジュール、batch≥2、RemixIT検証損失の解釈)とデータ資産の参照は可。
