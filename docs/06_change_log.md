# 変更ログ (Change Log)

コードの変更を「何を・なぜ・どう変えたか(どういう考えで)」の形式で記録する。
研究方針の決定は [03_decision_log.md](03_decision_log.md)、このファイルは実装の記録。

---

## 2026-07-20: 初期実装(W1 データ準備・評価基盤)

### scripts/conf/data_jvs2mix.yaml — 混合生成条件の外部化(新規)

- **考え**: 再現性のため、データ生成の全パラメータ(SNR範囲、混合数、seed)を設定ファイルに明文化し、コマンドライン引数での場当たり的な上書きを避ける(jchat-sep の教訓: 手動変更は再現性を壊す)。
- **内容**: 8kHz(教師 ckpt に合わせる, DEC-008)、SNR一様[-5,5]dB、train 20k / valid 1k / test 1k、seed固定。

### scripts/prepare_jvs_mix.py — JVS 2話者混合の生成(新規)

- **考え**:
  - **2段階構成**(①16k→8kリサンプルキャッシュ → ②混合生成)にした。リサンプルは全12,997発話に対し1回だけ行えばよく、混合生成のやり直し(条件変更)時に再実行不要になるため。キャッシュには index JSON(パス→サンプル数)を持たせ、2回目以降は音声を読まずに長さフィルタできる。
  - **SNRのかけ方**: s1は無加工、s2のみスケール。クリップ回避は「全信号を同率スケール」にした。s1/s2/mixを別々に正規化すると mix = s1 + s2 の関係とSNRが壊れ、評価が狂うため。
  - **min長 fully-overlapped**(短い方に合わせて切る): TF-Locoformer論文のWSJ0-2mix系標準(min版)に合わせた。
  - split ごとに seed をずらして固定(train:+0, valid:+1, test:+2)。同一 seed での再実行は同一データを再生成する(決定的)。
- **検証**: 20混合のスモークで mix−(s1+s2) の最大誤差 0、実測SNRとメタデータの一致、8kHz出力を確認。

### scripts/download_teacher.sh — 教師チェックポイント取得(新規)

- **考え**: MERL リポジトリの学習済み .pth は Git LFS 管理で、通常は `git clone`+`git lfs pull` で取るのが筋。しかし **NAS(NFSv3)上では git clone が index-pack で失敗**し、ローカルディスクは満杯(DEC-009)でクローン先がない。そこで (1) コードは GitHub の tarball 展開で取得(gitオブジェクト不要)、(2) LFS ポインタファイル(oid/size 記載)から **LFS batch API を直接叩いて実体をダウンロード**する方式にした。
- **安全策**: ダウンロード後にサイズ照合し、不一致なら失敗させる。実体化済みファイルはスキップ(冪等)。
- **結果**: 5つ全チェックポイント(WSJ0-2mix / Libri2mix / WHAMR! Medium+Small / DNS)を実体化。WSJ0-2mix 版(Medium 15.0M params)の state_dict 読み込みを確認。

### scripts/eval_separation.py — PIT + SI-SNR/SI-SNRi 評価(新規)

- **考え**:
  - 推論は ESPnet の `SeparateSpeech` を使う。教師 ckpt は config.yaml 同梱の ESPnet 形式なので、自前でモデルを組むより構成ミスのリスクが小さい。E0(教師)と E1-E3(生徒)で**同一の評価コード**を使うことで、実験間の比較を公平にする。
  - SI-SNR は数式通り自前実装(数十行)。mir_eval 等への依存を増やさない。PIT は2話者なので全2置換の総当たり。
  - SI-SNRi は「入力 mix をそのまま推定とみなした SI-SNR」を引いて算出。per-file CSV と要約 JSON(平均・中央値・標準偏差)を出力し、発表用の箱ひげ図が後から作れるようにした。

### data/ exp/ → NAS シンボリックリンク(構成変更)

- **考え**: ルートFSが満杯(空き199MB、主因は他ユーザー)で、ローカルにデータもチェックポイントも置けない(DEC-009)。リポジトリ内の相対パス(`data/...`)を保ったまま実体をNASに置くため、ディレクトリ本体ではなくシンボリックリンクにした。コードはパスを意識しなくてよい。

### scripts/prepare_jvs_mix.py — 雑音付与対応(DEC-010) (2026-07-20)

- **背景**: E0 で英語教師がクリーン日本語混合をゼロショット 25dB で分離してしまい、適応の余地がないと判明。適応先を WHAM! noise つき混合に難化する(DEC-010)。
- **考え**:
  - config の `noise:` セクションの有無で同一スクリプトがクリーン版/雑音版を両対応。クリーン版の再生成結果は従来と完全一致(後方互換)。
  - 雑音 SNR は WHAM! 準拠で「大きい方の話者パワー」基準の一様 [-6,+3] dB。雑音素材は WHAM! の tr/cv/tt を train/valid/test に対応させ disjoint に保つ。
  - **雑音専用の独立 rng**(seed+1000)を使う。共有 rng だと雑音の乱数消費が後続の話者ペア抽選をずらし、クリーン版と混合系列が一致しなくなるため。同一 seed のクリーン版と「同じ話者ペア・同じ話者間 SNR で雑音だけ違う」対照比較ができる。
  - 雑音はステレオの ch0 を使用(WHAM! の 1ch タスクの慣例)、短い場合はタイル+ランダムオフセット切り出し。noise/ サブディレクトリにも保存し、診断や E4 に使えるようにした。

### scripts/train_remixit.py — 早期終了の追加(E2実行時間の見積もりミス修正) (2026-07-20)

- **問題**: E1/E2 学習開始 1.5〜2 時間後、W&B ログの実測ペースから E2 が約183分/epoch と判明。当時の config(epochs:60, 早期終了なし)のままだと E2 だけで約7.6日かかり、締切に間に合わないことが分かった(詳細: DEC-013)。
- **考え**: E1 には早期終了を実装していたのに E2 に入れ忘れていたのは単純な抜け漏れ。ただし E2/E3 は「教師更新」がある分 E1 より複雑で、素朴に E1 のロジックをコピーすると sequential 教師更新(E3)で誤動作する: 教師が変わると検証損失の意味(教師との一致度)がリセットされるため、**教師更新をまたいで best を比較するとその時点の教師の良し悪しに引きずられ、正しく収束判定できない**。そのため教師更新のたびに `best, patience = inf, 0` にリセットするようにした。
- **設定変更**: epochs 60→30、early_stop_patience:8 追加、update_every_epochs 20→15(epochs 縮小に比例)。
- **運用の教訓**: 学習ジョブ起動後は最初の1エポックが終わった時点で W&B の実測ペースを確認し、config の epochs 数 × 実測時間が締切に収まるか検算する運用にする。
- **プロセス停止の注意点**: `conda run -n ... python script.py &` で起動したジョブを止める際、`kill <conda run のPID>` だけでは実体の python プロセス(と DataLoader worker の子プロセス)が終了しない。`ps aux` で実体 PID を確認し、`nvidia-smi --query-compute-apps` で GPU メモリが実際に解放されたことを確認してから次のジョブを起動する必要がある(今回、最初の kill で子プロセスが生き残っていたことに気づかず一度追加調査が必要になった)。

### ドキュメント整備(適応先変更・教師確定の反映) (2026-07-20)

- **考え**: 適応先の難化(DEC-010)・教師の WHAMR! 確定・E1/E2 起動という一連の変更が、README / CLAUDE.md / 00_research_motivation.md には反映されておらず、設計変更前の記述(クリーン JVS のみ、教師未確定)が残っていた。研究ドキュメントは常に「現状」を指すべきという方針(CLAUDE.md ルール3,4)に沿って棚卸しした。
- **変更内容**:
  - README.md: 「現在の状況」節を新設(E0結果・教師確定・学習中の実験)。セットアップ手順に WHAM! noise 取得・雑音版生成・テスト実行を追記。`download_wham_noise.sh` を新規作成(これまで場当たり的な curl コマンドで取得していたものをスクリプト化し再現可能にした)。
  - CLAUDE.md: 主要パス表に雑音版データセット・WHAM! noise・remixit/・tests/ を追加、教師を WHAMR! 版と明記。
  - docs/00_research_motivation.md: スコープの記述を「クリーン日本語適応」→「言語+音響環境の二重ドメインギャップ適応」に更新。設計変更の経緯(クリーンでは簡単すぎた発見)を明文化し、想定Q&Aに追加。
  - docs/03_decision_log.md DEC-003: 方針確定前の下書き(SpeechBrain/Asteroid 教師案、Libri2Mix 適応先案)が最終決定と矛盾したまま残っていたため、「不採用の下書き」と明記して誤読を防いだ(削除ではなく経緯として保持)。
  - docs/04_experiment_plan.md: 07_results.md への進捗ポインタを追加。

### W2 中核実装: remixit/ パッケージ + 学習スクリプト (2026-07-20)

- **remixit/remix.py**: バッチ内リミキシング。置換は**不動点なし(derangement)**でサンプル — 不動点があるとその要素は「教師推定の和 ≒ 元の混合」となり学習信号が弱まるため。
- **remixit/datasets.py**: `MixtureOnlyDataset` は mix/ しか参照しない設計(RemixIT が正解に触れないことをクラスレベルで保証)。テストで s1/s2 へのファイルアクセスがないことを監視して検証。
- **remixit/losses.py**: 2話者 PIT + negative SI-SNR を自前実装(2置換の総当たり)。ESPnet の損失クラスへの依存を避け、リミキシング損失と評価で同一実装を使う。
- **remixit/separator.py**: 教師(MERL exp から構築)と生徒(自前 config からスクラッチ)を同一の波形 in/out ラッパ `SeparationModel` に統一。RemixIT の sequential 教師更新は `deepcopy(student)` で実現(教師と生徒のアーキテクチャが異なっても成立する。RemixIT 論文も学生で教師を置換する方式)。
- **remixit/training.py**: warmup+cosine スケジューラ(jchat-sep の教訓: 最初から自動化)、自前 ckpt 形式、W&B 初期化。
- **scripts/train_supervised.py (E1) / train_remixit.py (E2/E3)**: 1スクリプト1実験。RemixIT の検証損失は「教師との一致度」であり発散検知用(docstring に明記)。teacher_update: static/sequential を config で切替。
- **バッチサイズの実測 (2080 Ti 11GB)**: batch4×4s → OOM、batch4×3s → OOM(10.6GB)、**batch2×4s → OK** → batch 2 + 勾配累積 2(実効4)で確定。B=2 でも derangement は成立し(=スワップ)、エポックごとにバッチの組成が変わるためリミキシングの多様性はステップ間で確保される。
- **eval_separation.py**: `--student_ckpt` を追加し、教師(ESPnet形式)と生徒(自前形式)を同一評価経路で比較できるようにした。
- スモーク: E1/E2 とも GPU で数ステップの学習が通ることを確認。ユニットテスト 6 件 pass。

### remixit/espnet_compat.py — TF-Locoformer の実行時登録(新規)

- **問題**: 教師 ckpt の config.yaml は `separator: tflocoformer` を指定するが、環境の ESPnet 202402 の `separator_choices` に tflocoformer が未登録で、`SeparateSpeech` の構築が ValueError で失敗した(モジュール自体は環境に存在)。MERL 公式の手順は espnet2/tasks/enh.py への**パッチ適用**。
- **考え**: ESPnet 本体は改変しない方針(05_architecture.md)のため、パッチではなく**実行時登録**にした。`separator_choices.classes` は素の dict なので、互換レイヤモジュール `remixit/espnet_compat.py` の import 時に 1 エントリ追加するだけで済む。ESPnet を再インストールしても壊れず、何をしているかがコード上で明示される。
- **使い方**: モデル構築(SeparateSpeech / build_model)より前に `import remixit.espnet_compat` する。

### リポジトリ骨組み(新規)

- README.md / CLAUDE.md / .gitignore / docs 一式。設計の全体像は [05_architecture.md](05_architecture.md)。
- git init 済み。`data` `exp`(シンボリックリンク)と `*.pth` は git 管理外。
