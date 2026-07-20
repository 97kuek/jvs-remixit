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

### リポジトリ骨組み(新規)

- README.md / CLAUDE.md / .gitignore / docs 一式。設計の全体像は [05_architecture.md](05_architecture.md)。
- git init 済み。`data` `exp`(シンボリックリンク)と `*.pth` は git 管理外。
