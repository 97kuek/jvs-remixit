"""ESPnet 本体を改変せずに TF-Locoformer をタスクへ登録する互換レイヤ。

MERL 公式は espnet2/tasks/enh.py へのパッチ適用を前提とするが、本プロジェクトは
ESPnet をライブラリとして使う方針(docs/05_architecture.md)のため、
実行時に separator_choices へ登録する。SeparateSpeech やモデル構築より前に
このモジュールを import すること。
"""
from espnet2.enh.separator.tflocoformer_separator import TFLocoformerSeparator
from espnet2.tasks.enh import separator_choices

separator_choices.classes["tflocoformer"] = TFLocoformerSeparator
