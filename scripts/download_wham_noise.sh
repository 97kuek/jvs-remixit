#!/usr/bin/env bash
# WHAM! noise(17GB, 適応先ドメインの難化用, DEC-010)を NAS へ取得・展開する。
# 公式URL(storage.googleapis.com/whisper-public/wham_noise.zip)は 404 のため、
# WHAM! 公式サイトが案内する S3 バケット経由の URL を使う。
# 使い方: bash scripts/download_wham_noise.sh
set -euo pipefail

URL="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
DEST_DIR=/mnt/kiso-qnap5/kueki/proken-A/corpora
ZIP="$DEST_DIR/wham_noise.zip"

mkdir -p "$DEST_DIR"
if [ -d "$DEST_DIR/wham_noise/tr" ]; then
    echo "already extracted: $DEST_DIR/wham_noise"
    exit 0
fi

echo "downloading wham_noise.zip (~17GB) to $DEST_DIR ..."
curl -sL -C - -o "$ZIP" "$URL"
echo "extracting ..."
(cd "$DEST_DIR" && unzip -q wham_noise.zip)
echo "done:"
for s in tr cv tt; do
    echo "  $s: $(ls "$DEST_DIR/wham_noise/$s" | wc -l) files"
done
