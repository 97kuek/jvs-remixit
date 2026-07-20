#!/usr/bin/env bash
# MERL tf-locoformer の学習済みチェックポイント(Git LFS)の実体を取得する。
# NAS(NFS)上では git clone が失敗するため、tarball 展開 + LFS batch API 直叩きで代替。
# 使い方: bash scripts/download_teacher.sh
set -euo pipefail

REPO_DIR=/mnt/kiso-qnap5/kueki/proken-A/third_party/tf-locoformer
LFS_URL=https://github.com/merlresearch/tf-locoformer.git/info/lfs/objects/batch

find "$REPO_DIR/egs2" -name "*.pth" | while read -r ptr; do
    # 既に実体化済み(LFSポインタでない)ならスキップ
    if ! head -1 "$ptr" 2>/dev/null | grep -q "git-lfs"; then
        echo "skip (already real): $ptr"
        continue
    fi
    oid=$(grep -oP '(?<=sha256:)[0-9a-f]+' "$ptr")
    size=$(grep -oP '(?<=^size )\d+' "$ptr")
    echo "fetching $ptr (oid=${oid:0:12}..., size=$size)"
    href=$(curl -s -X POST "$LFS_URL" \
        -H "Accept: application/vnd.git-lfs+json" \
        -H "Content-Type: application/vnd.git-lfs+json" \
        -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$oid\",\"size\":$size}]}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['objects'][0]['actions']['download']['href'])")
    curl -sL "$href" -o "$ptr.tmp"
    actual=$(stat -c%s "$ptr.tmp")
    if [ "$actual" != "$size" ]; then
        echo "ERROR: size mismatch for $ptr ($actual != $size)" >&2; rm -f "$ptr.tmp"; exit 1
    fi
    mv "$ptr.tmp" "$ptr"
done
echo "all checkpoints materialized:"
find "$REPO_DIR/egs2" -name "*.pth" -exec ls -lh {} \;
