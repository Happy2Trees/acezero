#!/usr/bin/env bash
# Run ScanNet preprocessing for a single scene.
# Usage: scripts/preprocess_scannet_sample.sh [scene_id] [output_root]
# Scene ID와 출력 경로 정도만 인자로 받고, 나머지 파라미터는 아래 상수로 조정하세요.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
scene_id="${1:-scene0000_00}"
output_root="${2:-${repo_root}/data/scannet}"

# ==== 전처리 기본값 (필요하면 여기만 수정하세요) ====
frame_skip=10          # 모든 프레임 사용하려면 1 유지
max_frames="all"      # 전체 프레임 처리; 일부만 원하면 숫자로 바꾸세요 (예: 300)
resize_long_edge="640" # 리사이즈 끄려면 "none"
# ======================================================

echo "=== Preprocessing ${scene_id} -> ${output_root} ==="
echo "frame_skip=${frame_skip}, max_frames=${max_frames}, resize_long_edge=${resize_long_edge}"
cd "${repo_root}"

cmd=(
  pixi run python preprocess/preprocess_scannet.py
  --scene_id "${scene_id}"
  --output_root "${output_root}"
  --frame_skip "${frame_skip}"
  --export_depth
)

if [[ "${max_frames}" != "all" ]]; then
  cmd+=(--max_frames "${max_frames}")
fi

if [[ "${resize_long_edge}" != "none" ]]; then
  cmd+=(--resize_long_edge "${resize_long_edge}")
fi

"${cmd[@]}"

echo "Done. Output at ${output_root}/${scene_id}"
