#!/usr/bin/env bash
# Simple ACE0 run on ScanNet scene0000_00 using RGB only.
# Usage: scripts/train_scannet_scene0000_00.sh
# 환경에 맞게 gpu_id나 경로만 필요시 수정하세요.

set -euo pipefail

gpu_id=0
scene_root="/hdd3/acezero/data/scannet/scene0000_00"
rgb_glob="${scene_root}/images/*.jpg"
out_dir="out_jinhwoi"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
mkdir -p "${out_dir}"

echo "=== Running ace_zero.py on ${rgb_glob} -> ${out_dir} (GPU ${gpu_id}) ==="
pixi run python ace_zero.py \
  "${rgb_glob}" \
  "${out_dir}" \
  --render_visualization True \
  2>&1 | tee "${out_dir}/ace_zero.log"
