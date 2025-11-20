#!/usr/bin/env bash
# Simple ACE0 run on TexturePoorSfM_dataset/1000 with default params.
# Usage: scripts/train_texturepoor_1000.sh
# 환경에 맞게 gpu_id, 경로만 필요시 수정하세요.

set -euo pipefail

gpu_id=7              # 사용할 GPU ID
rgb_glob="/hdd3/acezero/data/TexturePoorSfM_dataset/1000/10bag*/images/*.png"
out_dir="out_jinhwoi"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
mkdir -p "${out_dir}"

echo "=== Running ace_zero.py on ${rgb_glob} -> ${out_dir} (GPU ${gpu_id}) ==="
pixi run python ace_zero.py \
  "${rgb_glob}" \
  "${out_dir}" \
  --render_visualization True \
  2>&1 | tee "${out_dir}/ace_zero.log"

