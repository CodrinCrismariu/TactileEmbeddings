#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

status=0
.venv/bin/hf download zhenyuxie-zhzh/EgoTouch_hdf5 \
  --repo-type dataset \
  --local-dir datasets/zhenyuxie-zhzh__EgoTouch_hdf5 \
  >> download_logs/zhenyuxie-zhzh__EgoTouch_hdf5.log 2>&1 || status=$?

if (( status == 0 )); then
  raw_bytes="$(du -sb datasets | awk '{print $1}')"
  combined_bytes="$(du -sb datasets processing | awk '{sum += $1} END {print sum}')"
  printf 'DONE repo=zhenyuxie-zhzh/EgoTouch_hdf5 raw_bytes=%s combined_bytes=%s\n' \
    "$raw_bytes" "$combined_bytes" \
    | tee -a download_logs/download.log download_logs/zhenyuxie-zhzh__EgoTouch_hdf5.log
fi

# The reservation is an ephemeral budget marker owned by this worker.
rm -f datasets/.reservations/zhenyuxie-zhzh__EgoTouch_hdf5.bytes
exit "$status"
