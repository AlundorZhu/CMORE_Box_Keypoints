#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kid/CMORE_Box_Keypoints/data"
cd "$ROOT"

dirs=(
    "Box and Blocks_CONK_Affected_PT"
    "Box and Blocks_CONK_Unaffected_PT"
    "Box and Blocks_FRAP_Affected_FU"
    "Box and Blocks_FRAP_Unaffected_FU"
    "Box and Blocks_MASR_Affected_FU"
    "Box and Blocks_MASR_Unaffected_FU"
    "Box and Blocks_ZHAE_Unaffected_FU"
)

for dir in "${dirs[@]}"; do
    [ -d "$dir/labels" ] || continue
    (
        cd "$dir/labels"

        find . -maxdepth 1 -type f -name '*_frame_*.txt' -print0 \
            | sort -z -V -r \
            | xargs -0 rename 's/(frame_)([0-9]{6})/$1.sprintf("%06d",$2+1)/e'
    )
done
