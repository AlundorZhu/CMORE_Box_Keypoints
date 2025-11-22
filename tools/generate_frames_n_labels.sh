#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./generate_frames_and_labels.sh ./IMG_1156.mov ./IMG_1156.txt data/train/images data/train/labels

VIDEO_PATH="${1:-}"
LABEL_TXT_PATH="${2:-}"
FRAME_DIR="${3:-}"
LABEL_DIR="${4:-}"

if [[ -z "$VIDEO_PATH" || -z "$LABEL_TXT_PATH" || -z "$FRAME_DIR" || -z "$LABEL_DIR" ]]; then
  echo "Args: <video_path> <label_txt_path> <frame_output_dir> <label_output_dir>"
  exit 1
fi

if [[ ! -f "$LABEL_TXT_PATH" ]]; then
  echo "Label file not found: $LABEL_TXT_PATH"
  exit 0
fi
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video file not found: $VIDEO_PATH"
  exit 1
fi

mkdir -p "$FRAME_DIR" "$LABEL_DIR"

video_base="$(basename "$VIDEO_PATH")"
fileName="${video_base%.*}"

# Extract frames with default ffmpeg numbering (starts at 1)
ffmpeg -hide_banner -loglevel error -i "$VIDEO_PATH" -vsync 0 "${FRAME_DIR}/${fileName}_frame_%06d.jpg"

label_content="$(cat "$LABEL_TXT_PATH")"

# Enable nullglob so the for loop expands to nothing (instead of the literal pattern)
# if no matching jpg files exist, preventing unintended iteration.
shopt -s nullglob
# For every extracted frame image, create a matching label file
for frame_path in "${FRAME_DIR}/${fileName}_frame_"*.jpg; do
    # Strip .jpg extension and get the basename for consistent label filename
    frame_base="$(basename "${frame_path%.jpg}")"
    # Write the (static) label content into a per-frame .txt file
    echo "$label_content" > "${LABEL_DIR}/${frame_base}.txt"
done
# Restore default globbing behavior
shopt -u nullglob

echo "Done."