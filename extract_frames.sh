#!/bin/bash

# extract all the frames in following video files:
# Box and Blocks_CONK_Affected_PT.mov
# Box and Blocks_CONK_Unaffected_PT.mov
# Box and Blocks_FRAP_Affected_FU.mov
# Box and Blocks_FRAP_Unaffected_FU.mov
# Box and Blocks_MASR_Affected_FU.mov
# Box and Blocks_MASR_Unaffected_FU.mov
# Box and Blocks_ZHAE_Unaffected_FU.mov
# Subj_02_Left_BlackPhone.mp4
# Subj_02_Left_Ipad.mp4  
# Subj_02_Left_RedPhone.mp4
# Subj_04_Right_BlackPhone.mp4
# Subj_05_LeftHand.mov
# Subj_07_RightHand.mov
# Subj_01_LeftHand_Stabilized.mp4
# Subj_02_RightHand_Stabilized_EyesOpen.mp4
# Subj_01_Right_RedPhone.mp4

input_dir="/media/kid/HDD/UCD/Research/inputs/"
output_dir="/home/kid/CMORE_Box_Keypoints/data/"

video_files=(
    # "Box and Blocks_CONK_Affected_PT.mov"
    "Box and Blocks_CONK_Unaffected_PT.mov"
    "Box and Blocks_FRAP_Affected_FU.mov"
    "Box and Blocks_FRAP_Unaffected_FU.mov"
    "Box and Blocks_MASR_Affected_FU.mov"
    "Box and Blocks_MASR_Unaffected_FU.mov"
    "Box and Blocks_ZHAE_Unaffected_FU.mov"
    "Subj_02_Left_BlackPhone.mp4"
    "Subj_02_Left_Ipad.mp4"
    "Subj_02_Left_RedPhone.mp4"
    "Subj_04_Right_BlackPhone.mp4"
    "Subj_05_LeftHand.mov"
    "Subj_07_RightHand.mov"
    "Subj_01_LeftHand_Stabilized.mp4"
    "Subj_02_RightHand_Stabilized_EyesOpen.mp4"
    "Subj_01_Right_RedPhone.mp4"
)

# run 4 ffmpeg processes in parallel
max_parallel=4
current_jobs=0
for video_file in "${video_files[@]}"; do
    input_path="$input_dir/$video_file"
    output_subdir="${output_dir}/${video_file%.*}/images"
    mkdir -p "$output_subdir"
    
    ffmpeg -i "$input_path" "$output_subdir/frame_%06d.jpg" &
    current_jobs=$((current_jobs + 1))
    
    if [ "$current_jobs" -ge "$max_parallel" ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
done