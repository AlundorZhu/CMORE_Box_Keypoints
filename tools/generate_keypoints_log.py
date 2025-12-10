from keypoint_detector import BoxDetector
import sys
import cv2
import os

# Check argument count
if len(sys.argv) != 4:
    print("Error: Expected 3 arguments")
    print("Usage: python generate_keypoints_log.py <model_path> <videos_path> <csv_output_path>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
VIDEOS_PATH = sys.argv[2]
CSV_OUTPUT_FOLDER = sys.argv[3]

# Validate model path exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

# Validate video path exists
if not os.path.exists(VIDEOS_PATH):
    print(f"Error: Video file not found at {VIDEOS_PATH}")
    sys.exit(1)

# Validate output directory is writable
output_dir = os.path.dirname(CSV_OUTPUT_FOLDER)
if output_dir and not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        sys.exit(1)

boxDetector = BoxDetector(MODEL_PATH)

boxDetector.start_logging(CSV_OUTPUT_FOLDER + "/" + VIDEOS_PATH.split('/')[-1].split('.')[0] + '_box_keypoints.csv')

# Detect keypoints in every 10 frames of the video
cap = cv2.VideoCapture(VIDEOS_PATH)
frame_count = 0

# Get total frames for progress reporting
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Starting keypoint detection on video with {total_frames} total frames")

while cap.isOpened():

    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    ret, frame = cap.read()

    # get timestamp
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    if not ret:
        break

    ok, box_detection = boxDetector.detect(frame)
    if ok:
        boxDetector.append(box_detection, current_time, frame_count)

    # Print progress every 100 frames
    if frame_count % 100 == 0:
        progress_pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"Processing frame {frame_count}/{total_frames} ({progress_pct:.1f}%)")

    frame_count += 10

cap.release()
boxDetector.close_log()
print("Keypoints log saved to", CSV_OUTPUT_FOLDER)