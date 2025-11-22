import os
import sys
import cv2 as cv

def generate_frames_and_labels(vidPath, label_txt_path, save_frame_path, save_label_path):
    if not os.path.exists(label_txt_path):
        return

    os.makedirs(save_frame_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
                   
    fileName = vidPath.split('/')[-1]
    fileName = fileName.split('.')[0]
    
    cap = cv.VideoCapture(vidPath)
    with open(label_txt_path, 'r') as f:
            label_data = f.read()
    while cap.isOpened():
        frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(save_frame_path, f"{fileName}_frame_{int(frame_idx):06d}.jpg")
        cv.imwrite(frame_path, frame)
                                                                                                                        
        label_path = os.path.join(save_label_path, f"{fileName}_frame_{int(frame_idx):06d}.txt")
        with open(label_path, 'w') as f:
            f.write(label_data)
            

# use sys.argv to get command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_frames_n_labels.py <video_path> <label_txt_path> <save_frame_path> <save_label_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    label_txt_path = sys.argv[2]
    save_frame_path = sys.argv[3]
    save_label_path = sys.argv[4]
    generate_frames_and_labels(video_path, label_txt_path, save_frame_path, save_label_path)