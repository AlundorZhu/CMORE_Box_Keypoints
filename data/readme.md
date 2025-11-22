# Storage
Google drive: https://drive.google.com/drive/folders/1YqL_EXgGVSgUE8StAst-dKBM6HgvL_WS?usp=drive_link 
## File structure
### For `IMG_*.MOV` files
These are the videos I shot, I made sure the box didn't move. So, There's only one `.txt` label file per video, which is used accross all frames.
You can use the following script to duplicate the labels and extract frames:

```python
import os
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
            

generate_frames_and_labels(
    vidPath='./IMG_1156.mov',
    label_txt_path='./IMG_1156.txt',
    save_frame_path='data/train/images',
    save_label_path='data/train/labels'
)
```

### For `*.tar.gz` files
These contains all the frames extracted from each video, and their corresponding label files.
```
video_name/
    frames/
        video_name_frame_000001.jpg
        video_name_frame_000002.jpg
        ...
    labels/
        video_name_frame_000001.txt
        video_name_frame_000002.txt
        ...
```
