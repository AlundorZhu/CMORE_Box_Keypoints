# Storage
Google drive: https://drive.google.com/drive/folders/1YqL_EXgGVSgUE8StAst-dKBM6HgvL_WS?usp=drive_link 
## File structure
### For `IMG_*` folders

These are the videos I shot, I made sure the box didn't move. So, There's only one `.txt` label file per video, which is used accross all frames.
You can use the following script to duplicate the labels and extract frames:
- [python script](../tools/generate_frames_n_labels.py)
- [bash script](../tools/generate_frames_n_labels.sh) (faster)

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
