# Measure and Improve keypoint detection 

## Dataset
The videos:
```
- Box and Blocks_CONK_Affected_PT.mov
- Box and Blocks_CONK_Unaffected_PT.mov
- Box and Blocks_FRAP_Affected_FU.mov
- Box and Blocks_FRAP_Unaffected_FU.mov
- Box and Blocks_MASR_Affected_FU.mov
- Box and Blocks_MASR_Unaffected_FU.mov
- Box and Blocks_ZHAE_Unaffected_FU.mov
- Subj_02_Left_BlackPhone.mp4
- Subj_02_Left_Ipad.mp4  
- Subj_02_Left_RedPhone.mp4
- Subj_04_Right_BlackPhone.mp4
- Subj_05_LeftHand.mov
- Subj_07_RightHand.mov
- Subj_01_LeftHand_Stabilized.mp4
- Subj_02_RightHand_Stabilized_EyesOpen.mp4
- Subj_01_Right_RedPhone.mp4
- IMG_1156.mov
- IMG_2740.MOV
- IMG_2741.MOV
- IMG_2742.MOV
- IMG_2743.MOV
- IMG_2744.MOV
- IMG_2745.MOV
- IMG_2746.MOV
- IMG_2747.MOV
- IMG_2748.MOV
- IMG_2749.MOV
- IMG_2791.MOV
- IMG_2846.MOV
- IMG_2847.MOV
```

### Random test and train split
```python
>>> len(allVideos)
30
>>> allVideos
array(['Box and Blocks_CONK_Affected_PT.mov',
       'Box and Blocks_CONK_Unaffected_PT.mov',
       'Box and Blocks_FRAP_Affected_FU.mov',
       'Box and Blocks_FRAP_Unaffected_FU.mov',
       'Box and Blocks_MASR_Affected_FU.mov',
       'Box and Blocks_MASR_Unaffected_FU.mov',
       'Box and Blocks_ZHAE_Unaffected_FU.mov',
       'Subj_02_Left_BlackPhone.mp4', 'Subj_02_Left_Ipad.mp4',
       'Subj_02_Left_RedPhone.mp4', 'Subj_04_Right_BlackPhone.mp4',
       'Subj_05_LeftHand.mov', 'Subj_07_RightHand.mov',
       'Subj_01_LeftHand_Stabilized.mp4',
       'Subj_02_RightHand_Stabilized_EyesOpen.mp4',
       'Subj_01_Right_RedPhone.mp4', 'IMG_1156.mov', 'IMG_2740.MOV',
       'IMG_2741.MOV', 'IMG_2742.MOV', 'IMG_2743.MOV', 'IMG_2744.MOV',
       'IMG_2745.MOV', 'IMG_2746.MOV', 'IMG_2747.MOV', 'IMG_2748.MOV',
       'IMG_2749.MOV', 'IMG_2791.MOV', 'IMG_2846.MOV', 'IMG_2847.MOV'],
      dtype='<U41')
>>> random_indices = np.random.choice(30, size=6, replace=False)
>>> test_videos = allVideos[random_indices]
>>> test_videos
array(['Subj_01_LeftHand_Stabilized.mp4', 'IMG_2745.MOV', 'IMG_2746.MOV',
       'IMG_2791.MOV', 'IMG_2740.MOV',
       'Box and Blocks_MASR_Unaffected_FU.mov'], dtype='<U41')
>>> train_videos = np.delete(allVideos, random_indices)
>>> train_videos
array(['Box and Blocks_CONK_Affected_PT.mov',
       'Box and Blocks_CONK_Unaffected_PT.mov',
       'Box and Blocks_FRAP_Affected_FU.mov',
       'Box and Blocks_FRAP_Unaffected_FU.mov',
       'Box and Blocks_MASR_Affected_FU.mov',
       'Box and Blocks_ZHAE_Unaffected_FU.mov',
       'Subj_02_Left_BlackPhone.mp4', 'Subj_02_Left_Ipad.mp4',
       'Subj_02_Left_RedPhone.mp4', 'Subj_04_Right_BlackPhone.mp4',
       'Subj_05_LeftHand.mov', 'Subj_07_RightHand.mov',
       'Subj_02_RightHand_Stabilized_EyesOpen.mp4',
       'Subj_01_Right_RedPhone.mp4', 'IMG_1156.mov', 'IMG_2741.MOV',
       'IMG_2742.MOV', 'IMG_2743.MOV', 'IMG_2744.MOV', 'IMG_2747.MOV',
       'IMG_2748.MOV', 'IMG_2749.MOV', 'IMG_2846.MOV', 'IMG_2847.MOV'],
      dtype='<U41')
>>> 
```

### Dataset structure