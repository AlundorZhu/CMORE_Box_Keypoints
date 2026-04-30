import cv2
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import config

def load_yolo_dataset(root_path, split='train'):
    cache_path = os.path.join(root_path, f'.{split}_cache.pkl')
    if os.path.exists(cache_path):
        print(f"Loading {split} data from cache ({cache_path})...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    images_dir = os.path.join(root_path, split, 'images')
    labels_dir = os.path.join(root_path, split, 'labels')
    samples = []

    if not os.path.isdir(images_dir):
        print(f"Warning: Directory not found for {split} split: {images_dir}")
        return samples

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for img_file in tqdm(image_files, desc=f"Loading {split} data"):
        img_path = os.path.join(images_dir, img_file)
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            continue

        try:
            data = np.loadtxt(label_path)
            if data.ndim == 1:
                # Check if there are enough values for bbox and keypoints
                if data.size < 5 + config.NUM_KEYPOINTS * 3:
                    continue
                raw_kpts = data[5:].reshape(-1, 3)
            elif data.ndim == 2 and data.shape[0] > 0:
                # Take the first object in the file
                raw_kpts = data[0, 5:].reshape(-1, 3)
            else:
                continue

            # Ensure we have the correct number of keypoints
            if raw_kpts.shape[0] != config.NUM_KEYPOINTS:
                continue

            visibility = raw_kpts[:, 2]
            samples.append({
                'image_path': img_path,
                'kpts_norm': raw_kpts[:, :2],
                'visibility': visibility
            })
        except Exception as e:
            print(f"Warning: Could not load or parse label file {label_path}. Error: {e}")
            continue

    print(f"Saving {split} cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(samples, f)

    return samples

class KeypointDataset(Dataset):
  def __init__(self, data_samples, transform=None, flip_pairs=None) -> None:
    self.data = data_samples
    self.transform = transform
    self.img_size = config.IMG_SIZE
    self.flip_pairs = flip_pairs  # list of (i, j) index pairs to swap on horizontal flip

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = self.data[idx]
    image = cv2.imread(sample['image_path'])
    if image is None:
        raise FileNotFoundError(f"Could not read image: {sample['image_path']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    kpts_norm = sample['kpts_norm']
    kpts_px = kpts_norm * np.array([w, h], dtype=np.float32)
    original_visibility = sample['visibility'].copy()

    # Horizontal flip with symmetric keypoint swapping (train only)
    if self.flip_pairs is not None and np.random.random() < 0.5:
        image = np.ascontiguousarray(np.fliplr(image))
        kpts_px = kpts_px.copy()
        kpts_px[:, 0] = w - kpts_px[:, 0]
        for a, b in self.flip_pairs:
            kpts_px[[a, b]] = kpts_px[[b, a]]
            original_visibility[[a, b]] = original_visibility[[b, a]]

    if self.transform:
      transformed = self.transform(image=image, keypoints=kpts_px)
      image = transformed['image']
      keypoints = transformed['keypoints']
    else:
      keypoints = kpts_px

    keypoints = np.array(keypoints, dtype=np.float32)

    # Visibility: 0=not labeled, 1=occluded, 2=visible
    # We want a binary target: 1 if visible (original_visibility > 1), 0 otherwise.
    target_vis = (np.array(original_visibility) > 1).astype(np.float32)
    # Mark keypoints pushed outside the image by augmentation as not visible
    out_of_bounds = (
        (keypoints[:, 0] < 0) | (keypoints[:, 0] >= self.img_size) |
        (keypoints[:, 1] < 0) | (keypoints[:, 1] >= self.img_size)
    )
    target_vis[out_of_bounds] = 0.0

    # Clamp keypoints to be within image bounds after transform
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, self.img_size - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, self.img_size - 1)

    # Normalize keypoints to [0, 1] for the loss function
    keypoints[:, 0] /= self.img_size
    keypoints[:, 1] /= self.img_size

    return image, torch.tensor(keypoints, dtype=torch.float32), torch.tensor(target_vis, dtype=torch.float32)

class DataPrefetcher:
    """Wraps a DataLoader to asynchronously prefetch the next batch to GPU
    using a separate CUDA stream, keeping the GPU fed without idle gaps.
    Falls back to normal iteration on CPU."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self._use_cuda = device == 'cuda' and torch.cuda.is_available()
        self.stream = torch.cuda.Stream() if self._use_cuda else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next = next(self._iterator)
        except StopIteration:
            self._next = None
            return
        if self._use_cuda:
            with torch.cuda.stream(self.stream):
                self._next = tuple(t.to(self.device, non_blocking=True) for t in self._next)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self._use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
            for t in self._next:
                t.record_stream(torch.cuda.current_stream())
        batch = self._next
        self._preload()
        return batch

def get_train_transforms():
  return A.Compose([
      A.Resize(config.IMG_SIZE, config.IMG_SIZE),
      A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.65, 1.35), rotate=(-30, 30), p=0.8),
      A.Perspective(scale=(0.05, 0.15), p=0.5),
      A.RandomBrightnessContrast(p=0.5),
      A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
      A.GaussNoise(p=0.2),
      A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 25), hole_width_range=(8, 25), fill=0, p=0.5),
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
  ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_val_transforms():
  return A.Compose([
      A.Resize(config.IMG_SIZE, config.IMG_SIZE),
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2()
  ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
