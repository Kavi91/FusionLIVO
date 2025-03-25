import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

class KITTIDataset(Dataset):
    def __init__(self, config, mode="train", monocular=True):
        with open(config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.data_dir = self.cfg['data_dir']
        self.sequences = self.cfg['sequences'][mode]
        self.img_height = self.cfg['img_height']
        self.img_width = self.cfg['img_width']
        self.seq_len = self.cfg['seq_len']
        self.left_rgb_dir = self.cfg['modalities']['left_rgb']
        self.monocular = monocular
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for seq in self.sequences:
            left_dir = os.path.join(self.data_dir, seq, self.left_rgb_dir)
            rel_pose_file = os.path.join(self.cfg['pose_dir'], f"{seq}_rel_poses.npy")
            
            if not os.path.exists(left_dir):
                raise FileNotFoundError(f"Left image directory not found: {left_dir}")
            if not os.path.exists(rel_pose_file):
                raise FileNotFoundError(f"Relative pose file not found: {rel_pose_file}. Run compute_rel_poses.py first.")
            
            num_frames = len(os.listdir(left_dir))
            print(f"Sequence {seq}: Found {num_frames} frames in {left_dir}")
            rel_poses = np.load(rel_pose_file)
            print(f"Loaded precomputed relative poses from {rel_pose_file}")
            
            for i in range(num_frames - self.seq_len + 1):
                entry = {
                    'seq': seq,
                    'left_imgs': [os.path.join(left_dir, f"{i+j:010d}.png") for j in range(self.seq_len)],
                    'rel_poses': rel_poses[i:i+self.seq_len-1]
                }
                if not self.monocular:
                    right_dir = os.path.join(self.data_dir, seq, self.cfg['modalities']['right_rgb'])
                    entry['right_imgs'] = [os.path.join(right_dir, f"{i+j:010d}.png") for j in range(self.seq_len)]
                data.append(entry)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        left_imgs = []
        
        for left_path in entry['left_imgs']:
            left_img = cv2.imread(left_path)
            if left_img is None:
                raise ValueError(f"Failed to load image: {left_path}")
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_img, (self.img_width, self.img_height))
            left_img = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
            left_imgs.append(left_img)
        
        left_imgs = torch.stack(left_imgs)
        
        if not self.monocular:
            right_imgs = []
            for right_path in entry['right_imgs']:
                right_img = cv2.imread(right_path)
                if right_img is None:
                    raise ValueError(f"Failed to load image: {right_path}")
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                right_img = cv2.resize(right_img, (self.img_width, self.img_height))
                right_img = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
                right_imgs.append(right_img)
            right_imgs = torch.stack(right_imgs)
        
        rel_poses = torch.from_numpy(entry['rel_poses']).float()

        if self.monocular:
            return left_imgs, rel_poses
        return left_imgs, right_imgs, rel_poses

def visualize_frames(left_imgs, batch_idx):
    """Visualize consecutive frames in a batch with correct colors."""
    num_frames = left_imgs.shape[1]  # seq_len
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))
    
    for i in range(num_frames):
        img = left_imgs[0, i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")
    
    plt.suptitle(f"Batch {batch_idx} - Consecutive Frames")
    plt.savefig(f"batch_{batch_idx}_frames.png")
    plt.close()

if __name__ == "__main__":
    # Test with monocular mode and visualization
    dataset = KITTIDataset("config.yaml", mode="train", monocular=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for i, (left_imgs, rel_poses) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Left images: {left_imgs.shape}")
        print(f"Relative poses: {rel_poses.shape}")
        visualize_frames(left_imgs, i)
        if i == 2:
            break
    print("Visualization saved as batch_X_frames.png")