import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rotmat_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

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
        self.img_mean, self.img_std = self.load_img_stats()
        # Limit validation samples for debugging
        if mode == "val":
            self.data = self.data[:1000]  # First 1000 samples

    def _load_data(self):
        data = []
        for seq in self.sequences:
            left_dir = os.path.join(self.data_dir, seq, self.left_rgb_dir)
            rel_pose_file = os.path.join(self.cfg['pose_dir'], f"{seq}_rel_poses.npy")
            
            if not os.path.exists(left_dir):
                logger.error(f"Left image directory not found: {left_dir}")
                raise FileNotFoundError(f"Left image directory not found: {left_dir}")
            if not os.path.exists(rel_pose_file):
                logger.error(f"Relative pose file not found: {rel_pose_file}. Run compute_rel_poses.py first.")
                raise FileNotFoundError(f"Relative pose file not found: {rel_pose_file}. Run compute_rel_poses.py first.")
            
            num_frames = len(os.listdir(left_dir))
            logger.info(f"Sequence {seq}: Found {num_frames} frames in {left_dir}")
            rel_poses = np.load(rel_pose_file)
            logger.info(f"Loaded precomputed relative poses from {rel_pose_file}")
            
            rel_poses_euler = np.zeros((rel_poses.shape[0], 6))
            for i in range(rel_poses.shape[0]):
                rel_poses_euler[i, :3] = rel_poses[i, :3]
                R = rel_poses[i, 3:].reshape(3, 3)
                euler = rotmat_to_euler(R)
                rel_poses_euler[i, 3:] = euler
            
            for i in range(num_frames - self.seq_len + 1):
                entry = {
                    'seq': seq,
                    'left_imgs': [os.path.join(left_dir, f"{i+j:010d}.png") for j in range(self.seq_len)],
                    'rel_poses': rel_poses[i:i+self.seq_len-1],
                    'rel_poses_euler': rel_poses_euler[i:i+self.seq_len-1]
                }
                data.append(entry)
        return data

    def load_img_stats(self):
        logger.info('Loading precomputed image statistics...')
        norm_path = os.path.join('Norms', f"{self.cfg['log_project']}_train")
        mean_path = os.path.join(norm_path, 'img_mean.txt')
        std_path = os.path.join(norm_path, 'img_std.txt')
        
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            logger.error(f"Image statistics not found at {norm_path}. Run compute_img_stats.py first.")
            raise FileNotFoundError(f"Image statistics not found at {norm_path}. Run compute_img_stats.py first.")
        
        mean = np.loadtxt(mean_path)
        std = np.loadtxt(std_path)
        logger.info('Image statistics loaded.')
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        left_imgs = []
        
        for left_path in entry['left_imgs']:
            left_img = cv2.imread(left_path)
            if left_img is None:
                logger.error(f"Failed to load image: {left_path}")
                raise ValueError(f"Failed to load image: {left_path}")
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_img, (self.img_width, self.img_height))
            left_img = left_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            # Standardize on-the-fly
            for i in range(3):
                left_img[i, :, :] = (left_img[i, :, :] - self.img_mean[i]) / self.img_std[i]
            left_imgs.append(left_img)
        
        left_imgs = torch.from_numpy(np.stack(left_imgs)).float()
        rel_poses = torch.from_numpy(entry['rel_poses']).float()
        rel_poses_euler = torch.from_numpy(entry['rel_poses_euler']).float()
        return left_imgs, rel_poses, rel_poses_euler

if __name__ == "__main__":
    dataset = KITTIDataset("config.yaml", mode="train", monocular=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for i, (left_imgs, rel_poses, rel_poses_euler) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Left images: {left_imgs.shape}")
        print(f"Relative poses (rotation matrix): {rel_poses.shape}")
        print(f"Relative poses (Euler): {rel_poses_euler.shape}")
        if i == 2:
            break