import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import yaml

class DataManager(Dataset):
    def __init__(self, config, mode="train", monocular=True):
        self.config = config
        self.mode = mode
        self.monocular = monocular
        self.data_dir = config['data_dir']
        self.sequences = config['sequences'][mode]
        self.img_height = config['img_height']
        self.img_width = config['img_width']
        self.seq_len = config['seq_len']
        self.left_rgb_dir = config['modalities']['left_rgb']
        self.data = self._load_data()
        self.img_mean, self.img_std = self.load_img_stats()

    def _load_data(self):
        data = []
        for seq in self.sequences:
            left_dir = os.path.join(self.data_dir, seq, self.left_rgb_dir)
            rel_pose_file = os.path.join(self.config['pose_dir'], f"{seq}_rel_poses.npy")
            
            if not os.path.exists(left_dir):
                raise FileNotFoundError(f"Left image directory not found: {left_dir}")
            if not os.path.exists(rel_pose_file):
                raise FileNotFoundError(f"Relative pose file not found: {rel_pose_file}. Run compute_rel_poses.py first.")
            
            num_frames = len(os.listdir(left_dir))
            print(f"Sequence {seq}: Found {num_frames} frames in {left_dir}")
            rel_poses = np.load(rel_pose_file)  # [num_frames-1, 12]
            print(f"Loaded precomputed relative poses from {rel_pose_file}")
            
            # Convert rotation matrices to Euler angles for loss computation
            rel_poses_euler = np.zeros((rel_poses.shape[0], 6))
            for i in range(rel_poses.shape[0]):
                rel_poses_euler[i, :3] = rel_poses[i, :3]  # t_x, t_y, t_z
                R = rel_poses[i, 3:].reshape(3, 3)
                euler = self.rotmat_to_euler(R)
                rel_poses_euler[i, 3:] = euler  # r_x, r_y, r_z
            
            for i in range(num_frames - self.seq_len + 1):
                entry = {
                    'seq': seq,
                    'left_imgs': [os.path.join(left_dir, f"{i+j:010d}.png") for j in range(self.seq_len)],
                    'rel_poses': rel_poses[i:i+self.seq_len-1],  # [seq_len-1, 12]
                    'rel_poses_euler': rel_poses_euler[i:i+self.seq_len-1]  # [seq_len-1, 6]
                }
                if not self.monocular:
                    right_dir = os.path.join(self.data_dir, seq, self.config['modalities']['right_rgb'])
                    entry['right_imgs'] = [os.path.join(right_dir, f"{i+j:010d}.png") for j in range(self.seq_len)]
                data.append(entry)
        return data

    def rotmat_to_euler(self, R):
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
        return np.array([x, y, z], dtype=np.float32)

    def load_img_stats(self):
        print('Loading precomputed image statistics...')
        norm_path = os.path.join('Norms', f"{self.config['log_project']}_train")  # Always use training stats
        mean_path = os.path.join(norm_path, 'img_mean.txt')
        std_path = os.path.join(norm_path, 'img_std.txt')
        
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(f"Image statistics not found at {norm_path}. Run compute_img_stats.py first.")
        
        mean = np.loadtxt(mean_path)
        std = np.loadtxt(std_path)
        print('Image statistics loaded.')
        return mean, std

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
            left_img = left_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            # Standardize on-the-fly
            for i in range(3):
                left_img[i, :, :] = (left_img[i, :, :] - self.img_mean[i]) / self.img_std[i]
            left_imgs.append(left_img)
        
        left_imgs = torch.from_numpy(np.stack(left_imgs)).float()
        rel_poses = torch.from_numpy(entry['rel_poses']).float()
        rel_poses_euler = torch.from_numpy(entry['rel_poses_euler']).float()

        if self.monocular:
            return left_imgs, rel_poses, rel_poses_euler
        raise NotImplementedError("Stereo mode not implemented.")