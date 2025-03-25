import os
import torch
import numpy as np
import yaml

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

def compute_and_save_rel_poses(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    pose_dir = cfg['pose_dir']
    all_sequences = (
        cfg['sequences']['train'] +
        cfg['sequences']['val'] +
        cfg['sequences']['test']
    )
    
    for seq in all_sequences:
        pose_file = os.path.join(pose_dir, f"{seq}.txt")
        rel_pose_file = os.path.join(pose_dir, f"{seq}_rel_poses.npy")
        
        if os.path.exists(rel_pose_file):
            print(f"Relative poses already exist for {seq}: {rel_pose_file}")
            continue
        
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        
        poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
        poses = np.concatenate([poses, np.zeros((len(poses), 1, 4))], axis=1)
        poses[:, 3, 3] = 1
        poses = torch.from_numpy(poses).float()
        
        rel_poses = []
        for i in range(len(poses) - 1):
            pose_diff = torch.inverse(poses[i]) @ poses[i+1]
            t = pose_diff[:3, 3].numpy()
            R = pose_diff[:3, :3].numpy()
            rel_poses.append(np.concatenate([t, R.flatten()]))  # [t_x, t_y, t_z, R_11, ..., R_33]
        rel_poses = np.array(rel_poses)
        
        np.save(rel_pose_file, rel_poses)
        print(f"Saved relative poses for {seq} to {rel_pose_file} with shape {rel_poses.shape}")

if __name__ == "__main__":
    compute_and_save_rel_poses("config.yaml")