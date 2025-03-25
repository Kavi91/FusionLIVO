import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from kitti_dataloader import KITTIDataset
from stereo_deepvo import StereoDeepVO
import yaml
import os

def euler_to_rotmat(euler):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix."""
    roll, pitch, yaw = euler
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    
    R_x = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x
    return R

def accumulate_poses(rel_poses):
    """Accumulate relative poses into absolute poses."""
    abs_poses = [torch.eye(4)]  # Start at identity
    for rel_pose in rel_poses:
        t = rel_pose[:3]
        r = euler_to_rotmat(rel_pose[3:])
        T = torch.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        abs_poses.append(abs_poses[-1] @ T)
    return torch.stack(abs_poses)

def test(config):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StereoDeepVO(config).to(device)  # Note: We'll use it in monocular mode
    model.load_state_dict(torch.load(cfg['model_save_path']))
    model.eval()
    
    os.makedirs(cfg['pose_save_dir'], exist_ok=True)
    os.makedirs(cfg['plot_save_dir'], exist_ok=True)
    
    for seq in cfg['sequences']['test']:
        dataset = KITTIDataset(config, mode="test", monocular=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Load ground truth absolute poses
        gt_pose_file = os.path.join(cfg['pose_dir'], f"{seq}.txt")
        gt_poses = np.loadtxt(gt_pose_file).reshape(-1, 3, 4)
        gt_poses = np.concatenate([gt_poses, np.zeros((len(gt_poses), 1, 4))], axis=1)
        gt_poses[:, 3, 3] = 1
        gt_trajectory = gt_poses[:, :3, 3]  # Extract [x, y, z]
        
        pred_rel_poses = []
        with torch.no_grad():
            for left_imgs, _ in dataloader:  # Ignore precomputed rel_poses during test
                left_imgs = left_imgs.to(device)
                pred_pose = model(left_imgs).cpu()  # [1, seq_len-1, 6]
                pred_rel_poses.append(pred_pose.squeeze(0))  # [seq_len-1, 6]
        
        pred_rel_poses = torch.cat(pred_rel_poses, dim=0)  # [num_frames-1, 6]
        pred_abs_poses = accumulate_poses(pred_rel_poses)
        pred_trajectory = pred_abs_poses[:, :3, 3].numpy()  # [num_frames, 3]
        
        # Save predicted poses
        np.savetxt(os.path.join(cfg['pose_save_dir'], f"pred_poses_seq_{seq}.txt"), pred_trajectory, fmt="%.6f")
        
        # Plot XZ plane (KITTI standard)
        plt.figure(figsize=(10, 6))
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], label="Predicted", c="blue")
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], label="Ground Truth", c="red")
        plt.legend()
        plt.title(f"Trajectory - Sequence {seq}")
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.axis("equal")  # Ensure correct scale
        plt.grid(True)
        plt.savefig(os.path.join(cfg['plot_save_dir'], f"trajectory_seq_{seq}.png"))
        plt.close()

if __name__ == "__main__":
    # Dummy test (monocular)
    model = StereoDeepVO("config.yaml")
    left_imgs = torch.randn(1, 2, 3, 224, 224)
    pred_poses = model(left_imgs).detach()
    pred_abs_poses = accumulate_poses(pred_poses.squeeze(0))
    pred_trajectory = pred_abs_poses[:, :3, 3].numpy()
    
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], label="Dummy Predicted")
    plt.legend()
    plt.savefig("dummy_trajectory.png")
    plt.close()
    print("Dummy test completed, saved to dummy_trajectory.png")