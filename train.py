import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import KITTIDataset  # Replace DataManager with KITTIDataset
from model import StereoDeepVO
import numpy as np
import yaml
import psutil
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_component_losses(pred_poses, gt_poses):
    criterion = nn.MSELoss()
    t_pred = pred_poses[:, :, :3]
    t_gt = gt_poses[:, :, :3]
    r_pred = pred_poses[:, :, 3:6]
    r_gt = gt_poses[:, :, 3:6]
    t_loss = criterion(t_pred, t_gt)
    r_loss = criterion(r_pred, r_gt)
    return t_loss.item(), r_loss.item()

def euler_to_rotmat(euler, device):
    if not isinstance(euler, torch.Tensor):
        euler = torch.tensor(euler, dtype=torch.float32, device=device)
    roll, pitch, yaw = euler
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R_x = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=euler.device)
    R_y = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=euler.device)
    R_z = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=euler.device)
    R = R_z @ R_y @ R_x
    return R.cpu()

def accumulate_poses(rel_poses, device):
    rel_poses = torch.from_numpy(rel_poses).to(device)
    abs_poses = [torch.eye(4, device=device)]
    for rel_pose in rel_poses:
        t = rel_pose[:3]
        R = rel_pose[3:].reshape(3, 3)
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        T[:3, 3] = t
        abs_poses.append(abs_poses[-1] @ T)
    return torch.stack(abs_poses).cpu().numpy()

def align_trajectories(pred_traj, gt_traj, device):
    pred_traj = torch.from_numpy(pred_traj).to(device)
    gt_traj = torch.from_numpy(gt_traj).to(device)
    
    pred_t = pred_traj[:, :3, 3]
    gt_t = gt_traj[:, :3, 3]
    
    pred_mean = torch.mean(pred_t, dim=0)
    gt_mean = torch.mean(gt_t, dim=0)
    pred_centered = pred_t - pred_mean
    gt_centered = gt_t - gt_mean
    
    H = pred_centered.T @ gt_centered
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_mean - R @ pred_mean
    
    aligned_traj = pred_traj.clone()
    aligned_traj[:, :3, 3] = (R @ pred_t.T).T + t
    aligned_traj[:, :3, :3] = torch.matmul(R, pred_traj[:, :3, :3])
    return aligned_traj.cpu().numpy()

def compute_kitti_trajectory_metrics(pred_traj, gt_traj, lengths=[100, 200, 300, 400, 500, 600, 700, 800], frames_per_100m=100, step=100):
    t_errors, r_errors = [], []
    for length in lengths:
        length_frames = int(length / 100 * frames_per_100m)
        t_err, r_err = [], []
        for i in range(0, len(pred_traj) - length_frames, step):
            j = i + length_frames
            pred_seg = pred_traj[i:j+1]
            gt_seg = gt_traj[i:j+1]
            
            t_diff = pred_seg[:, :3, 3] - gt_seg[:, :3, 3]
            t_rmse = np.sqrt(np.mean(np.linalg.norm(t_diff, axis=1)**2))
            path_length = np.sum(np.linalg.norm(gt_seg[1:, :3, 3] - gt_seg[:-1, :3, 3], axis=1))
            t_err.append(t_rmse / (path_length / 100) * 100 if path_length > 0 else 0)
            
            r_diff = []
            for p, g in zip(pred_seg[:, :3, :3], gt_seg[:, :3, :3]):
                r_rel = np.arccos(np.clip((np.trace(p.T @ g) - 1) / 2, -1, 1))
                r_diff.append(np.degrees(r_rel))
            r_err.append(np.mean(r_diff) / (path_length / 100) if path_length > 0 else 0)
        
        t_errors.append(np.mean(t_err) if t_err else 0)
        r_errors.append(np.mean(r_err) if r_err else 0)
    
    return np.mean(t_errors), np.mean(r_errors)

def train(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_dataset = KITTIDataset(config_file, mode="train", monocular=True)
    val_dataset = KITTIDataset(config_file, mode="val", monocular=True)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers']
    )
    
    model = StereoDeepVO(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=0.9, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss, train_t_loss, train_r_loss = 0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg['epochs']}")
        for left_imgs, rel_poses, rel_poses_euler in train_bar:
            left_imgs, rel_poses_euler = left_imgs.to(device), rel_poses_euler.to(device)
            optimizer.zero_grad()
            pred_poses = model(left_imgs)
            t_pred = pred_poses[:, :, :3] * 0.8  # Scale down translation predictions
            t_gt = rel_poses_euler[:, :, :3]
            loss = 2.0 * criterion(t_pred, t_gt) + 100 * criterion(pred_poses[:, :, 3:], rel_poses_euler[:, :, 3:])
            t_loss, r_loss = compute_component_losses(pred_poses, rel_poses_euler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_t_loss += t_loss
            train_r_loss += r_loss
            train_bar.set_postfix({"loss": loss.item(), "t_loss": t_loss, "r_loss": r_loss})
        
        train_loss /= len(train_loader)
        train_t_loss /= len(train_loader)
        train_r_loss /= len(train_loader)
        
        model.eval()
        val_loss, val_t_loss, val_r_loss = 0, 0, 0
        all_pred_poses, all_gt_poses = [], []
        val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{cfg['epochs']}")
        with torch.no_grad():
            for i, (left_imgs, rel_poses, rel_poses_euler) in enumerate(val_bar):
                left_imgs, rel_poses, rel_poses_euler = left_imgs.to(device), rel_poses.to(device), rel_poses_euler.to(device)
                pred_poses = model(left_imgs)
                t_pred = pred_poses[:, :, :3] * 0.8  # Scale down translation predictions
                t_gt = rel_poses_euler[:, :, :3]
                loss = 2.0 * criterion(t_pred, t_gt) + 100 * criterion(pred_poses[:, :, 3:], rel_poses_euler[:, :, 3:])
                t_loss, r_loss = compute_component_losses(pred_poses, rel_poses_euler)
                val_loss += loss.item()
                val_t_loss += t_loss
                val_r_loss += r_loss
                
                all_pred_poses.append(pred_poses.cpu().numpy())
                all_gt_poses.append(rel_poses.cpu().numpy())
                val_bar.set_postfix({"loss": loss.item(), "t_loss": t_loss, "r_loss": r_loss})
                if i == 0:
                    logger.info(f"Val Batch 0 - Pred: {pred_poses[0, 0, :]}, GT: {rel_poses_euler[0, 0, :]}")
        
        val_loss /= len(val_loader)
        val_t_loss /= len(val_loader)
        val_r_loss /= len(val_loader)
        
        all_pred_poses = np.concatenate(all_pred_poses, axis=0)
        all_gt_poses = np.concatenate(all_gt_poses, axis=0)
        logger.info(f"Epoch {epoch+1}: Number of poses - Pred: {len(all_pred_poses)}, GT: {len(all_gt_poses)}")
        
        # Log validation loss and learning rate (without W&B)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} (t={train_t_loss:.4f}, r={train_r_loss:.4f}), "
                    f"Val Loss={val_loss:.4f} (t={val_t_loss:.4f}, r={val_r_loss:.4f}), "
                    f"Learning Rate={optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
    
    # After training, compute metrics
    logger.info("Training completed. Computing final metrics...")
    pred_poses_scaled = all_pred_poses.copy()
    pred_poses_scaled[:, :, :3] *= 0.8  # Scale down translation predictions
    pred_rel_poses = np.zeros((pred_poses_scaled.shape[0], pred_poses_scaled.shape[1], 12))
    for b in range(pred_poses_scaled.shape[0]):
        for s in range(pred_poses_scaled.shape[1]):
            t = pred_poses_scaled[b, s, :3]
            euler = pred_poses_scaled[b, s, 3:]
            R = euler_to_rotmat(euler, device).numpy()
            pred_rel_poses[b, s, :3] = t
            pred_rel_poses[b, s, 3:] = R.flatten()
    
    all_pred_poses = pred_rel_poses.reshape(-1, 12)
    all_gt_poses = all_gt_poses.reshape(-1, 12)
    
    start_time = time.time()
    pred_traj = accumulate_poses(all_pred_poses, device)
    gt_traj = accumulate_poses(all_gt_poses, device)
    pred_traj = align_trajectories(pred_traj, gt_traj, device)
    t_rel, r_rel = compute_kitti_trajectory_metrics(pred_traj, gt_traj, frames_per_100m=cfg['frames_per_100m'])
    logger.info(f"Final Metric computation time = {time.time() - start_time:.2f}s, Memory usage = {psutil.virtual_memory().percent}%")
    
    logger.info(f"Final Metrics: t_rel%={t_rel:.2f}, r_rel°/100m={r_rel:.2f}")
    
    torch.save(model.state_dict(), cfg['model_save_path'])
    logger.info(f"Model saved to {cfg['model_save_path']}")

if __name__ == "__main__":
    train("config.yaml")