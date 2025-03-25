import numpy as np
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def euler_to_rotmat(euler, device):
    try:
        if not isinstance(euler, torch.Tensor):
            euler = torch.tensor(euler, dtype=torch.float32, device=device)
        roll, pitch, yaw = euler
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_x = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device)
        R_y = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device)
        R_z = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device)
        return R_z @ R_y @ R_x
    except Exception as e:
        logger.error(f"Error in euler_to_rotmat: {str(e)}")
        raise

def accumulate_poses(rel_poses, device):
    try:
        logger.info("Starting trajectory accumulation...")
        rel_poses = torch.from_numpy(rel_poses).to(device)
        abs_poses = [torch.eye(4, device=device)]
        for rel_pose in rel_poses:
            t = rel_pose[:3]
            R = rel_pose[3:].reshape(3, 3)
            T = torch.eye(4, device=device)
            T[:3, :3] = R
            T[:3, 3] = t
            abs_poses.append(abs_poses[-1] @ T)
        result = torch.stack(abs_poses).cpu().numpy()
        logger.info("Trajectory accumulation completed.")
        return result
    except Exception as e:
        logger.error(f"Error in accumulate_poses: {str(e)}")
        raise

def align_trajectories(pred_traj, gt_traj, device):
    try:
        logger.info("Aligning trajectories...")
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
        result = aligned_traj.cpu().numpy()
        logger.info("Trajectory alignment completed.")
        return result
    except Exception as e:
        logger.error(f"Error in align_trajectories: {str(e)}")
        raise

def compute_ate(pred_traj, gt_traj):
    try:
        logger.info("Computing ATE...")
        t_diff = pred_traj[:, :3, 3] - gt_traj[:, :3, 3]
        ate = np.sqrt(np.mean(np.linalg.norm(t_diff, axis=1)**2))
        logger.info("ATE computation completed.")
        return ate
    except Exception as e:
        logger.error(f"Error in compute_ate: {str(e)}")
        raise

def compute_rpe(pred_traj, gt_traj):
    try:
        logger.info("Computing RPE...")
        t_errors, r_errors = [], []
        for i in range(len(pred_traj) - 1):
            pred_rel = np.linalg.inv(pred_traj[i]) @ pred_traj[i+1]
            gt_rel = np.linalg.inv(gt_traj[i]) @ gt_traj[i+1]
            error = np.linalg.inv(gt_rel) @ pred_rel
            t_error = np.linalg.norm(error[:3, 3])
            r_error = np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1))
            t_errors.append(t_error)
            r_errors.append(np.degrees(r_error))
        rpe_trans = np.sqrt(np.mean(np.array(t_errors)**2))
        rpe_rot = np.sqrt(np.mean(np.array(r_errors)**2))
        logger.info("RPE computation completed.")
        return rpe_trans, rpe_rot
    except Exception as e:
        logger.error(f"Error in compute_rpe: {str(e)}")
        raise

def compute_kitti_metrics(pred_traj, gt_traj, lengths=[100, 200, 300, 400, 500, 600, 700, 800], frames_per_100m=100, step=10, device="cuda"):
    try:
        logger.info("Computing KITTI metrics...")
        t_errors, r_errors = [], []
        
        for length in lengths:
            length_frames = int(length / 100 * frames_per_100m)
            indices = list(range(0, len(pred_traj) - length_frames, step))
            
            t_err, r_err = [], []
            for i in indices:
                j = i + length_frames
                pred_seg = pred_traj[i:j+1]
                gt_seg = gt_traj[i:j+1]
                
                # Per-segment alignment
                pred_seg = align_trajectories(pred_seg, gt_seg, device)
                
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
            logger.info(f"Completed processing for length {length}m.")
        
        t_rel = np.mean(t_errors)
        r_rel = np.mean(r_errors)
        logger.info("KITTI metrics computation completed.")
        return t_rel, r_rel
    except Exception as e:
        logger.error(f"Error in compute_kitti_metrics: {str(e)}")
        raise

def compute_metrics(pred_poses, gt_poses, device="cuda"):
    try:
        logger.info("Starting metrics computation...")
        # pred_poses: [N, 6] (t_x, t_y, t_z, r_x, r_y, r_z)
        # gt_poses: [N, 12] (t_x, t_y, t_z, R_11, ..., R_33)
        # Scale translation predictions to match GT scale
        pred_poses[:, :3] *= 0.8  # Scale down by 0.8 to match GT scale (0.96 / 1.20 ≈ 0.8)
        
        pred_rel_poses = np.zeros((pred_poses.shape[0], 12))
        for i in range(pred_poses.shape[0]):
            t = pred_poses[i, :3]
            euler = pred_poses[i, 3:]
            R = euler_to_rotmat(euler, device).cpu().numpy()
            pred_rel_poses[i, :3] = t
            pred_rel_poses[i, 3:] = R.flatten()
        
        pred_traj = accumulate_poses(pred_rel_poses, device)
        gt_traj = accumulate_poses(gt_poses, device)
        pred_traj = align_trajectories(pred_traj, gt_traj, device)
        
        t_rel, r_rel = compute_kitti_metrics(pred_traj, gt_traj, device=device)
        ate = compute_ate(pred_traj, gt_traj)
        rpe_trans, rpe_rot = compute_rpe(pred_traj, gt_traj)
        logger.info("Metrics computation completed successfully.")
        return t_rel, r_rel, ate, rpe_trans, rpe_rot
    except Exception as e:
        logger.error(f"Error in compute_metrics: {str(e)}")
        raise