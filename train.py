import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from kitti_dataloader import KITTIDataset
from stereo_deepvo import StereoDeepVO
import numpy as np
import yaml

def compute_kitti_metrics(pred_poses, gt_poses):
    ate = np.mean(np.linalg.norm(pred_poses[:, :3] - gt_poses[:, :3], axis=1))
    rpe = np.mean(np.linalg.norm(pred_poses[:, 3:] - gt_poses[:, 3:], axis=1))
    return ate, rpe, ate / np.mean(np.linalg.norm(gt_poses[:, :3], axis=1)) * 100, rpe * 180 / np.pi

def train(config):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=cfg['log_project'], config=cfg)
    
    train_dataset = KITTIDataset(config, mode="train")
    val_dataset = KITTIDataset(config, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    model = StereoDeepVO(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for left_imgs, right_imgs, gt_poses in train_bar:
            left_imgs, right_imgs, gt_poses = left_imgs.to(device), right_imgs.to(device), gt_poses.to(device)
            optimizer.zero_grad()
            pred_poses = model(left_imgs, right_imgs)
            loss = criterion(pred_poses, gt_poses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss, val_ate, val_rpe, val_t_rel, val_r_rel = 0, 0, 0, 0, 0
        with torch.no_grad():
            for left_imgs, right_imgs, gt_poses in val_loader:
                left_imgs, right_imgs, gt_poses = left_imgs.to(device), right_imgs.to(device), gt_poses.to(device)
                pred_poses = model(left_imgs, right_imgs)
                loss = criterion(pred_poses, gt_poses)
                val_loss += loss.item()
                ate, rpe, t_rel, r_rel = compute_kitti_metrics(pred_poses.cpu().numpy(), gt_poses.cpu().numpy())
                val_ate += ate
                val_rpe += rpe
                val_t_rel += t_rel
                val_r_rel += r_rel
        
        val_loss /= len(val_loader)
        val_ate /= len(val_loader)
        val_rpe /= len(val_loader)
        val_t_rel /= len(val_loader)
        val_r_rel /= len(val_loader)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ATE": val_ate,
            "val_RPE": val_rpe,
            "val_t_rel_%": val_t_rel,
            "val_r_rel_deg": val_r_rel
        })
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, ATE={val_ate:.4f}")
    
    torch.save(model.state_dict(), cfg['model_save_path'])

if __name__ == "__main__":
    # Test with dummy data
    class DummyDataset:
        def __len__(self): return 4
        def __getitem__(self, idx):
            return (torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224), torch.randn(1, 6))

    dummy_loader = DataLoader(DummyDataset(), batch_size=2)
    model = StereoDeepVO("config.yaml")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for left_imgs, right_imgs, gt_poses in dummy_loader:
        pred_poses = model(left_imgs, right_imgs)
        loss = criterion(pred_poses, gt_poses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Dummy training: Loss = {loss.item()}")
        break