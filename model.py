import torch
import torch.nn as nn
import torchvision.models as models
import yaml

class StereoDeepVO(nn.Module):
    def __init__(self, config):
        super(StereoDeepVO, self).__init__()
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        self.seq_len = cfg['seq_len']
        self.hidden_size = cfg['hidden_size']
        
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Input size: 512 for monocular (single image), 1024 for stereo (concatenated)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 6)

    def forward(self, left_imgs, right_imgs=None):
        B, T, C, H, W = left_imgs.shape
        
        features = []
        for t in range(T):
            left_feat = self.cnn(left_imgs[:, t])
            left_feat = self.pool(left_feat).flatten(1)  # [B, 512]
            if right_imgs is not None:
                right_feat = self.cnn(right_imgs[:, t])
                right_feat = self.pool(right_feat).flatten(1)
                feat = torch.cat([left_feat, right_feat], dim=1)  # [B, 1024]
            else:
                feat = left_feat  # [B, 512]
            features.append(feat)
        
        features = torch.stack(features, dim=1)  # [B, T, 512 or 1024]
        
        lstm_out, _ = self.lstm(features)
        poses = self.fc(lstm_out[:, :-1])  # [B, T-1, 6]
        return poses

if __name__ == "__main__":
    model = StereoDeepVO("config.yaml")
    left_imgs = torch.randn(2, 2, 3, 224, 224)
    poses = model(left_imgs)  # Monocular test
    print(f"Monocular output shape: {poses.shape}")  # [2, 1, 6]