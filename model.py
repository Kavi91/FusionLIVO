import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights enum

class StereoDeepVO(nn.Module):
    def __init__(self, config):
        super(StereoDeepVO, self).__init__()
        # Expect config to be a dictionary, not a file path
        self.cfg = config
        self.seq_len = self.cfg['seq_len']
        self.hidden_size = self.cfg.get('hidden_size', 1000)  # Default to 1000 if not specified
        
        # Load ResNet18 with pretrained weights
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define LSTM and fully connected layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 6)

    def forward(self, left_imgs, right_imgs=None):
        B, T, C, H, W = left_imgs.shape
        
        features = []
        for t in range(T):
            left_feat = self.cnn(left_imgs[:, t])
            left_feat = self.pool(left_feat).flatten(1)
            if right_imgs is not None:
                right_feat = self.cnn(right_imgs[:, t])
                right_feat = self.pool(right_feat).flatten(1)
                feat = torch.cat([left_feat, right_feat], dim=1)
            else:
                feat = left_feat
            features.append(feat)
        
        features = torch.stack(features, dim=1)
        
        lstm_out, _ = self.lstm(features)
        poses = self.fc(lstm_out[:, :-1])
        return poses  # [batch, seq_len-1, 6]