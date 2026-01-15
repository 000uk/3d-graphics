import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, ch=60, view=24, dim=256):
        super().__init__()

        self.ch = ch
        self.view = view
        self.dim = dim
        
        self.block1 = nn.ModuleList([
            nn.Linear(ch, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        ])
        self.block2 = nn.ModuleList([
            nn.Linear(ch + dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), # 특징 벡터로 쓰려고 마지막 ReLU는 뺌
        ])
        self.sigma_head = nn.Linear(dim, 1)
        
        self.rgb_fc = nn.Linear(dim + view, 128)
        self.rgb_head = nn.Linear(128, 3)

    def forward(self, x, d): # d는 방향
        h = x
        
        for layer in self.block1:
            h = layer(h)
            
        h = torch.cat([h, x], dim=-1) # 256+60
        
        for layer in self.block2:
            h = layer(h)

        sigma = self.sigma_head(h)

        h_concat = torch.cat([h, d], dim=-1)
        feature = torch.relu(self.rgb_fc(h_concat))
        rgb = self.rgb_head(feature)

        return torch.cat([rgb, sigma], dim=-1)        