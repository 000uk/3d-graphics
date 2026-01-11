# 2. Positional Encoding
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, x):
        # x는 B, 3 크기의 텐서임! x,y,z 한 번에 받도록 고침
        # encoded = []
        encoded = [x]
        
        for i in range(self.L):
            t = (2.0**i) * torch.pi * x # 이거 매번 계산하면 느린가 근데
            
            encoded.append(torch.sin(t)) # math.sin은 미분 안됨
            encoded.append(torch.cos(t))

        # MLP에 넣을 수 있게 텐서로 변환
        return torch.cat(encoded, dim=-1)