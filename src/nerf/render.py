# 4. Volume Rendering
import torch
import torch.nn.Functional as F

def render(raw, t_vals):
    color = torch.zeros(3) # RGB (0, 0, 0) 배경은 검정색
    T = 1.0 # 투과율
    for i in range(len(raw)):
        rgb = torch.sigmoid(raw[i, :3]) # rgb는 0~1 사이
        sigma = F.relu(raw[i, 3]) # sigma는 양수
    
        if i < num_samples - 1:
            delta = t_vals[i+1] - t_vals[i] # 다음 점 - 현재 점
        else:
            delta = 1e10 # 마지막 점은 뒤가 무한대라고 가정
        
        # 부피 렌더링 방정식
        alpha = 1 - torch.exp(-sigma * delta)
        weight = T * alpha
        color += weight * rgb
        T *= (1 - alpha)

    return color