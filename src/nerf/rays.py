# 1. Ray Generation
import torch

def get_rays(H, W, focal, c2w):
    """
    각 픽셀을 카메라가 보는 dirs로 바꿈
    카메라의 회전을 적용하여 world 좌표계로 바꿈
    r(t) = o + td에서 o랑 d를 구하는 역할임

    1. 픽셀 그리드 만들기
    2. 카메라 내부 좌표계로 변환
    3. 카메라 회전 적용
    4. 카메라 위치도 반환
    """
    device = c2w.device
    
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device), 
                          torch.arange(H, dtype=torch.float32, device=device), 
                          indexing='xy') # (H, W)
    
    x = (i - W * .5) / focal # (H, W, 3): x,y,z가 H*W개 있음
    y = -(j - H * .5) / focal
    z = -torch.ones_like(i)
    dirs = torch.stack([x, y, z], dim=-1)
    
    rays_d = dirs @ c2w[:3, :3].T

    # c2w: [R|t]... 모든 광선은 카메라의 위치에서 출발하겠져?
    rays_o = c2w[:3, -1]
    
    return rays_o, rays_d