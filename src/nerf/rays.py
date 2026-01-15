# 1. Ray Generation
import torch

def get_rays(H, W, focal, c2w):
    """
    각 픽셀을 카메라가 보는 dirs로 바꿈
    카메라의 회전을 적용하여 world 좌표계로 바꿈

    1. 픽셀 그리드 만들기
    2. 카메라 내부 좌표계로 변환
    3. 카메라 회전 적용
    4. 카메라 위치도 반환
    """
    device = c2w.device

    if i is None and j is None:
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),
                              torch.arange(H, dtype=torch.float32, device=device),
                              indexing='xy') # (H, W)

    x = (i - W * .5) / focal
    y = -(j - H * .5) / focal
    z = -torch.ones_like(i) # nerf는 앞이 -z임
    dirs = torch.stack([x, y, z], dim=-1) # (H, W, 3)

    rays_d = dirs @ c2w[:3, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # c2w: [R|t]... 모든 광선은 카메라의 위치에서 출발하겠져?
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    # rays_d랑 크기 맞게 expand해줌

    return rays_o, rays_d