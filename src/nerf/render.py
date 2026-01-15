# 4. Volume Rendering
import torch
import torch.nn.functional as F

def volume_render(raw, z_vals):
    # color = torch.zeros(3) # RGB (0, 0, 0) 배경은 검정색
    # T = 1.0 # 투과율
    # for i in range(len(raw)):
    #     rgb = torch.sigmoid(raw[i, :3]) # rgb는 0~1 사이
    #     sigma = F.relu(raw[i, 3]) # sigma는 양수
    #     if i < num_samples - 1: delta = t_vals[i+1] - t_vals[i] # 다음 점 - 현재 점
    #     else: delta = 1e10 # 마지막 점은 뒤가 무한대라고 가정
    #     # 부피 렌더링 방정식
    #     alpha = 1 - torch.exp(-sigma * delta)
    #     weight = T * alpha
    #     color += weight * rgb
    #     T *= (1 - alpha)
    # return color
    # 이거 근데 개 느려서 벡터화함
    """
    raw: (n_rays, n_samples, 4) // r,g,b,sigma
    z_vals: (n_rays, n_samples)
    """
    T = torch.ones(z_vals.shape[0]).to(z_vals.device) # 각 ray의 첫번째 투과율은 1로 초기화

    rgb = torch.sigmoid(raw[..., :3]) # rgb는 0~1 사이
    # sigma = F.relu(raw[..., 3]) # sigma는 양수
    sigma = F.softplus(raw[..., 3]) # sigma는 양수여야 하는데 fine 학습 안되길래 이걸로 해봄

    dists = z_vals[..., 1:] - z_vals[..., :-1] # 다음 점 - 현재 점
    last = torch.ones_like(dists[..., :1]) * 1e10 # 마지막 점은 뒤가 무한대라고 가정
    dists = torch.cat([dists, last], dim=-1) # (n_rays, n_samples)

    # 부피 렌더링 방정식
    alpha = 1.0 - torch.exp(-sigma * dists) # (n_rays, n_samples)
    T = torch.cumprod(
        torch.cat([T.unsqueeze(-1), 1.-alpha + 1e-10], dim=-1), # (n_rays, n_samples + 1) 
        dim=-1 # rays는 그대로고 각 값을 누적곱 해야져
    )[:, :-1] # (n_rays, n_samples)
    weights = alpha * T
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2) # n_sample이 dt니까!! dim=-2를 sum

    return rgb_map, weights # (n_rays, 3)