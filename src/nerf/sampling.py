import torch

def sample_z_vals(near, far, n_samples, n_rays, device, train=False):
    """"
    z_vals sampling
    near, far 적분해야 하는데... 그것도 샘플링해서 하자..
    """
    z_vals = torch.linspace(near, far, n_samples, device=device)
    z_vals = z_vals.expand(n_rays, n_samples)

    if train:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand_like(z_vals, device=device)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals

def sample_pdf(bins, weights, n_samples, det=False):
    """
    Fine 단계: Coarse 단계에서 나온 weights를 보고, 중요한 곳을 다시 샘플링함
    (Inverse Transform Sampling 기법 사용)
    """
    # 1. Weights를 확률 분포(PDF)로 만듦
    weights = weights + 1e-5 # 0 나누기 방지
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    
    # 2. 누적 분포 함수(CDF) 계산
    cdf = torch.cumsum(pdf, -1)
    # 맨 앞에 0을 붙여서 (0 ~ 1) 범위를 만듦
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  

    # 3. 0~1 사이에서 랜덤 값 뽑기 (혹은 균일하게)
    if det:
        u = torch.linspace(0., 1., steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)

    # 4. Invert CDF (CDF에서 u에 해당하는 위치 찾기)
    # u값이 CDF의 어느 구간에 속하는지 찾아서 인덱스를 반환
    inds = torch.searchsorted(cdf, u, right=True)
    
    # 인덱스 정리 (범위 안나가게)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1) # (N_rays, N_samples, 2)

    # cdf_expanded: [N_rays, N_samples, N_bins_coarse+1]
    cdf_v = cdf.unsqueeze(1).expand(inds_g.shape[0], inds_g.shape[1], -1)
    cdf_g = torch.gather(cdf_v, 2, inds_g)
    
    # bins_v: [N_rays, N_samples, N_bins_coarse]
    bins_v = bins.unsqueeze(1).expand(inds_g.shape[0], inds_g.shape[1], -1)
    bins_inds_g = torch.clamp(inds_g, 0, bins.shape[-1] - 1) 
    bins_g = torch.gather(bins_v, 2, bins_inds_g)

    # 5. 해당 구간의 bin(z값)과 cdf값을 가져와서 선형 보간
    # (코드 길어 보이지만 그냥 비례식으로 정확한 z위치 찍는 과정임)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples