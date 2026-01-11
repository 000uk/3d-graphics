import torch

def sample_rays(rays_o, rays_d, target, n_rays):
    """
    아니 전체(H, W) rays로 학습하니까 메모리 터지네;;
    batch 뽑듯 몇개씩만 뽑아서 학습하자
    """
    H, W = rays_o.shape[:2]
    i = torch.randint(0, H, (n_rays,))
    j = torch.randint(0, W, (n_rays,))
    return rays_o[i, j], rays_d[i, j], target[i, j]

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