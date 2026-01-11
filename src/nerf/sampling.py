import torch

def sample_z_vals(near, far, n_samples, n_rays, train=False):
    """"
    z_vals sampling
    near, far 적분해야 하는데... 그것도 샘플링해서 하자..
    """
    z_vals = torch.linspace(near, far, n_samples)
    z_vals = z_vals.expand(n_rays, n_samples)

    if train:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals