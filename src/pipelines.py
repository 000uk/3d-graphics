import torch.nn.functional as F

from nerf.volume import volume_render
from src.nerf.sampling import sample_z_vals
from src.nerf.encoder import positional_encode
from src.nerf.rays import get_rays
from src.nerf.sampling import sample_rays
from renderer import run_nerf

class Pipeline():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.evel()

    def run_model(self)

class NeRF_Pipeline(Pipeline):
    def __init__(self, model, optimizer, H, W, focal):
        super().__init__(model, optimizer, H, W, focal)
        self.model = model
        self.optimizer = optimizer
        self.H = H
        self.W = W
        self.focal = focal
    
    def train(self,c2w, target):
        # Ray Generator
        full_rays_o, full_rays_d = get_rays(self.H, self.W, self.focal, c2w)
        rays_o, rays_d, target = sample_rays(full_rays_o, full_rays_d, target, config["n_rays"])

        # Positional Encoding / NeRF Model / Volume Rendering
        rgb = run_nerf(rays_o, rays_d, config["n_samples"], train=True)

        loss = F.mse_loss(rgb, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def render(self, c2w):
        full_rays_o, full_rays_d = get_rays(self.H, self.W, self.focal, c2w)

        # H*W 각 픽셀을 rays로 해서 계산하면 메모리 터지겠지?
        rgb_pred = []
        for i in range(0, self.H*self.W, config["chunk"]):
            rays_o = full_rays_o.reshape(-1, 3)[i:i+config["chunk"]]
            rays_d = full_rays_d.reshape(-1, 3)[i:i+config["chunk"]]

            rgb_chunk = run_nerf(rays_o, rays_d, config["n_samples"], train=False)

            rgb_pred.append(rgb_chunk)

        return rgb_pred
    
    def run_model(self, rays_o, rays_d, n_samples, train=False):
        # 좌표 정의 // 좌표(pts) = 출발점(o) + 거리(z_vals) x 방향(d)
        device = rays_o.device

        near, far = 2.0, 6.0
        z_vals = sample_z_vals(near, far, n_samples, rays_o.shape[0], device, train)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # (N_rand, 64, 3)
        flat_pts = pts.reshape(-1, 3).to(device) # MLP는 (B, D) 일케 2차원만 받으니까 flatten 해줘야함
        encoded_pts = positional_encode(flat_pts, L=10) # (n_rays * n_samples, 3)
        
        # dirs_expanded = rays_d[..., None, :].expand(pts.shape)
        # flat_dirs = dirs_expanded.reshape(-1, 3).to(device)
        # encoded_dirs = positional_encode(flat_dirs, L=4)
        encoded_dirs = positional_encode(rays_d, L=4)   # (n_rays, D)
        encoded_dirs = encoded_dirs[:, None, :].expand(-1, n_samples, -1)
        encoded_dirs = encoded_dirs.reshape(-1, encoded_dirs.shape[-1]).to(device)

        raw = self.model(encoded_pts, encoded_dirs) # (n_rays * n_samples, 4)
        raw = raw.reshape(rays_o.shape[0], n_samples, 4) # (n_rays, n_samples, 4)

        rgb = volume_render(raw, z_vals)
        return rgb