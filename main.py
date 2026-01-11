import yaml
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np

from .src.utils import set_seed
from .src.data_loader import load_data
from .src.nerf.encoder import PositionalEncoder
from .src.nerf.nerf import NeRF
from .src.nerf.rays import get_rays
from .src.nerf.render import volume_render
from .src.nerf.sampling import sample_z_vals

'''
config 에는...
exp_name:
seed: 42
base_dir: '/kaggle/input/nerf-synthetic-dataset/nerf_synthetic/lego'
lr: 5e-4
n_iters: 5000 # 학습 횟수 (100번은 너무 적어서 1000번으로 늘림)
n_samples: 256 # 적분 몇개로 나눠 할건지
'''
def main(config, data=None):
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    save_dir = os.path.join("results", config["exp_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_backup.yaml"), "w") as f:
        yaml.dump(config, f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = data if data else config["base_dir"] 
    datas = load_data(base_dir)
    H, W, focal = datas[0]["hwf"]
    H, W = int(H), int(W)

    pos_encoder = PositionalEncoder(L=10).to(device)
    dir_encoder = PositionalEncoder(L=4).to(device)
    model = NeRF(ch=63, view=27).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    train_loss = []

    for _ in range(config["iters"]):
        """
        NeRF 학습은 매 step마다 다른 위치(이미지, ray)에서 
        레이저를 쏴보면서 공간 전체를 스캔하는 과정임!!
        """
        # 랜덤으로 이미지 하나랑 그 중에서 n개의 광선 뽑는거임
        img_i = np.random.randint(0, len(datas))
        full_target = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)

        # 1. Ray Generator
        full_rays_o, full_rays_d = get_rays(H, W, focal, c2w)
        rays_o, rays_d, target = sample_rays(full_rays_o, full_rays_d, full_target, config["n_rays"])

        # 좌표 정의 // 좌표(pts) = 출발점(o) + 거리(z_vals) x 방향(d)
        near, far = 2.0, 6.0
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # (H*W, 64, 3)
        z_vals = sample_z_vals(near, far, config["n_samples"], config["n_rays"], train=True).to(device)
        dirs_expanded = rays_d[..., None, :].expand(pts.shape)

        # 2. Positional Encoding
        # MLP는 (B, D) 일케 2차원만 받으니까 flatten 해줘야함
        encoded_pts = pos_encoder(pts.reshape(-1, 3)) # (H*W*n_samples, 3)
        encoded_dirs = dir_encoder(dirs_expanded.reshape(-1, 3))

        # 3. NeRF Model
        raw = model(encoded_pts, encoded_dirs) # (H*W*n_samples, 4)
        raw = raw.reshape(config["n_rays"], config["n_samples"], 4) # (n_rays, n_samples, 4)

        # 4. Volume Rendering
        print(z_vals.shape) # H*W, n_samples
        rgb = volume_render(raw, z_vals)
        
        loss = F.mse_loss(rgb, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss.append(loss.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data", type=str, help="Optional data path")

    args = parser.parse_args()

    main(args.config, args.data)