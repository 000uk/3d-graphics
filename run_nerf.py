import yaml
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.data_loader import load_data
from src.nerf.nerf import NeRF
from src.nerf.rays import get_rays
from src.nerf.render import volume_render
from src.nerf.sampling import sample_z_vals, sample_pdf
from src.runner import run_nerf
from src.camera import get_360_poses

def main(config_path, data_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    save_dir = os.path.join("results", config["exp_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_backup.yaml"), "w") as f:
        yaml.dump(config, f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    datas = load_data(data_dir or config["data_dir"])

    """
    최종 목표는.. 
    for: loss = train()
    for: frames = render()
    이렇게 되도록 추상화하기!

    그래야 NeRF도 실험하고 3DGS도 실험하고 그러지
    """
    model_coarse = NeRF(ch=63, view=27).to(device)
    model_fine = NeRF(ch=63, view=27).to(device)
    optimizer = torch.optim.Adam([
        {'params': model_coarse.parameters()},
        {'params': model_fine.parameters()}
    ], lr=config["lr"])
    gamma = 0.1**(1/config["n_iters"]) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    model_coarse.train()
    model_fine.train()
    train_loss = []
    tqdm()
    for step in tqdm(range(config["n_iters"]), desc="[Train]"):
        """
        NeRF 학습은 매 step마다 다른 위치(이미지, ray)에서
        레이저를 쏴보면서 공간 전체를 스캔하는 과정임!!
        """
        # 랜덤으로 이미지 하나랑 그 중에서 n개의 광선 뽑는거임
        img_i = np.random.randint(0, len(datas))
        targets = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)

        # Ray Generator
        H, W, focal = datas[img_i]["hwf"]
        H, W = int(H), int(W)

        """
        아니 전체(H, W) rays로 학습하니까 메모리 터지네;;
        batch 뽑듯 몇개씩만 뽑아서 학습하자
        """
        i = torch.randint(0, W, (config["n_rays"],), device=device)
        j = torch.randint(0, H, (config["n_rays"],), device=device)
        rays_o, rays_d = get_rays(H, W, focal, c2w, i, j)
        target = targets[j, i]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # rgb = run_nerf(model, rays_o, rays_d, config["n_samples"], train=True)
        # Positional Encoding / NeRF Model / Volume Rendering
        # ---------------- [Coarse Stage] ----------------        
        z_coarse = sample_z_vals(2.0, 6.0, config["n_samples"], rays_o.shape[0], device, train=True)
        raw_coarse = run_nerf(model_coarse, rays_o, rays_d,  z_coarse)
        rgb_coarse, weights_coarse = volume_render(raw_coarse, z_coarse) # 여기서 weights가 나옴!
        # ---------------- [Fine Stage] ----------------
        if step > 500: 
            # Coarse 단계에서 나온 weights를 이용해 중요한 곳을 찍음 (Inverse Transform Sampling)
            # 각 샘플 사이의 중간값을 bins로 사용
            mids = .5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
            z_fine = sample_pdf(mids, weights_coarse[..., :-1].detach(), config["n_samples"]*2)
            z_fine = z_fine.detach() # 샘플 위치 자체는 상수로 취급
            # 순서가 섞이면 적분이 안 되니까 sort 필수
            z_vals, _ = torch.sort(torch.cat([z_coarse, z_fine], -1), dim=-1)

            raw_fine = run_nerf(model_fine, rays_o, rays_d, z_vals)
            rgb_fine, _ = volume_render(raw_fine, z_vals)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            loss_coarse = F.mse_loss(rgb_coarse, target)
            loss_fine = F.mse_loss(rgb_fine, target)
            loss = loss_coarse + loss_fine
        else:
            # Warm-up 기간: Coarse Loss만 계산
            loss = F.mse_loss(rgb_coarse, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    render_scale = 1
    H_render = H // render_scale
    W_render = W // render_scale
    focal_render = focal / render_scale

    model_coarse.eval()
    model_fine.eval()
    frames = []
    with torch.no_grad():
        poses = get_360_poses(n_frame = config["n_frame"], device=device)

        for c2w in tqdm(poses, desc="[Render]"):
            all_rgbs = []

            full_rays_o, full_rays_d = get_rays(H_render, W_render, focal_render, c2w)

            # H*W 각 픽셀을 rays로 해서 계산하면 메모리 터지겠지?
            for i in range(0, H_render*W_render, config["chunk"]):
                rays_o = full_rays_o.reshape(-1, 3)[i:i+config["chunk"]]
                rays_d = full_rays_d.reshape(-1, 3)[i:i+config["chunk"]]

                n_samples = config["n_samples"]
                n_importance = n_samples + n_samples//2
                z_coarse = sample_z_vals(2.0, 6.0, config["n_samples"], rays_o.shape[0], device, train=True)
                raw_coarse = run_nerf(model_coarse, rays_o, rays_d,  z_coarse)
                rgb_chunk, weights_coarse = volume_render(raw_coarse, z_coarse)
                mids = .5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
                z_fine = sample_pdf(mids, weights_coarse[..., 1:-1].detach(), n_importance)
                z_fine = z_fine.detach()
                z_vals, _ = torch.sort(torch.cat([z_coarse, z_fine], -1), dim=-1)
                raw_fine = run_nerf(model_fine, rays_o, rays_d, z_vals)
                rgb_chunk, w = volume_render(raw_fine, z_vals)

                all_rgbs.append(rgb_chunk)

            rgb = torch.cat(all_rgbs, dim=0) # 다시 이미지 모양으로 합치기
            rgb = rgb.reshape(H_render, W_render, 3)       # flatten해준거 다시 펴주고~!!
            rgb = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

            frames.append(Image.fromarray(rgb))

    plt.plot(train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('NeRF Training Loss')
    plt.grid(True, alpha=0.5) # 격자 추가 (보기 편함)
    plt.savefig(
        os.path.join(save_dir, f"loss_graph.jpg"),
        dpi=300
    )
    plt.close()
    
    # GIF 저장
    frames[0].save(
        os.path.join(save_dir,"result.gif"),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data", type=str, help="Optional data path")

    args = parser.parse_args()

    main(args.config, args.data)