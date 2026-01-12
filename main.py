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
from src.nerf.nerf import NeRF_Model
from src.nerf.rays import get_rays
from src.nerf.sampling import sample_rays
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
    H, W, focal = datas[0]["hwf"]
    H, W = int(H), int(W)

    """
    최종 목표는.. 
    for: loss = train()
    for: frames = render()
    이렇게 되도록 추상화하기!

    그래야 NeRF도 실험하고 3DGS도 실험하고 그러지
    """
    model = NeRF_Model(ch=63, view=27).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    model.train()
    train_loss = []
    for _ in tqdm(range(config["n_iters"]), desc="[Train]"):
        """
        NeRF 학습은 매 step마다 다른 위치(이미지, ray)에서 
        레이저를 쏴보면서 공간 전체를 스캔하는 과정임!!
        """
        # 랜덤으로 이미지 하나랑 그 중에서 n개의 광선 뽑는거임
        img_i = np.random.randint(0, len(datas))
        full_target = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)

        # Ray Generator
        full_rays_o, full_rays_d = get_rays(H, W, focal, c2w)
        rays_o, rays_d, target = sample_rays(full_rays_o, full_rays_d, full_target, config["n_rays"])

        # Positional Encoding / NeRF Model / Volume Rendering
        rgb = run_nerf(model, rays_o, rays_d, config["n_samples"], train=True)
        
        loss = F.mse_loss(rgb, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

    model.eval()
    frames = []
    with torch.no_grad():
        poses = get_360_poses(n_frames=config["n_frames"], device=device)
        for c2w in tqdm(poses, desc="[Render]"):
            rgb_pred = []

            full_rays_o, full_rays_d = get_rays(H, W, focal, c2w)

            # H*W 각 픽셀을 rays로 해서 계산하면 메모리 터지겠지?
            for i in range(0, H*W, config["chunk"]):
                rays_o = full_rays_o.reshape(-1, 3)[i:i+config["chunk"]]
                rays_d = full_rays_d.reshape(-1, 3)[i:i+config["chunk"]]

                rgb_chunk = run_nerf(model, rays_o, rays_d, config["n_samples"], train=False)

                rgb_pred.append(rgb_chunk)

            # chunk 땜에 짤렸을 수도 있어서 원래 이미지 모양으로 합쳐
            rgb = torch.cat(rgb_pred, dim=0)
            rgb = rgb.reshape(H, W, 3)       # flatten해준거 다시 펴주고~!!
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