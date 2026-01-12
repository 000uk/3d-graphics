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
from src.camera import get_360_poses
from src.nerf.nerf import NeRF_Model

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

    model = NeRF_Model()  라고 치고
    model_pipe = Pipeline(model, optimizer, H, W, focal)  라고 치고

    """
    NeRF 학습은 매 step마다 다른 위치(이미지, ray)에서 
    레이저를 쏴보면서 공간 전체를 스캔하는 과정임!!
    """
    # 랜덤으로 이미지 하나랑 그 중에서 n개의 광선 뽑는거임
    model_pipe.set_train()
    train_loss = []
    for _ in tqdm(range(config["n_iters"]), desc="[Train]"):
        img_i = np.random.randint(0, len(datas))
        target = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)

        loss = model_pipe.train(target, c2w)

        train_loss.append(loss)
    
    model_pipe.set_eval()
    frames = []
    poses = get_360_poses(n_frames=30, device=device)
    for c2w in tqdm(poses, desc="[Render]"):
        with torch.no_grad():    
            rgb_pred = model_pipe.render(c2w)
            
            # chunk 땜에 짤렸을 수도 있어서 원래 이미지 모양으로 합쳐
            rgb = torch.cat(rgb_pred, dim=0)
            rgb = rgb.reshape(H, W, 3)       # flatten해준거 다시 펴주고~!!
            rgb = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        frames.append(Image.fromarray(rgb))
    
    plt.plot(pipe.train_loop())
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('NeRF Training Loss')
    plt.grid(True, alpha=0.5) # 격자 추가 (보기 편함)
    plt.savefig(
        os.path.join(save_dir, f"loss_graph.jpg"),
        dpi=300
    )
    plt.close()
    
    frames = pipe.render_loop()
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