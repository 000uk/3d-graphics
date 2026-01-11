import torch

def render_gaussians(gaussians, w2c, focal):
    """
    World space (xyz, Σ)
    ↓ w2c
    Camera space (x, y, z, Σ')
    ↓ proj
    Image plane (x/z, y/z)
    ↓ focal, cx, cy
    Pixel coords (u, v)
    ↓ rasterization + alpha blending
    Image
    """
    xyz      = gaussians["xyz"]
    cov3d    = gaussians["cov3d"]
    rgb      = gaussians["rgb"]
    opacity  = gaussians["opacity"]
    
    device = xyz.device
    w2c = w2c.to(device) # (4, 4)

    # -----------------------------------------------------------
    # 1. world -> camera (t = W * p) // 초점거리 같은거 다 카메라 기준이잖아
    # -----------------------------------------------------------
    # 점들을 카메라 좌표계로 변환 (t = W * p)
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[...,:1])], dim=-1) # (N, 4) 동차 좌표계
    xyz_cam = (w2c @ xyz_homo.T).T
    x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2] # (N, 1)

    # -----------------------------------------------------------
    # 2. Σ' = JWΣWᵀJᵀ
    # -----------------------------------------------------------
    J = torch.zeros(xyz.shape[0], 2, 3, device=device) # 빈 행렬 만들기
    J[..., 0, 0] = focal / z
    J[..., 0, 2] = -(focal * x) / (z * z)
    J[..., 1, 1] = focal / z
    J[..., 1, 2] = -(focal * y) / (z * z)

    cov2d = (J@w2c[:3, :3])@cov3d@(J@w2c[:3, :3]).transpose(1, 2) # Σ'
    cov2d[:, 0, 0] += 0.3 # α 구할때 역행렬 구하다가 발산 안되게 0.3 더해줌 이유는 EWA
    cov2d[:, 1, 1] += 0.3

    # -----------------------------------------------------------
    # 3. Sorting
    # -----------------------------------------------------------
    indices = torch.argsort(z, descending=False) # 깊은게 뒤쪽
    cov2d = cov2d[indices]
    rgb = rgb[indices]
    opacity = opacity[indices]
    x, y, z = x[indices], y[indices], z[indices]

    # -----------------------------------------------------------
    # 4. 렌더링 (Rasterization) - C = ΣTᵢαᵢcᵢ
    # -----------------------------------------------------------
    흠.....