import torch
import torch.nn as nn

class GaussianModel(nn.Module):    
    """
    학습 가능한 Gaussian 집합
    """
    def __init__(self, num_points=100):
        super().__init__()
        # 원래는 SfM 돌린거 point 가져와서 하겠는데.. 일단은 랜덤으로 해보자 생성형 모델 할때도 좋겠고 뭐
        self.xyz = nn.Parameter(torch.rand(num_points, 3) * 2 - 1) # 0~1 -> -1~1
        
        # Σ = (RS)(RS)^{T} 용
        self.scale = nn.Parameter(torch.rand(num_points, 3) - 3.0) # S: 스케일
        self.rot_quat = nn.Parameter(torch.rand(num_points, 4)) # R: 쿼터니언 (w, x, y, z)

        # 원래 SH 써야하는데 일단은 그냥 rgb
        self.rgb = nn.Parameter(torch.rand(num_points, 3))

        self.opacity = nn.Parameter(torch.rand(num_points, 1))

    def build_rotation(self, q): # 쿼터니언 -> 회전 행렬 (R) 변환
        q = torch.nn.functional.normalize(q) # 쿼터니언 정규화 (Unit Quaternion)
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        return torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - r*z), 2*(x*z + r*y),
            2*(x*y + r*z), 1 - 2*(x*x + z*z), 2*(y*z - r*x),
            2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x*x + y*y)
        ], dim=-1).reshape(-1, 3, 3)
    
    def forward(self):
        # Σ = (RS)(RS)^{T}
        scale = self.get_scale()
        S = torch.diag_embed(scale)  # RS 계산하려고 3x3 대각행렬 만듬
        R = self.build_rotation(self.rot_quat)
        sigma = (R@S) @ (R@S).transpose(1, 2) # 지금 전부 (n, r, l) 일케 3차원이라 .T 쓰면 안됨

        return {
            "xyz": self.xyz,
            "cov3d": sigma,
            "rgb": self.rgb,
            "opacity": self.opacity
        }