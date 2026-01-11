# 2. Positional Encoding
import torch

# class Encoder:
#     def __init__(self, x, y, z, l):
#         self.x = x
#         self.y = y

# 원래 클래스였는데 걍 함수로 바꿈 굳이 싶어서
def positional_encode(coords, L=10): # x,y,z 한 번에 받도록 고침
    encoded = [coords] # []

    for i in range(L):
        t = (2.0 ** i) * torch.pi * coords # 이거 매번 계산하면 느린가 근데

        # math.sin은 미분 안됨
        encoded.append(torch.sin(t))
        encoded.append(torch.cos(t))

    # MLP에 넣을 수 있게 텐서로 변환
    return torch.cat(encoded, dim=-1) # ([B], 63) // 1 + D*2L