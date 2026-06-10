# test_lora_ohja.py

import torch

from ohja.lora_ohja_stylegan import build_detector

device = torch.device("cpu")

detector = build_detector(device)

x = torch.rand(
    1,
    3,
    256,
    256,
    device=device,
)

score = detector(x)

print(score)