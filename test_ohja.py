from ohja.detector import OjhaDetector
import torch

detector = OjhaDetector()

for p in detector.parameters():
    p.requires_grad_(False)

x = torch.randn(
    1, 3, 224, 224,
    requires_grad=True
)

score = detector(x)

loss = score.mean()

loss.backward()

print(x.grad.norm())

