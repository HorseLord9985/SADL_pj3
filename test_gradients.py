import torch
import dnnlib
import legacy

from lora_stylegan import (
    inject_lora,
    lora_parameters,
)

NETWORK = "stylegan2-ffhq-256x256.pkl"

device = torch.device("cuda")

print("Loading generator...")

with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)["G_ema"].to(device)

print("Injecting LoRA...")

G = inject_lora(
    G,
    rank=8,
    alpha=8,
)

#
# Dummy latent
#

z = torch.randn(
    1,
    G.z_dim,
    device=device,
)

label = torch.zeros(
    [1, G.c_dim],
    device=device,
)

#
# Forward
#

img = G(
    z,
    label,
    truncation_psi=1.0,
    noise_mode="const",
)

print("Image shape:", img.shape)

#
# Fake loss
#
# Basta una loss qualsiasi
#

loss = img.mean()

print("Loss:", loss.item())

#
# Backward
#

loss.backward()

print()
print("Checking gradients...")
print()

num_with_grad = 0

for name, param in G.named_parameters():

    if "lora_" not in name:
        continue

    if param.grad is None:
        print(f"[NO GRAD] {name}")
        continue

    grad_norm = param.grad.norm().item()

    print(
        f"[OK] {name:40s} "
        f"grad_norm={grad_norm:.6f}"
    )

    if grad_norm > 0:
        num_with_grad += 1

print()
print("LoRA tensors with gradients:", num_with_grad)

assert num_with_grad > 0

print()
print("SUCCESS")
print("Gradients reach LoRA.")
