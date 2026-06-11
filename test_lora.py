import torch
import dnnlib
import legacy

from lora_stylegan import (
    inject_lora,
    lora_parameters,
    print_trainable_parameters,
)

NETWORK = "stylegan2-ffhq-256x256.pkl"

device = torch.device("cuda")

print("Loading network...")

with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)["G_ema"].to(device)

print("Injecting LoRA...")

G = inject_lora(
    G,
    rank=8,
    alpha=8,
)

print_trainable_parameters(G)

lora_count = sum(
    p.numel()
    for p in lora_parameters(G)
)

print()
print("LoRA parameters:", f"{lora_count:,}")
print()

#
# Generate one image
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

print("Generating image...")

with torch.no_grad():

    img = G(
        z,
        label,
        truncation_psi=1.0,
        noise_mode="const",
    )

print("Output shape:", img.shape)

assert img.shape == (1, 3, 256, 256)

print()
print("SUCCESS")
print("LoRA injection works.")
