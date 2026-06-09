import torch

import dnnlib
import legacy

from torchvision.utils import save_image

from lora_stylegan import (
    inject_lora,
    load_lora_state_dict,
)

# ==================================================
# CONFIG
# ==================================================

NETWORK = "stylegan2-ffhq-256x256.pkl"

LORA_CHECKPOINT = "checkpoints/lora_final.pt"

DEVICE = torch.device("cpu")

SEED = 42

# ==================================================
# FIXED LATENT
# ==================================================

torch.manual_seed(SEED)

# ==================================================
# ORIGINAL
# ==================================================

print("Loading original StyleGAN...")

with dnnlib.util.open_url(NETWORK) as f:

    G_original = legacy.load_network_pkl(
        f
    )["G_ema"]

G_original.eval()
G_original.to(DEVICE)

z = torch.randn(
    1,
    G_original.z_dim,
    device=DEVICE,
)

c = torch.zeros(
    [1, G_original.c_dim],
    device=DEVICE,
)

with torch.no_grad():

    img_original = G_original(
        z,
        c,
        truncation_psi=1.0,
        noise_mode="const",
    )

img_original = (img_original + 1.0) / 2.0

save_image(
    img_original,
    "original.png",
)

print("Saved original.png")

# ==================================================
# LORA
# ==================================================

print("Loading LoRA StyleGAN...")

with dnnlib.util.open_url(NETWORK) as f:

    G_lora = legacy.load_network_pkl(
        f
    )["G_ema"]

G_lora.eval()
G_lora.to(DEVICE)

G_lora = inject_lora(
    G_lora,
    rank=8,
    alpha=8,
)

state = torch.load(
    LORA_CHECKPOINT,
    map_location=DEVICE,
)

load_lora_state_dict(
    G_lora,
    state,
)

with torch.no_grad():

    img_lora = G_lora(
        z,
        c,
        truncation_psi=1.0,
        noise_mode="const",
    )

img_lora = (img_lora + 1.0) / 2.0

save_image(
    img_lora,
    "lora.png",
)

print("Saved lora.png")

# ==================================================
# DIFF IMAGE
# ==================================================

diff = torch.abs(
    img_original - img_lora
)

print()
print("Difference statistics")

print(
    "mean:",
    diff.mean().item()
)

print(
    "max:",
    diff.max().item()
)

#
# Raw difference
#

save_image(
    diff,
    "difference_raw.png",
)

#
# Amplified difference
#

diff_x10 = torch.clamp(
    diff * 10.0,
    0.0,
    1.0,
)

save_image(
    diff_x10,
    "difference_x10.png",
)

#
# Strong amplification
#

diff_x50 = torch.clamp(
    diff * 50.0,
    0.0,
    1.0,
)

save_image(
    diff_x50,
    "difference_x50.png",
)

print("Saved difference_raw.png")
print("Saved difference_x10.png")
print("Saved difference_x50.png")