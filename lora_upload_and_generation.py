import os

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

LORA_CHECKPOINT = "checkpoints/lora_best.pt"

OUTPUT_DIR = "generated_images"

N_IMAGES = 100

TRUNCATION_PSI = 1.0

DEVICE = torch.device("cpu")

# ==================================================
# OUTPUT DIR
# ==================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

# ==================================================
# LOAD STYLEGAN
# ==================================================

print("Loading StyleGAN...")

with dnnlib.util.open_url(NETWORK) as f:

    G = legacy.load_network_pkl(f)["G_ema"]

G.eval()
G.to(DEVICE)

# ==================================================
# INJECT LORA
# ==================================================

print("Injecting LoRA...")

G = inject_lora(
    G,
    rank=8,
    alpha=8,
)

# ==================================================
# LOAD LORA
# ==================================================

print("Loading LoRA checkpoint...")

state_dict = torch.load(
    LORA_CHECKPOINT,
    map_location=DEVICE,
)

load_lora_state_dict(
    G,
    state_dict,
)

print("LoRA loaded successfully.")

# ==================================================
# GENERATE IMAGES
# ==================================================

print(f"Generating {N_IMAGES} images...")

with torch.no_grad():

    for idx in range(N_IMAGES):

        z = torch.randn(
            1,
            G.z_dim,
            device=DEVICE,
        )

        c = torch.zeros(
            [1, G.c_dim],
            device=DEVICE,
        )

        img = G(
            z,
            c,
            truncation_psi=TRUNCATION_PSI,
            noise_mode="const",
        )

        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)

        filename = os.path.join(
            OUTPUT_DIR,
            f"image_{idx:05d}.png",
        )

        save_image(
            img,
            filename,
        )

        if (idx + 1) % 10 == 0:

            print(
                f"Generated {idx + 1}/{N_IMAGES}"
            )

print()
print("Generation completed.")
print(f"Images saved to: {OUTPUT_DIR}")