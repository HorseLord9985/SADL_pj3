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

ROOT_RUNS = "runs_ohja"
OUTPUT_ROOT = "generated_sets_ohja"

N_IMAGES = 100

TRUNCATION_PSI = 1.0

DEVICE = torch.device("cpu")

# ==================================================
# HELPERS
# ==================================================

def infer_rank(state_dict):

    for name, tensor in state_dict.items():

        if name.endswith("lora_A"):

            return tensor.shape[0]

    raise RuntimeError(
        "Cannot infer LoRA rank"
    )

# ==================================================
# LOAD BASE STYLEGAN ONCE
# ==================================================

print("Loading StyleGAN...")

with dnnlib.util.open_url(NETWORK) as f:

    base_G = legacy.load_network_pkl(f)["G_ema"]

base_G.eval()
base_G.to(DEVICE)

# ==================================================
# FIND ALL CHECKPOINTS
# ==================================================

runs = []

for run_name in sorted(os.listdir(ROOT_RUNS)):

    ckpt = os.path.join(
        ROOT_RUNS,
        run_name,
        "checkpoints",
        "lora_best.pt",
    )

    if os.path.exists(ckpt):

        runs.append(
            (run_name, ckpt)
        )

print()

for run_name, _ in runs:
    print(run_name)

print()
print(
    f"Found {len(runs)} checkpoints"
)
print()

# ==================================================
# GENERATE
# ==================================================

for run_name, ckpt_path in runs:

    print()
    print("=" * 60)
    print(run_name)
    print("=" * 60)

    state_dict = torch.load(
        ckpt_path,
        map_location=DEVICE,
    )

    rank = infer_rank(state_dict)

    print(f"Detected rank={rank}")

    #
    # clone StyleGAN
    #

    with dnnlib.util.open_url(NETWORK) as f:

        G = legacy.load_network_pkl(f)["G_ema"]

    G.eval()
    G.to(DEVICE)

    #
    # inject correct LoRA
    #

    G = inject_lora(
        G,
        rank=rank,
        alpha=rank,
    )

    #
    # load weights
    #

    load_lora_state_dict(
        G,
        state_dict,
    )

    #
    # output folder
    #

    out_dir = os.path.join(
        OUTPUT_ROOT,
        run_name,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    #
    # generate images
    #

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

            img = (img + 1) / 2
            img = img.clamp(0, 1)

            save_image(
                img,
                os.path.join(
                    out_dir,
                    f"image_{idx:05d}.png",
                ),
            )

    print(
        f"Saved {N_IMAGES} images "
        f"to {out_dir}"
    )

print()
print("DONE")