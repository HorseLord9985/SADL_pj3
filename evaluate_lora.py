import torch
import numpy as np

import dnnlib
import legacy

from lora_stylegan import (
    inject_lora,
    load_lora_state_dict,
)

from npr.detector import NPRDetector


# ==================================================
# CONFIG
# ==================================================

NETWORK = "stylegan2-ffhq-256x256.pkl"

LORA_CHECKPOINT = "checkpoints/lora_final.pt"

DEVICE = torch.device("cpu")

NUM_IMAGES = 100

BATCH_SIZE = 10


# ==================================================
# LOAD DETECTOR
# ==================================================

print("Loading NPR detector...")

detector = NPRDetector()

detector.eval()
detector.to(DEVICE)

for p in detector.parameters():
    p.requires_grad = False


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_generator(
    G,
    detector,
    name,
):

    logits_all = []

    print()
    print(f"Evaluating: {name}")

    num_batches = NUM_IMAGES // BATCH_SIZE

    with torch.no_grad():

        for batch_idx in range(num_batches):

            z = torch.randn(
                BATCH_SIZE,
                G.z_dim,
                device=DEVICE,
            )

            c = torch.zeros(
                [BATCH_SIZE, G.c_dim],
                device=DEVICE,
            )

            imgs = G(
                z,
                c,
                truncation_psi=1.0,
                noise_mode="const",
            )

            imgs = (imgs + 1.0) / 2.0

            logits = detector(imgs)

            logits_all.extend(
                logits.flatten().cpu().numpy()
            )

    logits_all = np.array(logits_all)

    probs = 1.0 / (
        1.0 + np.exp(-logits_all)
    )

    mean_logit = logits_all.mean()
    std_logit = logits_all.std()

    mean_prob = probs.mean()

    fake_rate = (
        (probs > 0.5)
        .astype(np.float32)
        .mean()
    )

    print(
        f"Images       : {len(logits_all)}"
    )

    print(
        f"Mean logit   : {mean_logit:.4f}"
    )

    print(
        f"Std logit    : {std_logit:.4f}"
    )

    print(
        f"Mean sigmoid : {mean_prob:.6f}"
    )

    print(
        f"Fake rate    : "
        f"{100*fake_rate:.2f}%"
    )

    return {
        "mean_logit": mean_logit,
        "std_logit": std_logit,
        "mean_prob": mean_prob,
        "fake_rate": fake_rate,
    }


# ==================================================
# ORIGINAL STYLEGAN
# ==================================================

print()
print("Loading original StyleGAN...")

with dnnlib.util.open_url(NETWORK) as f:

    G_original = legacy.load_network_pkl(
        f
    )["G_ema"]

G_original.eval()
G_original.to(DEVICE)

original_results = evaluate_generator(
    G_original,
    detector,
    "Original StyleGAN",
)


# ==================================================
# LORA STYLEGAN
# ==================================================

print()
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

print(
    f"Loading checkpoint: "
    f"{LORA_CHECKPOINT}"
)

state = torch.load(
    LORA_CHECKPOINT,
    map_location=DEVICE,
)

load_lora_state_dict(
    G_lora,
    state,
)

lora_results = evaluate_generator(
    G_lora,
    detector,
    "LoRA StyleGAN",
)


# ==================================================
# COMPARISON
# ==================================================

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print()

print(
    f"Original mean logit : "
    f"{original_results['mean_logit']:.4f}"
)

print(
    f"LoRA mean logit     : "
    f"{lora_results['mean_logit']:.4f}"
)

print()

improvement = (
    original_results["mean_logit"]
    - lora_results["mean_logit"]
)

print(
    f"Logit reduction     : "
    f"{improvement:.4f}"
)

print()

print(
    f"Original fake rate  : "
    f"{100*original_results['fake_rate']:.2f}%"
)

print(
    f"LoRA fake rate      : "
    f"{100*lora_results['fake_rate']:.2f}%"
)