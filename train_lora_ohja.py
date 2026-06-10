import os

import torch
import torch.optim as optim

import dnnlib
import legacy

from torchvision.utils import save_image

from lora_stylegan import (
    inject_lora,
    lora_parameters,
    get_lora_state_dict,
)

from ohja.lora_ohja_stylegan import build_detector


# ==================================================
# CONFIG
# ==================================================

NETWORK = "stylegan2-ffhq-256x256.pkl"

DEVICE = torch.device("cpu")

LOG_FILE = "training_log_ohja.csv"

STEPS = 25
BATCH_SIZE = 4

LR = 1e-4

SAVE_EVERY = 5

OUTPUT_DIR = "lora_samples_ohja"
CHECKPOINT_DIR = "checkpoints_ohja"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)


# ==================================================
# HELPERS
# ==================================================

def lora_norm(G):

    total = 0.0

    for p in lora_parameters(G):
        total += p.norm().item()

    return total


def save_monitor_image(
    G,
    detector,
    fixed_z,
    fixed_c,
    filename,
):

    with torch.no_grad():

        imgs = G(
            fixed_z,
            fixed_c,
            truncation_psi=1.0,
            noise_mode="const",
        )

        imgs01 = (imgs + 1.0) / 2.0

        logits = detector(imgs01)

        score = logits.mean().item()

        probs = torch.sigmoid(logits)

        save_image(
            imgs01,
            filename,
            nrow=4,
        )

    return score, probs.mean().item()


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
# LOAD OHJA
# ==================================================

print("Loading Ojha detector...")

detector = build_detector(DEVICE)


# ==================================================
# TRAINABLE PARAMS
# ==================================================

num_trainable = sum(
    p.numel()
    for p in lora_parameters(G)
)

print()
print("Trainable LoRA params:", f"{num_trainable:,}")
print()


# ==================================================
# FIXED LATENTS
# ==================================================

fixed_z = torch.randn(
    16,
    G.z_dim,
    device=DEVICE,
)

fixed_c = torch.zeros(
    [16, G.c_dim],
    device=DEVICE,
)


# ==================================================
# BASELINE
# ==================================================

baseline_score, baseline_prob = save_monitor_image(
    G,
    detector,
    fixed_z,
    fixed_c,
    os.path.join(
        OUTPUT_DIR,
        "baseline.png",
    ),
)

print(
    f"Baseline score={baseline_score:.4f} "
    f"prob={baseline_prob:.4f}"
)

print()


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = optim.Adam(
    lora_parameters(G),
    lr=LR,
)


# ==================================================
# TRAINING
# ==================================================

best_score = float("inf")

with open(LOG_FILE, "w") as f:

    f.write(
        "step,loss,score,prob,lora_norm,checkpoint\n"
    )

for step in range(STEPS):

    optimizer.zero_grad()

    z = torch.randn(
        BATCH_SIZE,
        G.z_dim,
        device=DEVICE,
    )

    c = torch.zeros(
        [BATCH_SIZE, G.c_dim],
        device=DEVICE,
    )

    fake = G(
        z,
        c,
        truncation_psi=1.0,
        noise_mode="const",
    )

    fake01 = (fake + 1.0) / 2.0

    logits = detector(fake01)

    #
    # Same objective used in NPR
    #

    loss = logits.mean()

    loss.backward()

    optimizer.step()

    if step % SAVE_EVERY == 0:

        score, prob = save_monitor_image(
            G,
            detector,
            fixed_z,
            fixed_c,
            os.path.join(
                OUTPUT_DIR,
                f"step_{step:04d}.png",
            ),
        )

        norm = lora_norm(G)

        print(
            f"step={step:04d} "
            f"loss={loss.item():.4f} "
            f"score={score:.4f} "
            f"prob={prob:.4f} "
            f"lora_norm={norm:.4f}"
        )

        checkpoint_name = ""

        if step % 50 == 0:

            checkpoint_name = (
                f"lora_step_{step:04d}.pt"
            )

        with open(LOG_FILE, "a") as f:

            f.write(
                f"{step},"
                f"{loss.item()},"
                f"{score},"
                f"{prob},"
                f"{norm},"
                f"{checkpoint_name}\n"
            )

        if step % 50 == 0:

            ckpt_path = os.path.join(
                CHECKPOINT_DIR,
                f"lora_step_{step:04d}.pt",
            )

            torch.save(
                get_lora_state_dict(G),
                ckpt_path,
            )

            print(
                f"Checkpoint saved: {ckpt_path}"
            )

        if score < best_score:

            best_score = score

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "lora_best.pt",
            )

            torch.save(
                get_lora_state_dict(G),
                best_path,
            )

            print(
                f"New best model! "
                f"score={score:.4f}"
            )

print()
print("Training finished.")

final_path = os.path.join(
    CHECKPOINT_DIR,
    "lora_final.pt",
)

torch.save(
    get_lora_state_dict(G),
    final_path,
)

print(
    f"Final checkpoint saved: "
    f"{final_path}"
)

print(
    f"Best detector score: "
    f"{best_score:.4f}"
)