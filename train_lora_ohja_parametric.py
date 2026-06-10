import os
import argparse

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-name",
    type=str,
    required=True,
)

parser.add_argument(
    "--steps",
    type=int,
    default=500,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "--rank",
    type=int,
    default=8,
)

args = parser.parse_args()

RUN_DIR = os.path.join(
    "runs_ohja",
    args.run_name,
)

OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "samples",
)

CHECKPOINT_DIR = os.path.join(
    RUN_DIR,
    "checkpoints",
)

LOG_FILE = os.path.join(
    RUN_DIR,
    "training_log_ohja.csv",
)

NETWORK = "stylegan2-ffhq-256x256.pkl"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

STEPS = args.steps
BATCH_SIZE = args.batch_size
LR = args.lr

SAVE_IMAGES_EVERY = 100
CHECKPOINT_EVERY = 50
PRINT_EVERY = 10
LOG_EVERY = 10

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
    rank=args.rank,
    alpha=args.rank,
)

G.to(DEVICE)


# ==================================================
# LOAD OHJA DETECTOR
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

    f.write(
        f"-1,"
        f"0,"
        f"{baseline_score},"
        f"{baseline_prob},"
        f"0,"
        f"baseline\n"
    )


print("\n===== CONFIG =====")
print(f"run      : {args.run_name}")
print(f"device   : {DEVICE}")
print(f"steps    : {STEPS}")
print(f"batch    : {BATCH_SIZE}")
print(f"lr       : {LR}")
print(f"rank     : {args.rank}")
print("==================\n")


for step in range(STEPS):

    optimizer.zero_grad()

    #
    # Random latent batch
    #

    z = torch.randn(
        BATCH_SIZE,
        G.z_dim,
        device=DEVICE,
    )

    c = torch.zeros(
        [BATCH_SIZE, G.c_dim],
        device=DEVICE,
    )

    #
    # Generate
    #

    fake = G(
        z,
        c,
        truncation_psi=1.0,
        noise_mode="const",
    )

    #
    # Convert to [0,1]
    #

    fake01 = (fake + 1.0) / 2.0

    #
    # Ojha detector
    #

    logits = detector(fake01)

    #
    # Goal:
    # make detector less confident
    #

    loss = logits.mean()

    loss.backward()

    optimizer.step()

    #
    # Checkpoint
    #

    if step % CHECKPOINT_EVERY == 0:

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

    #
    # Metrics
    #

    score = None
    prob = None
    norm = None

    if (
        step % PRINT_EVERY == 0
        or step % LOG_EVERY == 0
        or step % SAVE_IMAGES_EVERY == 0
    ):

        monitor_path = os.path.join(
            OUTPUT_DIR,
            f"step_{step:04d}.png",
        )

        score, prob = save_monitor_image(
            G,
            detector,
            fixed_z,
            fixed_c,
            monitor_path,
        )

        norm = lora_norm(G)

    #
    # Checkpoint
    #

    if step % CHECKPOINT_EVERY == 0:

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

    #
    # Print
    #

    if step % PRINT_EVERY == 0:

        print(
            f"step={step:04d} "
            f"loss={loss.item():.4f} "
            f"score={score:.4f} "
            f"prob={prob:.4f} "
            f"lora_norm={norm:.4f}"
        )

    #
    # CSV log
    #

    if step % LOG_EVERY == 0:

        checkpoint_name = ""

        if step % CHECKPOINT_EVERY == 0:

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

    #
    # Best model
    #

    if (
        score is not None
        and score < best_score
    ):

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