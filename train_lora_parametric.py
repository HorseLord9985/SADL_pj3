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

from npr.detector import NPRDetector
import argparse

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
    default=4,
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
    "runs",
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
    "training_log.csv",
)

NETWORK = "stylegan2-ffhq-256x256.pkl"

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
DEVICE = torch.device("cpu")

STEPS = args.steps
BATCH_SIZE = args.batch_size
LR = args.lr

SAVE_EVERY = 250
CHECKPOINT_EVERY = 100

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

        acc = (
            (probs > 0.5)
            .float()
            .mean()
            .item()
        )

        save_image(
            imgs01,
            filename,
            nrow=4,
        )

    return score, acc

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
    rank=args.rank,
    alpha=args.rank,
)
G.to(DEVICE)


# ==================================================
# LOAD NPR DETECTOR
# ==================================================

print("Loading NPR detector...")

detector = NPRDetector()

detector.eval()
detector.to(DEVICE)

for p in detector.parameters():
    p.requires_grad = False


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

baseline_score, baseline_acc = save_monitor_image(
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
    f"acc={100*baseline_acc:.2f}%"
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

print("Starting training...")
print()
with open(LOG_FILE, "w") as f:

    f.write(
        "step,loss,score,lora_norm\n"
    )
best_score = float("inf")

with open(LOG_FILE, "w") as f:

    f.write(
        "step,loss,score,acc,lora_norm,checkpoint\n"
    )

    f.write(
        f"-1,"
        f"0,"
        f"{baseline_score},"
        f"{baseline_acc},"
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
    # NPR detector
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
    # Monitoring
    #

    if step % 50 == 0:

        score, acc = save_monitor_image(
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

        checkpoint_name = ""

        if step % CHECKPOINT_EVERY == 0:

            checkpoint_name = (
                f"lora_step_{step:04d}.pt"
            )

        print(
            f"step={step:04d} "
            f"loss={loss.item():.4f} "
            f"score={score:.4f} "
            f"acc={100*acc:.2f}% "
            f"lora_norm={norm:.4f}"
        )

        with open(LOG_FILE, "a") as f:

            f.write(
                f"{step},"
                f"{loss.item()},"
                f"{score},"
                f"{acc},"
                f"{norm},"
                f"{checkpoint_name}\n"
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
