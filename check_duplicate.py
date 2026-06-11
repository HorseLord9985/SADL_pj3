import os
import torch

ROOT = "/home/serpentino/Desktop/Code/SADL/stylegan2-ada-pytorch/runs"

states = {}

for run in sorted(os.listdir(ROOT)):

    ckpt = os.path.join(
        ROOT,
        run,
        "checkpoints",
        "lora_best.pt"
    )

    if os.path.exists(ckpt):
        states[run] = torch.load(
            ckpt,
            map_location="cpu"
        )

runs = list(states.keys())

for i in range(len(runs)):
    for j in range(i + 1, len(runs)):

        s1 = states[runs[i]]
        s2 = states[runs[j]]

        compatible = True

        for k in s1:

            if k not in s2:
                compatible = False
                break

            if s1[k].shape != s2[k].shape:
                compatible = False
                break

        if not compatible:

            print(
                f"{runs[i]:12s} vs {runs[j]:12s} "
                f"SHAPE_MISMATCH"
            )

            continue

        diff = 0.0

        for k in s1:

            diff += (
                s1[k] - s2[k]
            ).abs().sum().item()

        print(
            f"{runs[i]:12s} vs {runs[j]:12s} "
            f"diff={diff:.3f}"
        )