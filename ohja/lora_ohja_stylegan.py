# lora_ohja_stylegan.py

import torch

from ohja.detector import OjhaDetector


def build_detector(device):

    detector = OjhaDetector()

    detector.eval()
    detector.to(device)

    for p in detector.parameters():
        p.requires_grad_(False)

    return detector