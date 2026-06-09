import torch
import dnnlib
import legacy

from npr.detector import NPRDetector

# -------------------------
# Load StyleGAN
# -------------------------

with dnnlib.util.open_url("stylegan2-ffhq-256x256.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"]

G.eval()

# -------------------------
# Load NPR
# -------------------------

detector = NPRDetector()

detector.eval()

# -------------------------
# Generate image
# -------------------------

z = torch.randn(1, G.z_dim)

c = torch.zeros([1, G.c_dim])

with torch.no_grad():

    img = G(
        z,
        c,
        truncation_psi=1.0,
        noise_mode="const"
    )

print("StyleGAN range:")
print("min =", img.min().item())
print("max =", img.max().item())

# -------------------------
# Version A
# [-1,1]
# -------------------------

with torch.no_grad():

    score_raw = detector(img)

print()
print("Detector score on [-1,1]:")
print(score_raw)

# -------------------------
# Version B
# [0,1]
# -------------------------

img01 = (img + 1) / 2

with torch.no_grad():

    score_01 = detector(img01)

print()
print("Detector score on [0,1]:")
print(score_01)

print()
print("sigmoid(raw) =", torch.sigmoid(score_raw))
print("sigmoid(01)  =", torch.sigmoid(score_01))

print(next(detector.parameters()).device)
print(type(detector))
print(hasattr(detector, "model"))
print("StyleGAN range:")
print("min =", img.min().item())
print("max =", img.max().item())

with torch.no_grad():
    score_raw = detector(img)

img01 = (img + 1) / 2

with torch.no_grad():
    score_01 = detector(img01)

print()
print("RAW LOGIT [-1,1] =", score_raw.item())
print("SIGMOID [-1,1]   =", torch.sigmoid(score_raw).item())

print()
print("RAW LOGIT [0,1] =", score_01.item())
print("SIGMOID [0,1]   =", torch.sigmoid(score_01).item())

print()
print("=== Gradient test ===")

img_grad = img.detach().requires_grad_(True)

logit = detector(img_grad)

print("Logit:", logit.item())

logit.mean().backward()

print("Mean grad:", img_grad.grad.abs().mean().item())
print("Max  grad:", img_grad.grad.abs().max().item())