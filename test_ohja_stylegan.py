import torch
import dnnlib
import legacy
from ohja.detector import OjhaDetector


NETWORK = "stylegan2-ffhq-256x256.pkl"

device = torch.device("cpu")

with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)["G_ema"]

G.eval()
G.to(device)


detector = OjhaDetector()
detector.eval()
detector.to(device)

z = torch.randn(1, G.z_dim, device=device)

c = torch.zeros(
    [1, G.c_dim],
    device=device
)

fake = G(
    z,
    c,
    truncation_psi=1.0,
    noise_mode="const",
)

fake01 = (fake + 1.0) / 2.0

print("Generated image:", fake01.shape)

score = detector(fake01)

print("Detector score:", score)