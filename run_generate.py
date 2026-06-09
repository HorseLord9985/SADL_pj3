import torch
import dnnlib
import legacy

network_pkl = "stylegan2-ffhq-256x256.pkl"

device = torch.device("cpu")

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)["G_ema"].to(device)

print("Loaded generator")
print(G.z_dim)