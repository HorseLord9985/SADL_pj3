import torch
import torch.nn as nn

from training.networks import (
    SynthesisLayer,
    modulated_conv2d,
)

from torch_utils import misc
from torch_utils.ops import bias_act


# --------------------------------------------------
# LoRA wrapper for StyleGAN SynthesisLayer
# --------------------------------------------------

class LoRASynthesisLayer(nn.Module):
    def __init__(
        self,
        original_layer: SynthesisLayer,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()

        self.layer = original_layer
        self.rank = rank
        self.alpha = alpha

        #
        # Freeze original layer
        #
        for p in self.layer.parameters():
            p.requires_grad = False

        #
        # Original conv weight
        #
        out_ch, in_ch, k, _ = self.layer.weight.shape

        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel_size = k

        #
        # View conv as matrix:
        #
        # [out_ch, in_ch * k * k]
        #
        self.in_features = in_ch * k * k
        self.out_features = out_ch

        device = original_layer.weight.device

        self.lora_A = nn.Parameter(
            torch.randn(
                rank,
                self.in_features,
                device=device
            ) * 0.01
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=device
            )
        )

    def get_delta_weight(self):

        #
        # Matrix LoRA
        #
        # [out_ch, rank]
        #     @
        # [rank, in_features]
        #
        delta = self.lora_B @ self.lora_A

        delta = delta.view(
            self.out_ch,
            self.in_ch,
            self.kernel_size,
            self.kernel_size,
        )

        scale = self.alpha / self.rank

        return delta * scale

    def forward(
        self,
        x,
        w,
        noise_mode="random",
        fused_modconv=True,
        gain=1,
    ):

        layer = self.layer

        assert noise_mode in ["random", "const", "none"]

        in_resolution = layer.resolution // layer.up

        misc.assert_shape(
            x,
            [None, layer.weight.shape[1], in_resolution, in_resolution],
        )

        #
        # Original style modulation
        #
        styles = layer.affine(w)

        #
        # Original noise
        #
        noise = None

        if layer.use_noise and noise_mode == "random":
            noise = (
                torch.randn(
                    [
                        x.shape[0],
                        1,
                        layer.resolution,
                        layer.resolution,
                    ],
                    device=x.device,
                )
                * layer.noise_strength
            )

        if layer.use_noise and noise_mode == "const":
            noise = (
                layer.noise_const
                * layer.noise_strength
            )

        #
        # LoRA update
        #
        effective_weight = (
            layer.weight
            + self.get_delta_weight().to(layer.weight.dtype)
        )

        flip_weight = (layer.up == 1)

        x = modulated_conv2d(
            x=x,
            weight=effective_weight,
            styles=styles,
            noise=noise,
            up=layer.up,
            padding=layer.padding,
            resample_filter=layer.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = layer.act_gain * gain

        act_clamp = (
            layer.conv_clamp * gain
            if layer.conv_clamp is not None
            else None
        )

        x = bias_act.bias_act(
            x,
            layer.bias.to(x.dtype),
            act=layer.activation,
            gain=act_gain,
            clamp=act_clamp,
        )

        return x


# --------------------------------------------------
# Inject LoRA into selected blocks
# --------------------------------------------------

def inject_lora(
    G,
    rank=8,
    alpha=8,
):

    targets = [
        ("b64", "conv0"),
        ("b64", "conv1"),
        ("b128", "conv0"),
        ("b128", "conv1"),
        ("b256", "conv0"),
        ("b256", "conv1"),
    ]

    for block_name, layer_name in targets:

        block = getattr(G.synthesis, block_name)

        original_layer = getattr(
            block,
            layer_name,
        )

        print(
            f"Injecting LoRA into "
            f"{block_name}.{layer_name}"
        )

        lora_layer = LoRASynthesisLayer(
            original_layer,
            rank=rank,
            alpha=alpha,
        )

        setattr(
            block,
            layer_name,
            lora_layer,
        )

    return G


# --------------------------------------------------
# Freeze everything
# --------------------------------------------------

def freeze_generator(G):

    for p in G.parameters():
        p.requires_grad = False


# --------------------------------------------------
# Collect LoRA parameters
# --------------------------------------------------

def lora_parameters(model):

    for name, param in model.named_parameters():

        if "lora_A" in name:
            yield param

        if "lora_B" in name:
            yield param


# --------------------------------------------------
# Count trainable params
# --------------------------------------------------

def print_trainable_parameters(model):

    trainable = 0
    total = 0

    for p in model.parameters():

        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

    print(
        f"Trainable params: "
        f"{trainable:,}"
    )

    print(
        f"Total params: "
        f"{total:,}"
    )

    print(
        f"Percent trainable: "
        f"{100 * trainable / total:.4f}%"
    )

# --------------------------------------------------
# Save LoRA
# --------------------------------------------------

def get_lora_state_dict(model):

    state = {}

    for name, param in model.named_parameters():

        if "lora_" in name:
            state[name] = param.detach().cpu()

    return state


# --------------------------------------------------
# Load LoRA
# --------------------------------------------------

def load_lora_state_dict(model, state_dict):

    model_state = model.state_dict()

    loaded = 0

    for name, tensor in state_dict.items():

        if name not in model_state:
            print(f"[WARNING] Missing key: {name}")
            continue

        model_state[name].copy_(tensor)

        loaded += 1

    print(f"Loaded {loaded} LoRA tensors")