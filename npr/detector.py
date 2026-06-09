import torch
from .resnet import resnet50

class NPRDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50(num_classes=1)

        state = torch.load(
            "npr/model_epoch_last_3090.pth",
            map_location="cpu"
        )

        self.model.load_state_dict(state)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)