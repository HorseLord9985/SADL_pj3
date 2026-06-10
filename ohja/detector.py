import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_models import CLIPModel

MEAN = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073]
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711]
).view(1, 3, 1, 1)

class OjhaDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = CLIPModel("ViT-L/14")

        state = torch.load(
            "ohja/pretrained_weights/fc_weights.pth",
            map_location="cpu"
        )

        self.model.fc.load_state_dict(state)


    def _preprocess(self, x):

        x = F.interpolate(
            x,
            size=(224,224),
            mode="bilinear",
            align_corners=False
        )

        mean = MEAN.to(x.device, x.dtype)
        std = STD.to(x.device, x.dtype)

        return (x - mean) / std
    
    def forward(self, x):
        x = self._preprocess(x)
        return self.model(x)

    def features(self, x):
        x = self._preprocess(x)
        return self.model(x, return_feature=True)