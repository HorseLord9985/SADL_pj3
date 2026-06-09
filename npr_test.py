from npr.detector import NPRDetector
import torch

npr = NPRDetector()

x = torch.randn(4,3,224,224)

y = npr(x)

print(y.shape)