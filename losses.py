import torch
import torch.nn as nn

class VariationLoss(nn.Module):
    def __init__(self, k_size: int) -> None:
        super().__init__()
        self.k_size = k_size

    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :])**2)
        tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size])**2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss