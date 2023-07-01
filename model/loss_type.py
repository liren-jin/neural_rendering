import torch
from torch import nn
import torch.nn.functional as F
from utils import util


class LogitWithUncertaintyLoss(nn.Module):
    """
    uncertainty estimation in logit space
    """

    def __init__(self, loss_type, reduction="none"):
        super().__init__()
        self.rgb_loss = get_rgb_loss(loss_type, reduction)

    def forward(self, predict, ground_truth):
        logit_mean = predict.logit_mean  # (ON, RB, 3) mean of RGB logit
        logit_log_var = predict.logit_log_var  # (ON, RB, 3) log variance of RGB logit
        gt = torch.clamp(ground_truth, min=1.0e-3, max=1.0 - 1.0e-3)  # (ON, RB, 3)
        logit_diff = self.rgb_loss(torch.logit(gt), logit_mean)  # (ON, RB, 3)
        gt_term = torch.log(gt * (1.0 - gt))  # (ON, RB, 3)
        loss = (
            0.5 * logit_log_var + gt_term + 0.5 * logit_diff / torch.exp(logit_log_var)
        )
        return torch.mean(loss)


class RGBLoss(nn.Module):
    """
    pure RGB photometric loss
    """

    def __init__(self, loss_type, reduction="mean"):
        super().__init__()
        self.rgb_loss = get_rgb_loss(loss_type, reduction)

    def forward(self, predict, ground_truth):
        rgb = predict.rgb
        return self.rgb_loss(rgb, ground_truth)


def get_rgb_loss(loss_type, reduction):
    if loss_type == "mse":
        return nn.MSELoss(reduction=reduction)
    elif loss_type == "l1":
        return nn.L1Loss(reduction=reduction)
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction=reduction)
