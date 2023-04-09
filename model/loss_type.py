import torch
from torch import nn
import torch.nn.functional as F
from utils import util


class LogitWithUncertaintyLoss(nn.Module):
    """
    uncertainty estimation in logit space
    """

    def __init__(self, reduction="none"):
        super().__init__()
        self.rgb_loss = get_rgb_loss("mse", reduction)

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


class RGBWithUncertaintyLoss(nn.Module):
    """
    uncertainty estimation in rgb space
    """

    def __init__(self, loss_type, reduction="none"):
        super().__init__()
        self.rgb_loss = get_rgb_loss("mse", reduction)

    def forward(self, predict, ground_truth):
        rgb = predict.rgb  # (ON, RB, 3)
        beta = predict.log_var.squeeze(-1)  # (ON, RB) log variance
        rgb_diff = self.rgb_loss(rgb, ground_truth)  # (ON, RN, 3)
        rgb_diff_mean = torch.mean(rgb_diff, dim=-1)
        loss = rgb_diff_mean / torch.exp(beta) + beta
        return torch.mean(loss)


class RGBWithConfidenceLoss(nn.Module):
    """
    confidence estimation in rgb space
    """

    def __init__(self, la, reduction="none"):
        super().__init__()
        self.rgb_loss = get_rgb_loss("l1", reduction)
        self.la = la

    def forward(self, predict, ground_truth):
        rgb = predict.rgb
        q = predict.uncertainty  # .unsqueeze(-1)  # confidence map
        # print(rgb.shape, q.shape, ground_truth.shape)
        blend_rgb = rgb * q + ground_truth * (1 - q)
        rgb_diff = self.rgb_loss(ground_truth, blend_rgb)  # (ON, RN, 3)
        rgb_diff_mean = torch.mean(rgb_diff, dim=-1)
        regularization = self.la * (1 - q.squeeze(-1)) ** 2
        return torch.mean(rgb_diff_mean + regularization)


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


class ReprojectionLoss(nn.Module):
    """
    implement reprojection loss, which is calculated based on photometric loss of
    rendered images reprojected to reference coordinates
    """

    def __init__(
        self, loss_type, use_min=True, use_confidence_weight=False, reduction="none"
    ):
        super().__init__()
        self.rgb_loss = get_rgb_loss(loss_type, reduction)
        self.use_min = use_min
        self.use_confidence_weight = use_confidence_weight
        self.device = "cpu"

    def reprojection(self, pc, focal, c, ref_images):

        # with torch.autograd.detect_anomaly():

        uv = -pc[:, :, :2] / (pc[:, :, 2:] + 1.0e-8)  # (ON*RN, PB, 2)
        # print(torch.isinf(uv).any())
        uv *= util.repeat_interleave(
            focal.unsqueeze(1), self.RN if focal.shape[0] > 1 else 1
        )
        uv += util.repeat_interleave(c.unsqueeze(1), self.RN if c.shape[0] > 1 else 1)
        uv = 2 * (uv / torch.tensor([self.W, self.H]).to(self.device)) - 1

        # mask for point that out of the FoV of all reference images
        mask = util.generate_mask(uv.reshape(self.ON, self.RN, -1, 2))  # (ON, RB)
        # print(mask)
        uv = uv.unsqueeze(2)  # (ON*RN, PB, 1, 2)
        ref_rgb = F.grid_sample(ref_images.reshape(-1, 3, self.H, self.W), uv)[
            ..., 0
        ]  # (ON*RN, 3, PB)

        ref_rgb = ref_rgb.transpose(1, 2)  # (ON*RN, PB, 3)
        # print(ref_rgb)
        return ref_rgb, mask

    def get_pointcloud(self, all_rays, depth):
        pointcloud = all_rays[:, :, :3] + depth * all_rays[:, :, 3:6]  # (ON, RB, 3)

        return pointcloud

    def coordinate_transform(self, pc, ref_poses):

        ref_poses = ref_poses.reshape(-1, 4, 4)

        # generate projection matrix, w2c
        rot = ref_poses[:, :3, :3].transpose(1, 2)  # (ON*RN, 3, 3)
        trans = -torch.bmm(rot, ref_poses[:, :3, 3:])  # (ON*RN, 3, 1)
        ref_poses = torch.cat((rot, trans), dim=-1)  # (ON*RN, 3, 4)

        pc = util.repeat_interleave(pc, self.RN)  # (ON*RN, PB, 3)

        pc_rot = torch.matmul(ref_poses[:, None, :3, :3], pc.unsqueeze(-1))[
            ..., 0
        ]  # (ON*RN, PB, 3)
        pc = (
            pc_rot + ref_poses[:, None, :3, 3]
        )  # (ON*RN, PB, 3) point cloud in reference views coordinate

        return pc

    def get_photometric_loss(self, ref_rgb, rgb):
        if self.use_min:
            ref_rgb = ref_rgb.reshape(self.ON, self.RN, -1, 3)
            rgb_diff = self.rgb_loss(rgb.unsqueeze(1), ref_rgb)  # (ON, RN, RB, 3)
            rgb_diff_mean = torch.mean(rgb_diff, dim=-1)  # (ON, RN, RB)

            rgb_diff_min, _ = torch.min(rgb_diff_mean, dim=1)  # (ON, RB)
            return rgb_diff_min
        else:
            rgb = util.repeat_interleave(rgb, self.RN)
            # print(rgb[mask], ref_rgb[mask])
            return self.rgb_loss(rgb, ref_rgb)

    def forward(self, all_rays, predict, ref_images, ref_poses, focal, c, gt):
        rgb = gt  # predict.rgb
        depth = predict.depth  # (ON, RB, 1)
        # weight = predict.depth_confidence  # (ON, RB, 1)

        self.device = rgb.device
        focal = focal.clone()
        focal[..., 1] *= -1.0
        ref_images = ref_images * 0.5 + 0.5

        self.ON, self.RN, _, self.H, self.W = ref_images.shape

        pc = self.get_pointcloud(all_rays, depth)  # (ON, RB, 3) in world coordinate
        pc_in_ref = self.coordinate_transform(pc, ref_poses)
        ref_rgb, mask = self.reprojection(pc_in_ref, focal, c, ref_images)
        loss = self.get_photometric_loss(ref_rgb, rgb)
        # if self.use_confidence_weight:
        #     loss = mask * weight.squeeze(-1) * loss
        # print("reprojection loss", loss)
        # return torch.mean(loss) + 0.2 * torch.mean((1 - weight) ** 2)
        return torch.mean(mask * loss)


def get_rgb_loss(loss_type, reduction):
    if loss_type == "mse":
        return nn.MSELoss(reduction=reduction)
    elif loss_type == "l1":
        return nn.L1Loss(reduction=reduction)
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction=reduction)
