from .encoder import Encoder
from .mlp import MLPFeature, MLPOut
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from utils import util


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.make_encoder(cfg["encoder"])
        self.make_mlp(cfg["mlp"])

    def make_encoder(self, cfg):
        self.encoder = Encoder.init_from_cfg(cfg)
        self.stop_encoder_grad = False

    def make_mlp(self, cfg):
        self.mlp_feature = MLPFeature.init_from_cfg(cfg["mlp_feature"])
        self.mlp_out = MLPOut.init_from_cfg(cfg["mlp_output"])

    def encode(self, images, poses, focal, c):
        """
        Encode feature map from reference images. Must be called before forward method.
        """
        with profiler.record_function("encode_reference_images"):
            self.num_objs = images.size(0)
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)  # Be consistent with RN
            self.num_ref_views = images.size(1)

            images = images.reshape(-1, *images.shape[2:])  # (ON*RN, 3, H, W)
            poses = poses.reshape(-1, 4, 4)

            # generate projection matrix, w2c
            rot = poses[:, :3, :3].transpose(1, 2)  # (ON*RN, 3, 3)
            trans = -torch.bmm(rot, poses[:, :3, 3:])  # (ON*RN, 3, 1)
            poses = torch.cat((rot, trans), dim=-1)  # (ON*RN, 3, 4)

            latent = self.encoder(images)  # (ON*RN, d_latent, H, W)

            self.focal = focal
            self.c = c

            self.ref_image = images
            self.ref_pose = poses
            self.ref_latent = latent

            self.image_shape = torch.empty(2, dtype=torch.float32)
            self.latent_scaling = torch.empty(2, dtype=torch.float32)

            self.image_shape[0] = images.shape[-1]
            self.image_shape[1] = images.shape[-2]
            self.latent_scaling[0] = latent.shape[-1]
            self.latent_scaling[1] = latent.shape[-2]
            self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1)

    def get_features(self, xyz, viewdirs):
        """
        Get encoded features from reference images

        Args:
            xyz: (ON, RB, 3) [x, y, z] in world coordinate, RB-ray batch
            viewdirs: (ON, RB, 3) [r_x, r_y, r_z]

        Returns:
            latent: extracted reference image featues (ON*RN*RB, d_latent)
            p_feature: extracted point pose features in reference coordinates (ON*RN*RB, 6)
        """

        with profiler.record_function("extract_features"):
            RN = self.num_ref_views
            device = xyz.device

            # project query points to camera coordinates
            xyz = util.repeat_interleave(xyz, RN)  # (ON*RN, RB, 3)
            xyz_rot = torch.matmul(self.ref_pose[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]  # (ON*RN, RB, 3)
            xyz = xyz_rot + self.ref_pose[:, None, :3, 3]  # (ON*RN, RB, 3)
            xyz_ref = xyz.reshape(-1, 3)  # (ON*RN*RB, 3)
            # p_feature = self.positional_encoding(p_feature)

            # viewdirs to camera coordinates
            viewdirs = util.repeat_interleave(viewdirs, RN)  # (ON*RN, RB, 3, 1)
            viewdirs = torch.matmul(
                self.ref_pose[:, None, :3, :3], viewdirs.unsqueeze(-1)
            )  # (ON*RN, RB, 3, 1)
            viewdirs_ref = viewdirs.reshape(-1, 3)  # (ON*RN*RB, 3)

            p_feature = torch.cat((xyz_ref, viewdirs_ref), dim=-1)

            uv = xyz[:, :, :2] / xyz[:, :, 2:]  # (ON*RN, RB, 2)
            uv *= util.repeat_interleave(
                self.focal.unsqueeze(1), RN if self.focal.shape[0] > 1 else 1
            )
            uv += util.repeat_interleave(
                self.c.unsqueeze(1), RN if self.c.shape[0] > 1 else 1
            )

            # make the interval compatible with grid_sample, [-1, 1]
            scale = (self.latent_scaling / self.image_shape).to(device)
            uv_feat = 2 * uv * scale - 1.0
            uv_feat = uv_feat.unsqueeze(2)  # (ON*RN, RB, 1, 2)

            latent = F.grid_sample(
                self.ref_latent,
                uv_feat,
                align_corners=True,
                mode=self.encoder.index_interp,
                padding_mode=self.encoder.index_padding,
            )  # (ON*RN, d_latent, RB, 1)
            latent = latent[:, :, :, 0]  # (ON*RN, d_latent, RB)

            if self.stop_encoder_grad:
                latent = latent.detach()

            latent = latent.transpose(1, 2).reshape(
                -1, self.encoder.latent_size
            )  # (ON*RN*RB, d_latent)

        return (
            latent,
            p_feature,
        )  # (ON*RN*RB, d_latent), (ON*RN*RB, 6)

    def forward(self, xyz, viewdirs):
        """
        Get model final prediction given surface point position and view directions.

        Args:
            xyz: (ON, RB, 3) [x, y, z] in world coordinate, RB-ray batch
            viewdirs: (ON, RB, 3) [r_x, r_y, r_z]

        Returns:
            output: (ON, RB, d_out)
        """

        with profiler.record_function("model_inference"):
            ON, RB, _ = xyz.shape
            latent, p_feature = self.get_features(xyz, viewdirs)  # (ON*RN*RB, d_latent)

            feature, weight = self.mlp_feature(
                latent,
                p_feature,
            )  # (ON*RN*RB, d_feature)
            feature = util.weighted_pooling(
                feature, inner_dims=(self.num_ref_views, RB), weight=weight
            ).reshape(
                ON * RB, -1
            )  # (ON*RB, 2*d_feature)
            final_output = self.mlp_out(feature).reshape(ON, RB, -1)  # (ON, RB, d_out)
        return final_output  # rgb logit mean and log variance
