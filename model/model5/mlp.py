import torch
from torch import nn
import torch.autograd.profiler as profiler
from utils import util
from .code import PositionalEncoding


class MLPFeature(nn.Module):
    def __init__(
        self, d_latent, d_feature, block_num, use_encoding, pe_config, use_view
    ):
        """
        Inits MLP model.

        Args:
            d_latent: encoder latent size.
            d_feature: mlp feature size.
            use_encoding: whether use positional encoding
            pe_config: configuration for positional encoding
        """

        super().__init__()
        print("load mlp for model 5")

        self.d_latent = d_latent
        self.d_pose = 3
        self.d_feature = d_feature
        self.block_num = block_num
        self.use_view = use_view

        self.use_encoding = use_encoding
        if self.use_encoding:
            self.positional_encoding = PositionalEncoding.init_from_cfg(pe_config)
            self.d_pose = self.positional_encoding.d_out

        if self.use_view:
            self.d_pose += 3

        self.lin_in_p = nn.Linear(self.d_pose, self.d_feature)
        nn.init.constant_(self.lin_in_p.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_in_p.weight, a=0, mode="fan_in")

        self.out_layer = nn.Linear(self.d_feature, self.d_feature + 1)
        nn.init.constant_(self.out_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.out_layer.weight, a=0, mode="fan_in")

        self.blocks = nn.ModuleList()
        self.lin_in_z = nn.ModuleList()
        for i in range(self.block_num):
            lin_z = nn.Linear(self.d_latent, self.d_feature)
            nn.init.constant_(lin_z.bias, 0.0)
            nn.init.kaiming_normal_(lin_z.weight, a=0, mode="fan_in")

            self.lin_in_z.append(lin_z)

            self.blocks.append(ResnetBlock(self.d_feature))

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, x):
        if self.use_encoding:
            p = self.positional_encoding(x[..., :3])
            if self.use_view:
                p = torch.cat((p, x[..., 3:]), dim=-1)

        p = self.lin_in_p(p)
        p = self.activation(p)

        for i in range(self.block_num):
            tz = self.lin_in_z[i](z)
            tz = self.activation(tz)
            p = p + tz
            p = self.blocks[i](p)
        out = self.out_layer(p)
        out_feat = self.activation(out[..., :-1])
        out_w = self.sigmoid(out[..., -1:])

        return out_feat, out_w

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            d_latent=cfg["d_latent"],
            d_feature=cfg["d_feature"],
            use_encoding=cfg["use_encoding"],
            pe_config=cfg["positional_encoding"],
            block_num=cfg["block_num"],
            use_view=cfg["use_view"],
        )


class MLPOut(nn.Module):
    def __init__(self, d_feature, d_out, n_blocks):
        """
        Inits MLP model.

        Args:
            d_feature: feature size.
            d_out: output size.
            n_blocks: number of Resnet blocks.
        """

        super().__init__()
        self.d_feature = d_feature
        self.d_out = d_out
        self.n_blocks = n_blocks

        self.lin_out = nn.Linear(self.d_feature, self.d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(ResnetBlock(self.d_feature))

    def forward(self, x):

        for blkid in range(self.n_blocks):
            x = self.blocks[blkid](x)

        out = self.lin_out(x)
        return out

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            d_out=cfg["d_out"],
            d_feature=cfg["d_feature"],
            n_blocks=cfg["n_blocks"],
        )


class ResnetBlock(nn.Module):
    """
    Fully connected ResNet Block class.
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        """
        Inits Resnet block.

        Args:
            size_in: input dimension.
            size_out: output dimension.
            size_h: hidden dimension.
        """

        super().__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            # nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            res = self.fc_0(x)
            res = self.activation(res)
            res = self.fc_1(res)

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            out = self.activation(x_s + res)
            return out
