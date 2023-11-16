import os

import torch
import torch.nn as nn
import typing as t

import yaml
from timm.models.layers import DropPath
from other_utils.conv import PartialConv
from functools import partial


class FasterNetBlock(nn.Module):

    def __init__(
            self,
            dim,
            n_div,
            mlp_ratio,
            drop_path,
            layer_scale_init_value,
            act_layer,
            norm_layer,
            pconv_fw_type,
    ):
        super(FasterNetBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        mlp_hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(*(
            nn.Conv2d(dim, mlp_hidden, 1, bias=False),
            norm_layer(mlp_hidden),
            act_layer(),
            nn.Conv2d(mlp_hidden, dim, 1, bias=False)
        ))

        self.spatial_mixing = PartialConv(dim, n_div, pconv_fw_type)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.spatial_mixing(x)
        x = identity + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.spatial_mixing(x)
        x = identity + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):
    """
    Composed of multiple FasterNetBlock stacked together
    """

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            FasterNetBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class PatchEmbedding(nn.Module):
    """
    like Swin, divide the image into patches.
    """

    def __init__(
            self,
            patch_size: int,
            patch_stride: int,
            in_chans: int,
            embed_dim: int,
            norm_layer: t.Optional[t.Callable] = None
    ):
        super(PatchEmbedding, self).__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):
    """
    like Swin, merge the patches, combining adjacent small patches into large patches,
    which is equivalent to downsampling in CNN
    """

    def __init__(
            self,
            patch_size2: int,
            patch_stride2: int,
            dim: int,
            norm_layer: t.Optional[t.Callable] = None
    ):
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(2 * dim) if norm_layer is not None else nn.Identity()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Module):
    """
    default size is 224x224
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 96,
            depths: t.Tuple = (1, 2, 8, 2),
            mlp_ratio: float = 2.,
            n_div: int = 4,
            patch_size: int = 4,
            patch_stride: int = 4,
            patch_size2: int = 2,
            patch_stride2: int = 2,
            patch_norm: bool = True,
            feature_dim: int = 1280,
            drop_path_rate: float = 0.1,
            layer_scale_init_value: int = 0,
            norm_layer: str = 'BN',
            act_layer: str = 'RELU',
            init_cfg=None,
            pretrained=None,
            pconv_fw_type='split_cat',
            **kwargs
    ):

        super(FasterNet, self).__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(
                dim=int(embed_dim * 2 ** i_stage),
                n_div=n_div,
                depth=depths[i_stage],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[: i_stage]): sum(depths[: i_stage + 1])],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            stages_list.append(stage)

            # add three PatchMerging
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(
                        patch_size2=patch_size2,
                        patch_stride2=patch_stride2,
                        dim=int(embed_dim * 2 ** i_stage),
                        norm_layer=norm_layer
                    )
                )
        self.stages = nn.Sequential(*stages_list)

        # add a norm layer for each output(FasterNetBlock)
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x: torch.Tensor) -> t.List[torch.Tensor]:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            # add norm to each FasterNet Block
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs


def fasternet_t0(weights: t.Optional[str] = None, cfg: str = 'fasternet_t0.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


def fasternet_t1(weights: t.Optional[str] = None, cfg: str = 'fasternet_t1.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


def fasternet_t2(weights: t.Optional[str] = None, cfg: str = 'fasternet_t2.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


def fasternet_s(weights: t.Optional[str] = None, cfg: str = 'fasternet_s.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


def fasternet_m(weights: t.Optional[str] = None, cfg: str = 'fasternet_m.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


def fasternet_l(weights: t.Optional[str] = None, cfg: str = 'fasternet_l.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weights = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain_weights)
    return model


if __name__ == '__main__':
    x_ = torch.randn((2, 3, 640, 640))
    model = fasternet_t0()
    # s = model.state_dict()
    for i in model(x_):
        print(i.size())
