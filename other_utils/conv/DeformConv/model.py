import torch
import torch.nn as nn

from other_utils.utils import auto_pad


class DeformConv(nn.Module):
    """
    Deformable Convolution
    Split convolution process:
    1.Perform zero padding on the feature map.
    2.Calculate the deformable regions (pn + p0 + offset) where the convolution kernel participates in the convolution.
    3.Perform convolution operation weighting, generate output.

    note: For the reshape operation, my understanding is that by learning the sampling space offset, the receptive field
     of the current layer has increased. In order to fully utilize the increased receptive field, the kxk region is
     treated as a small feature map separately, and then conventional convolution is performed on it.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int,
            padding: int,
            groups: int,
            bias: bool = False
    ):
        super(DeformConv, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=auto_pad(kernel_size))
        # note: this stride of conv is kernel_size, inorder to increase receptive field.
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, kernel_size, groups=groups, bias=bias)

    def _get_pn(self, dtype: str) -> torch.Tensor:
        """
        Obtain a fixed offset set for the target pixel P0 in conventional convolution
        return dim is (2 * kernel * kernel, 1)
        """
        y, x = torch.meshgrid(
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1),
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1)
        )
        pn = torch.cat((y.flatten(), x.flatten()), dim=-1)
        pn = pn.view(1, 2 * self.kernel_size * self.kernel_size, 1, 1).type(dtype)
        return pn

    def _get_p0(self, h: int, w: int, dtype: str):
        """
        Calculate the center point coordinates of each convolution kernel
        if self.stride == 2, it means downsampling through convolution
        """
        # h * self.stride means size before down sample
        y, x = torch.meshgrid(
            torch.arange(self.kernel_size // 2, h * self.stride + 1, self.stride),
            torch.arange(self.kernel_size // 2, w * self.stride + 1, self.stride)
        )
        p0_x = x.flatten().view(1, 1, h, w).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        p0_y = y.flatten().view(1, 1, h, w).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        return torch.cat((p0_y, p0_x), dim=1).type(dtype)

    def _get_p(self, offset: torch.Tensor, dtype: str):
        """
        For each pixel on the feature map, its corresponding kxk convolution region
        has offsets in both x and y directions
        output dim is (b, 2n, h, w)
        note: offset dim is (b, 2n, h, w)

        """
        # n is kernel_size * kernel_size
        h, w = offset.size(2), offset.size(3)
        # pn dim is (1, 2n, 1, 1)
        pn = self._get_pn(dtype)
        # p0 dim is (1, 2n, h, w)
        p0 = self._get_p0(h, w, dtype)
        p = p0 + pn + offset
        return p

    def forward(self, x: torch.Tensor):
        offset = self.offset_conv(x)
        dtype = offset.type()
        n = self.kernel_size * self.kernel_size
        if self.padding:
            x = self.zero_padding(x)

        # dim is (b, 2n, h, w)
        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        # q_lt, q_rb, q_lb, q_rt only represent 4 nearest neighbor position points,
        # so, it does not need calculate generated graph
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        # construct the coordinates of four points in the neighborhood
        q_lt = torch.cat([torch.clamp(q_lt[..., :n], 0, x.size(2) - 1),
                          torch.clamp(q_lt[..., n:], 0, x.size(3) - 1)], dim=-1)
        q_rb = torch.cat([torch.clamp(q_rb[..., :n], 0, x.size(2) - 1),
                          torch.clamp(q_rb[..., n:], 0, x.size(3) - 1)], dim=-1)
        q_lb = torch.cat([q_lt[..., :n], q_rb[..., n:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :n], q_lt[..., n:]], dim=-1)

        # clip p, prevent crossing boundaries
        p = torch.cat([torch.clamp(p[..., :n], 0, x.size(2) - 1),
                       torch.clamp(p[..., n:], 0, x.size(3) - 1)], dim=-1)

        # calculate offset pixel values using bilinear interpolation
        g_lt = (1 - (p[..., :n] - q_lt[..., :n])) * (1 - (p[..., n:] - q_lt[..., n:])).long()
        g_rb = (1 - (q_rb[..., :n] - p[..., :n])) * (1 - (q_rb[..., n:] - p[..., n:])).long()
        g_lb = (1 - (p[..., :n] - q_lb[..., :n])) * (1 - (q_lb[..., n:] - p[..., n:]))
        g_rt = (1 - (q_rt[..., :n] - p[..., :n])) * (1 - (p[..., n:] - q_rt[..., n:]))

        # (b, c, h, w, N), offset pixel values
        x_q_lt = self._get_x_q(x, q_lt, n)
        x_q_rb = self._get_x_q(x, q_rb, n)
        x_q_lb = self._get_x_q(x, q_lb, n)
        x_q_rt = self._get_x_q(x, q_rt, n)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset)
        out = self.conv(x_offset)

        return out

    def _get_x_q(self, x: torch.Tensor, q: torch.Tensor, n: int):
        """
        Obtain pixel values for specified coordinates in feature maps
        """
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :n] * padded_w + q[..., n:]  # offset_x * w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1).long()

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, n)

        return x_offset

    def _reshape_x_offset(self, x_offset):
        """
        n = kernel_size * kernel_size,
        now given the pixel value of an area of kxk = n size, which obtained through deformation of convolutional
        kernel sampling space, then, the kxk region is treated as a separate feature map,
        and conventional convolution operations are applied on it to increase the receptive field.
        """
        b, c, h, w, n = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + self.kernel_size].contiguous().view(b, c, h, w * self.kernel_size)
                              for s in range(0, n, self.kernel_size)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * self.kernel_size, w * self.kernel_size)

        return x_offset


if __name__ == '__main__':
    _x = torch.rand((2, 64, 14, 14))
    conv = DeformConv(64, 64, 3, 1, 1, 1)
    res = conv(_x)
    print(res.size())
