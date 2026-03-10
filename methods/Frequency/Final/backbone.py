# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.fft


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# --- Linear module ---
class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = False

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
                self.beta)).expand_as(out)
            out = gamma * out + beta
        return out


# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- Simple Conv Block ---
class ConvBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = FeatureWiseTransformation2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

ratio_low=0.5
ratio_high=1.0
def get_high_mask(H, W):
    global ratio_low, ratio_high
    # Create frequency mask once
    yy, xx = torch.meshgrid(torch.arange(H // 2 + 1), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2 + 1, W // 2 + 1
    dist = ((yy - center_y) ** 2 + (xx - center_x) ** 2).float()
    threshold_low = min(H * ratio_low, W * ratio_low)
    threshold_high = min(H * ratio_high, W * ratio_high)
    mask = (dist >= threshold_low).float() *  (dist <= threshold_high).float()# high freq = 1, low freq = 0
    print('frequency:',ratio_low,' - ',ratio_high)
    print(mask.sum()/H/W)
    return mask

Size  = {64:56,128:28,256:14,512:7}

ablate_highfreqconv = None
class HighFreqConv(nn.Module):
    def __init__(self, in_dim, out_dim, H, W, half_res):
        super(HighFreqConv, self).__init__()
        self.H, self.W = H, W
        if half_res:
            mask = get_high_mask(H*2, W*2)
            self.register_buffer('high_mask1', mask[None, None, :, :])  # shape: (1, 1, H//2+1, W)
        else:
            mask = get_high_mask(H, W)
            self.register_buffer('high_mask1', mask[None, None, :, :])  # shape: (1, 1, H//2+1, W)
        mask = get_high_mask(H, W)
        self.register_buffer('high_mask2', mask[None, None, :, :])  # shape: (1, 1, H//2+1, W)

        # Shared convolution for real and imag
        if half_res:
            self.C1 = nn.Conv2d(in_dim * (2 if ablate_highfreqconv==None else 1), out_dim, 3, stride=2, padding=1)
        else:
            self.C1 = nn.Conv2d(in_dim * (2 if ablate_highfreqconv==None else 1), out_dim, 1, stride=1)
        self.BN1 = nn.BatchNorm2d(out_dim)

        self.C2 = nn.Conv2d(out_dim, out_dim * (2 if ablate_highfreqconv==None else 1), 1)

        self.BN2 = nn.BatchNorm2d(out_dim)
        # Activation
        self.relu1 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        for layer in self.parametrized_layers:
            init_layer(layer)

    def _build_symmetric(self, fft_half, H, W):
        """
        将上半部分 (1, C, H_half, W) 扩展为复共轭对称 (1, C, H, W)
        支持偶数和奇数 H
        """
        # 上下翻转
        if H % 2 == 0:
            # 偶数 H，filt_half 包含 H//2 + 1 行（含中心）
            lower = torch.flip(fft_half[:, :, 1: - 1, :], dims=[-2])  # 去除第0行和中心行，对剩下的做镜像
        else:
            # 奇数 H，filt_half 通常就是 (H+1)//2，最后一行是中心行
            lower = torch.flip(fft_half[:, :, : -1, :], dims=[-2])  # 去掉中心行再翻转，对剩下的做镜像

        # 左右翻转
        if W % 2 == 0:
            lower[:, :, :, 1:] = torch.flip(lower[:, :, :, 1:], dims=[-1])
        else:
            lower = torch.flip(lower, dims=[-1])

        # 复共轭对称：取共轭
        lower = torch.conj(lower)

        # 拼接得到完整频谱
        filt_full = torch.cat([fft_half, lower], dim=-2)  # H 维拼接

        return filt_full

    def forward(self, x):
        """
        x: real-valued tensor, shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        if ablate_highfreqconv== None:
            # FFT2 and shift
            x_fft = torch.fft.fft2(x)
            x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

            # Only keep upper half due to conjugate symmetry
            x_fft_half = x_fft_shift[:, :, :H // 2 + 1, :]  # shape: [B, C, H//2+1, W]

            # Separate real and imag parts
            real, imag = x_fft_half.real, x_fft_half.imag
            real_imag = torch.cat([real,imag],dim=1)
            real_imag = real_imag * self.high_mask1
            real_imag = self.C1(real_imag)
            real_imag = self.BN1(real_imag)
            real_imag = self.relu1(real_imag)
            real_imag = self.C2(real_imag * self.high_mask2)
            real, imag = real_imag.chunk(2, dim=1)
            # Recombine into complex tensor
            x_fft_out = torch.complex(real, imag)
            x_fft_out = self._build_symmetric(x_fft_out,self.H, self.W)
            # Step 4: ifft
            x_fft_filtered = torch.fft.ifftshift(x_fft_out, dim=(-2, -1))
            x_out = torch.fft.ifft2(x_fft_filtered).real

        elif ablate_highfreqconv == 'freq2image':
            x_fft = torch.fft.fft2(x)
            x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

            # Only keep upper half due to conjugate symmetry
            x_fft_half = x_fft_shift[:, :, :H // 2 + 1, :]  # shape: [B, C, H//2+1, W]

            # Separate real and imag parts
            real, imag = x_fft_half.real, x_fft_half.imag
            real_imag = torch.cat([real, imag], dim=1)
            real_imag = real_imag * self.high_mask1
            real, imag = real_imag.chunk(2, dim=1)
            # Recombine into complex tensor
            x_complex = torch.complex(real, imag)
            x_complex = self._build_symmetric(x_complex,H,W)
            x_complex = torch.fft.ifftshift(x_complex, dim=(-2, -1))
            x = torch.fft.ifft2(x_complex).real
            x = self.C1(x)
            x = self.BN1(x)
            x = self.relu1(x)
            x_out = self.C2(x)
        elif ablate_highfreqconv=='image':
            x = self.C1(x)
            x = self.BN1(x)
            x = self.relu1(x)
            x_out = self.C2(x)
        x_out = self.BN2(x_out)
        return x_out  # complex-valued output in frequency domain

# --- Simple ResNet Block ---
class SimpleBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res, leaky=False, **kwargs):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = FeatureWiseTransformation2d_fw(
                outdim)  # feature-wise transformation at the end of each residual block
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        self.FConv = HighFreqConv(indim, outdim, Size[outdim], Size[outdim], half_res=half_res)


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.drop_prob = 0.0
        self.FE = True


    def channel_drop(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 生成形状为 (B, C, 1, 1) 的 mask
        mask = torch.bernoulli(torch.full((B, C, 1, 1), 1 - self.drop_prob, device=device))
        return x * mask

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        if self.training:
            out = self.channel_drop(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        if self.FE:
            freq_out = self.FConv(x)
            out = out + freq_out + short_out
        else:
            out = out + short_out
        out = self.relu2(out)
        return out


# --- ConvNet module ---
class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out

# --- ConvNetNopool module ---
class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- ResNet module ---
class ResNet(nn.Module):
    maml = False

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False, se_layers=[],
                 low_freq_layers=[], **kwargs):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu, se_layer=i in se_layers,
                          low_freq=i in low_freq_layers)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out

    def forward_block1(self, x):
        out = self.trunk[:5](x)
        return out

    def forward_block2(self, x):
        out = self.trunk[5:6](x)
        return out

    def forward_block3(self, x):
        out = self.trunk[6:7](x)
        return out

    def forward_block4(self, x):
        out = self.trunk[7:8](x)
        return out

    def forward_rest(self, x):
        out = self.trunk[8:](x)
        return out

# --- Conv networks ---
def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


# --- ResNet networks ---
def ResNet10(flatten=True, leakyrelu=False, outdim=512, se_layers=[]):
    print('backbone:', 'return resnet10')
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, outdim], flatten, leakyrelu, se_layers)


def ResNet18(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten, leakyrelu)


def ResNet34(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten, leakyrelu)


model_dict = dict(Conv4=Conv4,
                  Conv6=Conv6,
                  ResNet10=ResNet10,
                  ResNet18=ResNet18,
                  ResNet34=ResNet34)

if __name__ == '__main__':
    model = FreqFilter(C=3, H=224, W=224)
    x = torch.randn(8, 3, 224, 224)
    out = model(x, domain='A')
    print(out.shape)
    L = out.mean()
    L.backward()
    print(model.filt_half_A.grad)