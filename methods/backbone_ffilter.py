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


# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if self.maml:
            self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
            self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
        else:
            self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
            self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        if hidden is None:
            hx = torch.zeors_like(x)
            cx = torch.zeros_like(x)
        else:
            hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


# --- LSTM module for matchingnet ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        assert (self.num_layers == 1)

        self.lstm = LSTMCell(input_size, hidden_size, self.bias)

    def forward(self, x, hidden=None):
        # swap axis if batch first
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h0, c0 = hidden

        # forward
        outs = []
        hn = h0[0]
        cn = c0[0]
        for seq in range(x.size(0)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn.unsqueeze(0))
        outs = torch.cat(outs, dim=0)

        # reverse foward
        if self.num_directions == 2:
            outs_reverse = []
            hn = h0[1]
            cn = c0[1]
            for seq in range(x.size(0)):
                seq = x.size(1) - 1 - seq
                hn, cn = self.lstm(x[seq], (hn, cn))
                outs_reverse.append(hn.unsqueeze(0))
            outs_reverse = torch.cat(outs_reverse, dim=0)
            outs = torch.cat([outs, outs_reverse], dim=2)

        # swap axis if batch first
        if self.batch_first:
            outs = outs.permute(1, 0, 2)
        return outs


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


class FreqFilter(nn.Module):
    def __init__(self, C, H, W):
        super(FreqFilter, self).__init__()
        self.C, self.H, self.W = C, H, W
        self.H_half = H // 2 + 1

        # 只保留上半部分（H//2 行），W 保留全部
        self.filt_half_S = nn.Parameter(torch.ones(1, C, self.H_half, W))
        self.filt_half_A = nn.Parameter(torch.ones(1, C, self.H_half, W))
        self.relu = nn.ReLU(inplace=True)
        self.domain=None
        self.stop_grad=False

    def _build_symmetric_filter(self, filt_half):
        """
        将上半部分 (1, C, H_half, W) 扩展为复共轭对称 (1, C, H, W)
        支持偶数和奇数 H
        """
        # 上下翻转
        if self.H % 2 == 0:
            # 偶数 H，filt_half 包含 H//2 + 1 行（含中心）
            upper = filt_half  # (1, C, H//2 + 1, W)
            lower = torch.flip(filt_half[:, :, 1: - 1, :], dims=[-2])  # 去除第0行和中心行，对剩下的做镜像
        else:
            # 奇数 H，filt_half 通常就是 (H+1)//2，最后一行是中心行
            upper = filt_half  # (1, C, (H+1)//2, W)
            lower = torch.flip(filt_half[:, :, : -1, :], dims=[-2])  # 去掉中心行再翻转，对剩下的做镜像

        #左右翻转
        if self.W % 2 == 0:
            lower[:, :, :, 1:] = torch.flip(lower[:, :, :, 1:], dims=[-1])
        else:
            lower = torch.flip(lower, dims=[-1])

        filt_full = torch.cat([upper, lower], dim=-2)  # dim=-2 即 H 维度
        return self.relu(filt_full)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W and C == self.C

        # Step 2: 构造滤波器
        # if self.domain=='S': print('using source filter')
        # elif self.domain=='A': print('using target filter')
        # else: print('not using filter')
        if self.domain == 'S':
            filt_half = self.filt_half_S
        elif self.domain == 'A':
            filt_half = self.filt_half_A
        else:
            return x

        filt = self._build_symmetric_filter(filt_half)
        if self.stop_grad:
            filt = filt.detach()

        # Step 1: FFT + shift
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # Step 3: 点乘
        x_fft_filtered = x_fft * filt

        # Step 4: ifft
        x_fft_filtered = torch.fft.ifftshift(x_fft_filtered, dim=(-2, -1))
        x_out = torch.fft.ifft2(x_fft_filtered).real

        return x_out

class FreqBlock(nn.Module):
    def __init__(self, channel, H, W):
        super(FreqBlock, self).__init__()
        self.H, self.W = H, W

        # Shared convolution for real and imag
        self.C1 = nn.Conv2d(channel, channel, 1)
        self.BNR1 = nn.BatchNorm2d(channel)
        self.BNI1 = nn.BatchNorm2d(channel)

        self.C2 = nn.Conv2d(channel, channel, 1)
        self.BNR2 = nn.BatchNorm2d(channel)
        self.BNI2 = nn.BatchNorm2d(channel)

        # Activation
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BNR1, self.BNR2, self.BNI1, self.BNI2]
        for layer in self.parametrized_layers:
            init_layer(layer)

    def _build_symmetric(self, fft_half):
        """
        将上半部分 (1, C, H_half, W) 扩展为复共轭对称 (1, C, H, W)
        支持偶数和奇数 H
        """
        # 上下翻转
        if self.H % 2 == 0:
            # 偶数 H，filt_half 包含 H//2 + 1 行（含中心）
            lower = torch.flip(fft_half[:, :, 1: - 1, :], dims=[-2])  # 去除第0行和中心行，对剩下的做镜像
        else:
            # 奇数 H，filt_half 通常就是 (H+1)//2，最后一行是中心行
            lower = torch.flip(fft_half[:, :, : -1, :], dims=[-2])  # 去掉中心行再翻转，对剩下的做镜像

        # 左右翻转
        if self.W % 2 == 0:
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
        x_shortcut = x
        # FFT2 and shift
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # Only keep upper half due to conjugate symmetry
        x_fft_half = x_fft_shift[:, :, :H // 2 + 1, :]  # shape: [B, C, H//2+1, W]

        # Separate real and imag parts
        real = x_fft_half.real
        real = self.C1(real)
        real = self.BNR1(real)
        real = self.relu1(real)
        real = self.C2(real)
        real = self.BNR2(real)

        imag = x_fft_half.imag
        imag = self.C1(imag)
        imag = self.BNI1(imag)
        imag = self.relu1(imag)
        imag = self.C2(imag)
        imag = self.BNI2(imag)

        # Recombine into complex tensor
        x_fft_out = torch.complex(real, imag)
        x_fft_out = self._build_symmetric(x_fft_out)

        # Step 4: ifft
        x_fft_filtered = torch.fft.ifftshift(x_fft_out, dim=(-2, -1))
        x_out = torch.fft.ifft2(x_fft_filtered).real

        x_out = self.relu2(x_out+x_shortcut)
        return x_out  # complex-valued output in frequency domain

Size  = {64:56,128:28,256:14,512:7}

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
                outdim)# feature-wise transformation at the end of each residual block
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.freq_filter = FreqFilter(outdim,Size[outdim],Size[outdim])
        self.BN3 = nn.BatchNorm2d(outdim)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2, self.BN3]

        self.half_res = half_res

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

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        self.a = out
        out = self.freq_filter(out)
        out = self.BN3(out)
        out = self.C2(out)
        out = self.BN2(out)
        self.b = out
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        self.c = out
        out = self.relu2(out)
        self.d = out
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