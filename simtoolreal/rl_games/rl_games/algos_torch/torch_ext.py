import numpy as np
import torch
import torch.nn as nn


def shape_whc_to_cwh(shape):
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])
    
    return shape


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1, validate_args=False)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(torch.from_numpy(np.array(alpha)))
    p = alpha_normal_cdf + (normal.cdf(torch.from_numpy(np.array(beta))) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x

def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
    return parameterized_truncated_normal(uniform, mu, sigma, a, b)

def sample_truncated_normal(shape=(), mu=0.0, sigma=1.0, a=-2, b=2):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)), mu, sigma, a, b)

def variance_scaling_initializer(tensor, mode='fan_in',scale = 2.0):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    print(fan, scale)
    sigma = np.sqrt(scale / fan)
    with torch.no_grad():
        tensor[:] = sample_truncated_normal(tensor.size(), sigma=sigma)
        return tensor


class CoordConv2d(nn.Conv2d):
    pool = {}
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels + 2, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
    @staticmethod
    def get_coord(x):
        key = int(x.size(0)), int(x.size(2)), int(x.size(3)), x.type()
        if key not in CoordConv2d.pool:
            theta = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
            coord = torch.nn.functional.affine_grid(theta, torch.Size([1, 1, x.size(2), x.size(3)])).permute([0, 3, 1, 2]).repeat(
                x.size(0), 1, 1, 1).type_as(x)
            CoordConv2d.pool[key] = coord
        return CoordConv2d.pool[key]
    def forward(self, x):
        return torch.nn.functional.conv2d(torch.cat([x, self.get_coord(x).type_as(x)], 1), self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LayerNorm2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.register_buffer("gamma", torch.ones(features).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("beta", torch.ones(features).unsqueeze(-1).unsqueeze(-1))

        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)


class IdentityRNN(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(IdentityRNN, self).__init__()
        assert(in_shape == out_shape)
        self.identity = torch.nn.Identity()

    def forward(self, x, h):
        return self.identity(x), h

 
