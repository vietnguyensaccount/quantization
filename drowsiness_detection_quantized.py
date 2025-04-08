import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def quantize_tensor(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = qmin - min_val / scale
    zero_point = int(max(qmin, min(qmax, round(zero_point))))
    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point

def dequantize_tensor(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.flatten_dim = 32 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class QuantizedConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.bias = conv.bias
        q_weight, self.scale, self.zero_point = quantize_tensor(conv.weight.data)
        self.q_weight = q_weight

    def forward(self, x):
        w = dequantize_tensor(self.q_weight, self.scale, self.zero_point)
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)

class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.bias = linear.bias
        q_weight, self.scale, self.zero_point = quantize_tensor(linear.weight.data)
