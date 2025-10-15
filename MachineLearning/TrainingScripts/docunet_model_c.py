import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from torchsummary import summary

def autopad(k, p=None, d=1):  
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.GELU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Clone để tránh lỗi CUDA Graphs khi dùng torch.compile()
        return self.act(self.bn(self.conv(x))).clone()

    def forward_fuse(self, x):
        return self.act(self.conv(x)).clone()


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

    
# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N = N
        self.c = int(c2 * e / self.N)
        self.add = shortcut and c1 == c2
        
        self.pwconv1 = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2 = Conv(c2//2, c2, 1, 1)
        self.m = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        # Clone residual để tránh lỗi CUDA Graphs
        x_residual = x.clone() if self.add else None
        
        x = self.pwconv1(x)

        # Tránh phép toán in-place, tạo list mới
        x_list = [x[:, 0::2, :, :].clone(), x[:, 1::2, :, :].clone()]
        
        # Cascade operations
        for m in self.m:
            x_list.append(m(x_list[-1]))
        
        # Tránh in-place addition
        x_list[0] = x_list[0] + x_list[1]
        x_list.pop(1)
        
        y = torch.cat(x_list, dim=1)
        y = self.pwconv2(y)
        
        # Tránh in-place addition
        return y + x_residual if self.add else y


'''
U-shape/U-like Model
'''
# Encoder in TinyU-Net
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.cmrf(x)
        # Clone skip connection để tránh lỗi CUDA Graphs
        x = self.downsample(skip)
        return x, skip.clone()
    

# Decoder in TinyU-Net
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        # Sử dụng upsample thay vì F.interpolate trực tiếp
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x
    

# TinyU-Net
class TinyUNet(nn.Module):
    """TinyU-Net with args(in_channels, num_classes)."""
    def __init__(self, in_channels=3, num_classes=2):
        super(TinyUNet, self).__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        
        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        feature_maps = self.decoder1(x, skip1)

        # Clone feature_maps để tránh lỗi khi reuse
        y = self.final_conv(feature_maps)
        return y, feature_maps.clone()


class TinyDocUnet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(TinyDocUnet, self).__init__()
        # U-net1
        self.U_net1 = TinyUNet(input_channels, n_classes)
        self.U_net2 = TinyUNet(64 + n_classes, n_classes)

    def forward(self, x):
        y1, feature_maps = self.U_net1(x)
        # Clone để tránh lỗi CUDA Graphs khi concat
        x_concat = torch.cat((feature_maps, y1), dim=1)
        y2, _ = self.U_net2(x_concat)
        return y1, y2


if __name__ == '__main__':
    model = TinyDocUnet(input_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Compile model với cấu hình tối ưu cho CUDA Graphs
    # Tắt cudagraphs nếu vẫn gặp lỗi
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available or failed: {e}")
        print("Running without compilation")

    summary(model, (3, 512, 512))
        
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    flops, params = profile(model, (dummy_input, ), verbose=False)
    
    # flops * 2 because profile does not consider convolution as two operations.
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.4f")
    print(f'Total GFLOPs: {flops}')
    print(f'Total Params: {params}')