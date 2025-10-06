import torch
from torch import nn
from torch.nn import functional as F

# 共性的卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.LeakyReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

# 下采样改用卷积来取代max-pooling
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

# 最邻近插值法进行上采样，除此之外还需使用1x1的卷积降通道
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv3d(channel, channel // 2, kernel_size=1, stride=1)

    def forward(self, x, left_out):
        # 获取 left_out 的尺寸
        target_size = left_out.size()[2:]  # 获取深度、高度、宽度

        # 使用 trilinear 插值方法，并指定目标尺寸
        up = F.interpolate(x, size=target_size, mode='trilinear', align_corners=True)

        # 使用1x1卷积降低通道数
        out = self.layer(up)

        # 拼接
        return torch.cat((left_out, out), dim=1)

class  UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # contract path
        self.conv1 = ConvLayer(1, 64)  # 输入为单通道
        self.down1 = DownSample(64)
        self.conv2 = ConvLayer(64, 128)
        self.down2 = DownSample(128)
        self.conv3 = ConvLayer(128, 256)
        self.down3 = DownSample(256)
        self.conv4 = ConvLayer(256, 512)
        self.down4 = DownSample(512)
        self.conv5 = ConvLayer(512, 1024)
        # expansive path
        self.up1 = UpSample(1024)
        self.conv6 = ConvLayer(1024, 512)
        self.up2 = UpSample(512)
        self.conv7 = ConvLayer(512, 256)
        self.up3 = UpSample(256)
        self.conv8 = ConvLayer(256, 128)
        self.up4 = UpSample(128)
        self.conv9 = ConvLayer(128, 64)

        # 输出层
        self.out = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.active_func = nn.Sigmoid()

    def forward(self, x):
        r1 = self.conv1(x)
        r2 = self.conv2(self.down1(r1))
        r3 = self.conv3(self.down2(r2))
        r4 = self.conv4(self.down3(r3))
        r5 = self.conv5(self.down4(r4))
        o1 = self.conv6(self.up1(r5, r4))
        o2 = self.conv7(self.up2(o1, r3))
        o3 = self.conv8(self.up3(o2, r2))
        o4 = self.conv9(self.up4(o3, r1))

        return self.active_func(self.out(o4))

# 验证一下网络结构
if __name__ == '__main__':
    x = torch.randn(1, 1, 64, 64, 64)  # 输入数据为 (batch_size, channels, depth, height, width)
    model = UNet()
    output = model(x)
    print(output.shape)
