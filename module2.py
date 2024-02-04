import torch.nn as nn
import torch
import torch.jit

'''
融合DGI物理层的U-Net模型
DGI_reconstruction ---> 差分鬼成像公式，用于将（batch_size, numpattern, 1）的桶测量值与 pattern关联生成差分鬼成像

'''


def DGI_reconstruction(bucket, pattern, batch_size, numpattern, px):
    # print('bucket shape', bucket.shape)
    # print('pattern[10][0]', pattern[10][0])
    bucket = torch.reshape(bucket, (batch_size, numpattern, 1))
    pattern = torch.reshape(pattern, (numpattern, px ** 2))  # （1024，16384）
    ave_pattern = torch.mean(pattern, dim=0)  # （16384，）
    CF_pattern = pattern - ave_pattern  # (1024,16384)
    comput2 = torch.sum(pattern, dim=1)  # (1024,1)
    mean_sum_pattern = torch.mean(comput2)  # 标量

    ave_bucket = torch.mean(bucket, dim=1) * 0.5  # (10,1)
    gamma = ave_bucket / mean_sum_pattern  # (10,1)
    temp = gamma * comput2
    temp = temp.unsqueeze(2)  # (10,1024,1)
    CF_bucket = bucket - temp
    CF_bucket = CF_bucket.squeeze(2)  # (10,1024)
    DGI = torch.matmul(CF_bucket, CF_pattern)  # （10，16384）
    DGI = torch.reshape(DGI, (batch_size, 1, px, px))  # （10， 1， 128， 128）

    return DGI


# 归一化DGI物理层
class EncoderLayer(nn.Module):

    def __init__(self, in_ch):
        super(EncoderLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_ch)
        self.bin_weights = None

    def forward(self, x, pattern, batch_size, numpattern, px):

        cc_out = DGI_reconstruction(x, pattern, batch_size, numpattern, px)  # 生成DGI
        cc_out = self.bn(cc_out)

        return cc_out


class DownsampleLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 编码层
        self.En1 = EncoderLayer(1)
        # 下采样
        self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], 1, 1, 1, 0),
            #             nn.Sigmoid(),
        )

    def forward(self, x, pattern, batch_size, numpattern, px):
        out_0 = self.En1(x, pattern, batch_size, numpattern, px)  # out0是低质量DGI
        out_1, out1 = self.d1(out_0)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)

        return out_0, out


if __name__ == '__main__':
    batch_size = 20
    numpattern = 1024  # 决定采样率 numpattern/(px**2)
    px = 128
    test_data = torch.randn(batch_size, numpattern, 1).cuda()
    learn_pattern = torch.randn(numpattern, 1, px, px).cuda()
    # pattern = torch.randn(1630, 1, 128, 128).cuda()

    model = UNet().cuda()
    # # 保存模型为ScriptModule
    # torch.jit.save(torch.jit.script(model), 'scripted_model.pth')

    DGI, out = model(test_data, learn_pattern, batch_size, numpattern, px)
    print('DGI shape', DGI.shape)
    print('out shape', out.shape)
    # print(out)
