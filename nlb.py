import torch
from torch import nn
from torch.nn import functional as F

class SNLB(nn.Module):
    def __init__(self, in_channels, sub_sample=False):
        super(SNLB, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant(self.W[1].weight, 0)
        nn.init.constant(self.W[1].bias, 0)


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=2))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class CNLB(nn.Module):
    def __init__(self, in_channels):
        super(CNLB, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant(self.W[1].weight, 0)
        nn.init.constant(self.W[1].bias, 0)


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # (b, c, wh)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # (b, c, wh)
        theta_x = theta_x.permute(0, 2, 1)
        # (b, wh, c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # (b, c, wh)

        f = torch.matmul(phi_x, theta_x)
        # (b, c, c)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        # (b, c, wh)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class Split_SNLB(nn.Module):
    def __init__(self, in_channels, block_size, sub_sample=True):
        super(Split_SNLB, self).__init__()

        self.block_size = block_size
        self.nlb = SNLB(in_channels, sub_sample=sub_sample)

    def forward(self, x):
        b, c, h, w = x.size()

        stencil_size_h = (h + self.block_size - 1) / self.block_size
        stencil_size_w = (w + self.block_size - 1) / self.block_size

        stride_h = h * 1.0 / self.block_size
        stride_w = w * 1.0 / self.block_size

        h_idxs = [int(round(stride_h * i)) for i in torch.arange(self.block_size)]
        w_idxs = [int(round(stride_w * i)) for i in torch.arange(self.block_size)]
        if h_idxs[-1] + stencil_size_h != h:
            h_idxs[-1] = h - stencil_size_h
        if w_idxs[-1] + stencil_size_w != w:
            w_idxs[-1] = w - stencil_size_w

        block_list = []
        for h_idx in h_idxs:
            for w_idx in w_idxs:
                block_list.append(self.nlb(x[:,:, h_idx:h_idx+stencil_size_h, w_idx:w_idx+stencil_size_w]))

        block_fearues = torch.cat(block_list, dim=1)
        block_total_h = block_fearues.size(2) * self.block_size

        block_fearues = block_fearues.view(b, c, block_total_h, -1)

        return block_fearues


class Split_CNLB(nn.Module):
    def __init__(self, in_channels, block_size):
        super(Split_CNLB, self).__init__()

        self.block_size = block_size
        self.block_channel = (in_channels + self.block_size - 1) / self.block_size
        self.nlb = CNLB(self.block_channel)

    def forward(self, x):
        b, c, h, w = x.size()

        stencil_size_c = (c + self.block_size - 1) / self.block_size

        stride_c = c * 1.0 / self.block_size

        c_idxs = [int(round(stride_c * i)) for i in torch.arange(self.block_size)]
        if c_idxs[-1] + stencil_size_c != c:
            c_idxs[-1] = c - stencil_size_c


        block_list = []
        for c_idx in c_idxs:
            block_list.append(self.nlb(x[:, c_idx:c_idx+stencil_size_c, :, :]))

        block_fearues = torch.cat(block_list, dim=1)
        block_fearues = block_fearues.view(b, -1, h, w)

        return block_fearues


class SPM(nn.Module):
    def __init__(self, in_channels, sub_sample=False):
        super(SPM, self).__init__()

        self.space_nlb_1 = Split_SNLB(in_channels, block_size=5, sub_sample=sub_sample)
        self.space_nlb_2 = Split_SNLB(in_channels, block_size=3, sub_sample=sub_sample)
        self.space_nlb_3 = Split_SNLB(in_channels, block_size=2, sub_sample=sub_sample)
        self.space_nlb_4 = Split_SNLB(in_channels, block_size=1, sub_sample=sub_sample)

        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 5, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels * 5),
            nn.PReLU(),
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
        )

    def forward(self, x):
        space_nlb_1 = self.space_nlb_1(x)
        space_nlb_2 = self.space_nlb_2(x)
        space_nlb_3 = self.space_nlb_3(x)
        space_nlb_4 = self.space_nlb_4(x)

        space_nlb_1 = F.upsample_bilinear(space_nlb_1, size=space_nlb_1.size()[2:])
        space_nlb_2 = F.upsample_bilinear(space_nlb_2, size=space_nlb_1.size()[2:])
        space_nlb_3 = F.upsample_bilinear(space_nlb_3, size=space_nlb_1.size()[2:])
        space_nlb_4 = F.upsample_bilinear(space_nlb_4, size=space_nlb_1.size()[2:])
        x = F.upsample_bilinear(x, size=space_nlb_1.size()[2:])

        SF = self.spatial_fuse(torch.cat((x, space_nlb_1, space_nlb_2, space_nlb_3, space_nlb_4), 1))

        return SF


class CPM(nn.Module):
    def __init__(self, in_channels):
        super(CPM, self).__init__()

        self.channel_nlb_1 = Split_CNLB(in_channels, block_size=16)
        self.channel_nlb_2 = Split_CNLB(in_channels, block_size=8)
        self.channel_nlb_3 = Split_CNLB(in_channels, block_size=4)
        self.channel_nlb_4 = Split_CNLB(in_channels, block_size=1)

        self.channel_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 5, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels * 5),
            nn.PReLU(),
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
        )

    def forward(self, x):


        channel_nlb_1 = self.channel_nlb_1(x)
        channel_nlb_2 = self.channel_nlb_2(x)
        channel_nlb_3 = self.channel_nlb_3(x)
        channel_nlb_4 = self.channel_nlb_4(x)
        CF = self.channel_fuse(torch.cat((x, channel_nlb_1, channel_nlb_2, channel_nlb_3, channel_nlb_4), 1))

        return CF


class DPM(nn.Module):
    def __init__(self, in_channels, sub_sample=False):
        super(DPM, self).__init__()

        self.space_nlb_1 = Split_SNLB(in_channels, block_size=5, sub_sample=sub_sample)
        self.space_nlb_2 = Split_SNLB(in_channels, block_size=3, sub_sample=sub_sample)
        self.space_nlb_3 = Split_SNLB(in_channels, block_size=2, sub_sample=sub_sample)
        self.space_nlb_4 = Split_SNLB(in_channels, block_size=1, sub_sample=sub_sample)

        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 5, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels * 5),
            nn.PReLU(),
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
        )

        # type1: 16, 8, 4, 1
        self.channel_nlb_1 = Split_CNLB(in_channels, block_size=16)
        self.channel_nlb_2 = Split_CNLB(in_channels, block_size=8)
        self.channel_nlb_3 = Split_CNLB(in_channels, block_size=4)
        self.channel_nlb_4 = Split_CNLB(in_channels, block_size=1)
        
        self.channel_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 5, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels * 5),
            nn.PReLU(),
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
        )

    def forward(self, x):
        space_nlb_1 = self.space_nlb_1(x)
        space_nlb_2 = self.space_nlb_2(x)
        space_nlb_3 = self.space_nlb_3(x)
        space_nlb_4 = self.space_nlb_4(x)

        space_nlb_1 = F.upsample_bilinear(space_nlb_1, size=space_nlb_1.size()[2:])
        space_nlb_2 = F.upsample_bilinear(space_nlb_2, size=space_nlb_1.size()[2:])
        space_nlb_3 = F.upsample_bilinear(space_nlb_3, size=space_nlb_1.size()[2:])
        space_nlb_4 = F.upsample_bilinear(space_nlb_4, size=space_nlb_1.size()[2:])
        x = F.upsample_bilinear(x, size=space_nlb_1.size()[2:])

        SF = self.spatial_fuse(torch.cat((x, space_nlb_1, space_nlb_2, space_nlb_3, space_nlb_4), 1))

        channel_nlb_1 = self.channel_nlb_1(SF)
        channel_nlb_2 = self.channel_nlb_2(SF)
        channel_nlb_3 = self.channel_nlb_3(SF)
        channel_nlb_4 = self.channel_nlb_4(SF)

        CF = self.channel_fuse(torch.cat((SF, channel_nlb_1, channel_nlb_2, channel_nlb_3, channel_nlb_4), 1))

        return CF





