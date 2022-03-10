#！/usr/bin/env python
#!-*coding:utf-8 -*-
#!@Author :lzy
#!@File :gridnet.py
import torch
import torch.nn as nn

class LateralBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)

        return fx + x

class DownSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 2, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )
    def forward(self, x):
        return self.f(x)


class GridNet(nn.Module):
    def __init__(self, out_chs=3, grid_chs=[32, 64, 96]):
        # n_row = 3, n_col = 6, n_chs = [32, 64, 96]):
        super().__init__()

        self.n_row = 3
        self.n_col = 4
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(3, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):

                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                # 00 10 要特殊设置
                if r==0 and c==0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(3, out_ch))
                elif r==1 and c==0:
                    setattr(self, f'down_{r}_{c}', LateralBlock(3, out_ch))
                else:
                    setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x_l1,x_l2,x_l3):
        
        state_00 = self.lateral_init(x_l1)
        state_10 = self.down_0_0(x_l2)
        state_20 = self.down_1_0(x_l3)
       
        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_1(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_1(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_2(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_2(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_2(state_03)

        return self.lateral_final(state_04)

#if __name__=='__main__':
    #model=GridNet()
    #tensor1=torch.rand(1,3,256,256)
    #tensor2=torch.rand(1,3,128,128)
    #tensor3=torch.rand(1,3,64,64)
    #model.eval()
    #result=model(tensor1,tensor2,tensor3)
    #print(result.shape)
