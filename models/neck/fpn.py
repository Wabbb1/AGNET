import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Conv_BN_ReLU


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(512,
                                      128,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(128,
                                     128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.smooth2_ = Conv_BN_ReLU(128,
                                     128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.smooth3_ = Conv_BN_ReLU(128,
                                     128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(256,
                                       128,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.latlayer2_ = Conv_BN_ReLU(128,
                                       128,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.latlayer3_ = Conv_BN_ReLU(64,
                                       128,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(p2)

        # p3 = self._upsample(p3, p2)
        # p4 = self._upsample(p4, p2)
        # p5 = self._upsample(p5, p2)

        f2 = p2
        f3 = self._upsample(p3, p2)
        f4 = self._upsample(p4, p2)
        f5 = self._upsample(p5, p2)
        f = torch.cat((f2, f3, f4, f5), 1)  # 512

        return f
