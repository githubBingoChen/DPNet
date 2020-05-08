import torch
import torch.nn.functional as F
from torch import nn
from resnet import Resnet
from nlb import DPM


class DPNet(nn.Module):

    def __init__(self):
        super(DPNet, self).__init__()
        resnet = Resnet()
        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.nlb_paramid1 = DPM(256, sub_sample=False)
        self.nlb_paramid2 = DPM(256, sub_sample=False)

        self.reduce_layer4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),
        )

        self.reduce_layer3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),
        )
        self.reduce_layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),
        )
        self.reduce_layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),
        )

        self.predict_dnlp1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict4 = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict3 = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        reduce_layer4 = self.reduce_layer4(layer4)
        reduce_layer3 = self.reduce_layer3(layer3)
        reduce_layer2 = self.reduce_layer2(layer2)
        reduce_layer1 = self.reduce_layer1(layer1)

        reduce_layer4 = self.nlb_paramid1(reduce_layer4)
        predict_dnlp1 = self.predict_dnlp1(reduce_layer4)
        reduce_layer4 = self.nlb_paramid2(reduce_layer4)
        
        predict4 = self.predict4(torch.cat((predict_dnlp1, reduce_layer4), 1)) + predict_dnlp1
        predict4 = F.upsample_bilinear(predict4, size=reduce_layer3.size()[2:])
        reduce_layer4 = F.upsample_bilinear(reduce_layer4, size=reduce_layer3.size()[2:])

        fpn_layer3 = reduce_layer3 + reduce_layer4
        predict3 = self.predict3(torch.cat((predict4, fpn_layer3), 1)) + predict4
        predict3 = F.upsample_bilinear(predict3, size=layer2.size()[2:])
        fpn_layer3 = F.upsample_bilinear(fpn_layer3, size=layer2.size()[2:])

        fpn_layer2 = reduce_layer2 + fpn_layer3
        predict2 = self.predict2(torch.cat((predict3, fpn_layer2), 1)) + predict3
        predict2 = F.upsample_bilinear(predict2, size=layer1.size()[2:])
        fpn_layer2 = F.upsample_bilinear(fpn_layer2, size=layer1.size()[2:])

        fpn_layer1 = reduce_layer1 + fpn_layer2
        predict1 = self.predict1(torch.cat((predict2, fpn_layer1), 1)) + predict2

        predict1 = F.upsample_bilinear(predict1, size=x.size()[2:])
        predict2 = F.upsample_bilinear(predict2, size=x.size()[2:])
        predict3 = F.upsample_bilinear(predict3, size=x.size()[2:])
        predict4 = F.upsample_bilinear(predict4, size=x.size()[2:])

        predict_dnlp1 = F.upsample_bilinear(predict_dnlp1, size=x.size()[2:])

        if self.training:
            return F.sigmoid(predict1), F.sigmoid(predict2), F.sigmoid(predict3), F.sigmoid(predict4), F.sigmoid(predict_dnlp1)
        return F.sigmoid(predict1)