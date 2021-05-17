import sys

import config.yolov3_config_voc as cfg
import numpy as np
import torch
import torch.nn as nn
from utils.tools import *

from model.backbones.darknet65 import Darknet65
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
from model.necks.yolo_fpn_l import FPN_YOLOV3_L

sys.path.append("..")


class Yolov3_L(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, init_weights=True):
        super(Yolov3_L, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = Darknet65()
        self.__fpn = FPN_YOLOV3_L(fileters_in=[1024, 512, 256],
                                  fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # small
        self.__head_s = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))
