# coding=utf-8
import sys
sys.path.append("..")
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import params as pms


def parse_model_cfg(path):
    path = os.path.join(pms.PROJECT_PATH,path)
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def create_modules(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            stride = int(module_def["stride"])
            modules.add_module("conv_%d"%i, nn.Conv2d(in_channels=output_filters[-1],
                                                      out_channels=filters,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=pad,
                                                      bias=not bn))
            if bn:
                modules.add_module("batch_norm_%d"%i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d"%i, nn.LeakyReLU(0.1, inplace=True))

        elif module_def["type"] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def["stride"]))
            modules.add_module("upsample_%d"%i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(',')] # 注意：int(" 1") 和int("1 ")都为1，即空格不影响
            filters = sum([output_filters[i+1 if i >0 else i] for i in layers])
            modules.add_module("route_%d"%i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d"%i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(',')]
            anchors = [float(x) for x in module_def["anchors"].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def["classes"])

            yolo_layer = YOLOLayer(anchors, nC)
            modules.add_module("yolo_%d"%i, yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)
        self.nC = nC


    def forward(self, p, img_size, var=None):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.nA, 5 + self.nC , nG, nG).permute(0, 3, 4, 1, 2)

        # 训练和测试都进行解码
        p_de = self.__decode(p.clone().detach(), img_size)
        return (p, p_de)

    def __decode(self, p, img_size):
        conv_shape = p.shape

        device = p.device
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = self.nA
        num_classes = self.nC
        stride = img_size / output_size
        anchors = (1.0 * self.anchors / stride).to(device)


        conv_output = p.view(batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes)
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]


        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5+self.nC) if not self.training else pred_bbox  # 例如 shape : [bs, 13*13*3+26*26*3+52*52*3, 25]


class Darknet(nn.Module):
    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]["height"] = img_size
        self.hypeparams, self.module_list = create_modules(self.module_defs)


    def forward(self, x):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def["type"]
            if mtype in ["convolutional", "upsample"]:
                x = module(x)

            elif mtype == "route":
                layer_i = [int(x) for x in module_def["layers"].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif mtype == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif mtype == "yolo": # yolo层用到FPN各层的特征图和原图尺寸img_size,其中img_size是用来提供缩放比例的
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if self.training:
            p, p_d = list(zip(*output))
            return p, p_d
        else:
            p, p_d = list(zip(*output))
            return p, torch.cat(p_d, 0)


    def load_darknet_weights(self, weight_path):
        weight_path = os.path.join(pms.PROJECT_PATH, weight_path)

        fp = open(weight_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs[:75], self.module_list[:75])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()

                    bn_b = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b

                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b

                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w



if __name__ == "__main__":

    net = Darknet("cfg/yolov3-voc.cfg")
    print(net)
    in_img = torch.randn(2, 3, 416, 416)
    p, p_d = net(in_img)

    print(p[1].shape)
    print(p_d[0].shape)
