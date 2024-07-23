import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
from typing import Optional
import timm

class nutritionModel(nn.Module):
    def __init__(self, backbone ,tasks, use_end_relus=True, fc_d=2048):
        super().__init__()

        self.base_model = backbone

        self.tasks = tasks
        self.use_end_relus = use_end_relus

        self.fc2 = nn.Linear(fc_d, fc_d)

        self.task_layers = {}
        for task in self.tasks:
            self.task_layers[task] = [nn.Linear(fc_d, fc_d), nn.Linear(fc_d, 1)]

    def float(self):
        super().float()
        for task in self.tasks:
            for layer in self.task_layers[task]:
                for param in layer.parameters():
                    param.float()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # We need to manually send some layers to the appropriate device
        for task in self.tasks:
            for layer in self.task_layers[task]:
                layer.to(*args, **kwargs)
        return self


    def forward(self, x):
        # B * 2048 including fc1
        x = self.base_model.forward(x)
        # B * 2048
        # x = self.fc1(x)
        x = F.relu(x)

        # B * 2048
        x = self.fc2(x)
        x = F.relu(x)

        outputs = []
        for task in self.tasks:
            output = x
            if self.use_end_relus:
                # 0: fc3   1: 2048-1
                output = self.task_layers[task][0](output)
                output = F.relu(output)
                output = self.task_layers[task][1](output)
                output = F.relu(output)
            else:
                for layer in self.task_layers[task]:
                    output = layer(output)
                    output = F.relu(output)
            # outputs.append(torch.tensor(np.exp(output.cpu().detach().numpy())).cuda())
            outputs.append(output)
            # print(output)
        return outputs


from nutrition5k import swin_transformer_acmix

def create_model(backone, tasks, device, fc_d):
    if(backone=="SwinTransformer_acmix"):

        backone = swin_transformer_acmix.SwinTransformer_acmix(depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.1, num_classes=fc_d)
        backone = load_pretrained_acmix(backone, path="/home/stu_6/pretrained-model/swin_tiny_patch4_window7_224_22k.pth",device=device)
    else:
        backone = timm.create_model(backone, pretrained=True, num_classes=fc_d, drop_rate=0.1)
        backone = backone.to(device)
    # backone.eval()
    model = nutritionModel(backbone=backone, tasks=tasks, fc_d=fc_d, use_end_relus=False).to(device)

    return model

def load_pretrained(model, path="", map_localtion="cuda:0"):
        net_dict = model.state_dict()
        pretrained_model = torch.load(path, map_location=map_localtion)['net']
        state_dict = {}
        for k, v in pretrained_model.items():
            # if "module." in k:
            #     k = k.replace("module.", "")
            if k in net_dict.keys():
                state_dict[k] = v
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        model.load_state_dict(net_dict)
        return model

def load_pretrained_acmix(model = None, path = "", device="cuda:0"):
    net_dict = model.state_dict()
    pretrained_model = torch.load(path, map_location=device)
    state_dict = {}
    for k, v in pretrained_model['model'].items():
        print(k)
        if k in net_dict.keys():
            if "head" not in k:

                if pretrained_model['model'][k].shape == v.shape:
                    state_dict[k] = v
                    print(k)
                    print("-----")

    net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
    model.load_state_dict(net_dict)

    return model