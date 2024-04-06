import torch


def weights_init_kaiming(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
