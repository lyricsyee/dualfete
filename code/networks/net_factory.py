from networks.unet_3D import unet_3D
from networks.vnet import VNet

"""
Reference: 
1. VNet - https://github.com/ycwu1997/SS-Net/blob/main/code/networks/VNet.py
2. UNet_3D - https://github.com/HiLab-git/SSL4MIS/blob/master/code/networks/unet_3D.py
"""

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", normalization="batchnorm"):
    assert normalization in ["batchnorm", "groupnorm", "instancenorm"]
    
    if net_type == "unet" and mode == "train":
        net = unet_3D(in_channels=in_chns, n_classes=class_num, has_dropout=True).cuda()
    elif net_type == "unet" and mode == "test":
        net = unet_3D(in_channels=in_chns, n_classes=class_num, has_dropout=False).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    else:
        print("Erroneous network type: ", net_type)
        raise NotImplementedError
    return net
