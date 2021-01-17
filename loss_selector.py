import torch
from torch import nn

#1 vgg()
def loss_selector(loss_net):
    #base
    if loss_net == "vgg16":
        from torchvision.models.vgg import vgg16
        net = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(net.features)[:31]).eval()
        return loss_network
    elif loss_net == "vgg16_bn":
        from torchvision.models.vgg import vgg16_bn
        net = vgg16_bn(pretrained=True)
        loss_network = nn.Sequential(*list(net.features)[:44]).eval()
        return loss_network
    elif loss_net == "resnet50":
        from torchvision.models.resnet import resnet50
        net=resnet50(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "resnet101":
        from torchvision.models.resnet import resnet101
        net=resnet101(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "resnet152":
        from torchvision.models.resnet import resnet152
        net=resnet152(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "squeezenet1_1":
        from torchvision.models.squeezenet import squeezenet1_1
        net=squeezenet1_1(pretrained=True)
        classifier=[item for item in net.classifier.modules()][1:-1]
        loss_network=nn.Sequential(*[net.features,*classifier]).eval()
        return loss_network
    elif loss_net == "densenet121":
        from torchvision.models.densenet import densenet121
        net=densenet121(pretrained=True)
        loss_network=nn.Sequential(*[net.features,nn.ReLU()]).eval()
        return loss_network
    elif loss_net == "densenet169":
        from torchvision.models.densenet import densenet169
        net=densenet169(pretrained=True)
        loss_network=nn.Sequential(*[net.features,nn.ReLU()]).eval()
        return loss_network
    elif loss_net == "densenet201":
        from torchvision.models.densenet import densenet201
        net=densenet201(pretrained=True)
        loss_network=nn.Sequential(*[net.features,nn.ReLU()]).eval()
        return loss_network        
    elif loss_net == "mobilenet_v2":
        from torchvision.models.mobilenet import mobilenet_v2
        net=mobilenet_v2(pretrained=True)
        loss_network=nn.Sequential(*[net.features]).eval()
        return loss_network                
    elif loss_net == "resnext50_32x4d":
        from torchvision.models.resnet import resnext50_32x4d
        net=resnext50_32x4d(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network      
    elif loss_net == "resnext101_32x8d":
        from torchvision.models.resnet import resnext101_32x8d
        net=resnext101_32x8d(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "wide_resnet50_2":
        from torchvision.models.resnet import wide_resnet50_2
        net=wide_resnet50_2(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "wide_resnet101_2":
        from torchvision.models.resnet import wide_resnet101_2
        net=wide_resnet101_2(pretrained=True)
        loss_network=nn.Sequential(*[child_module for child_module in net.children()][:-2]).eval()
        return loss_network
    elif loss_net == "inception_v3":
        #from torchvision.models.inception import inception_v3
        #net=inception_v3(pretrained=True)
        #net=nn.Sequential(*[child_module for child_module in net.children()][:]).eval()
        #return net

