###################
# The basic methods of DnCNN
# This implementation is based on the code from https://github.com/basp-group/PnP-MMO-imaging
###################

import torch
import torch.nn as nn

from .basic_models import simple_CNN

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Denoiser:
    def __init__(self, file_name, ch):
        self.cost = 0
        self.network = load_network(file_name, cuda = torch.cuda.is_available(), channels = ch)

    def denoise(self, x):
        out = apply_model(x, self.network)
        return out
    
def load_checkpoint(model, file_name):
    checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint.module.state_dict())
    return model

def load_network(file_name, cuda = True, channels = 3):
    net = simple_CNN(ch_in=channels, ch_out=channels, ch=64, nl_type='relu', depth=20, bn=False)
    if(cuda):
        net.cuda()
        model = nn.DataParallel(net).cuda()
    else:
        model = nn.DataParallel(net)
    model = load_checkpoint(model, file_name)
    return model.eval()

def apply_model(x, network):
    x_in = torch.from_numpy(x)
    x_in.unsqueeze_(0)
    x_in = x_in.type(Tensor)                          

    with torch.no_grad():
        out_net = network(x_in)

    img = out_net[0, ...].cpu().detach().numpy()
    x_out = img                 
    return x_out