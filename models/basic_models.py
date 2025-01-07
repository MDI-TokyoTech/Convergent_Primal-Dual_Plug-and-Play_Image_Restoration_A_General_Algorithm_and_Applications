###################
# DnCNN with configurable depth, nonlinearity, batch normalization, and a residual skip connection
# This implementation is based on the code from https://github.com/basp-group/PnP-MMO-imaging
###################

import torch
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class simple_CNN(nn.Module):
    def __init__(self, ch_in = 3, ch_out = 3, ch = 64, nl_type = 'relu', depth = 20, bn = False):
        super(simple_CNN, self).__init__()

        self.nl_type    = nl_type
        self.depth      = depth
        self.bn         = bn

        self.in_conv    = nn.Conv2d(ch_in, ch, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_list  = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1, bias = True) for _ in range(self.depth-2)])
        self.out_conv   = nn.Conv2d(ch, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True)

        if self.nl_type == 'relu':
            self.nl_list = nn.ModuleList([nn.LeakyReLU() for _ in range(self.depth - 1)])
        if self.bn:
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(self.depth - 2)])

    def forward(self, x_in):

        x = self.in_conv(x_in)
        x = self.nl_list[0](x)

        for i in range(self.depth-2):
            x_l = self.conv_list[i](x)  
            if self.bn:
                x_l = self.bn_list[i](x_l)
            x = self.nl_list[i + 1](x_l)
        
        x_out = self.out_conv(x) + x_in

        return x_out
