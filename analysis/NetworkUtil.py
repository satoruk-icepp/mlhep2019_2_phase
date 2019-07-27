import torch
import torch.nn as nn
import torch.nn.functional as F

class ReducedConv(nn.Module):
    def __init__(self,input_size,output_size, input_dim, output_dim,kernel_size):
        super(ReducedConv, self).__init__()
        eps = 1e-3
        scale = float(output_dim+kernel_size-3+eps)/float(input_dim)
        self.ups = nn.Upsample(scale_factor = scale,mode = 'bilinear',align_corners=False )
        self.ref = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(input_size,output_size,kernel_size)
    def forward(self,x):
        return self.conv(self.ref(self.ups(x)))
#         return self.ref(self.ups(x))
#         return self.ref(self.ups(x))

class ResidualBlock(nn.Module):
    def __init__(self,input_size):
        super(ResidualBlock, self).__init__()        
        self.conv1 = nn.Conv2d(input_size,input_size,3,padding=1)
        self.conv2 = nn.Conv2d(input_size,input_size,3,padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)        
        self.activation = nn.LeakyReLU(0.2)
    def forward(self,xraw):
        x = self.activation(self.bn1(self.conv1(xraw)))
        x = self.activation(self.bn2(self.conv2(x))+xraw)
        return x