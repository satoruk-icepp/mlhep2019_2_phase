import torch
import torch.nn as nn
import torch.nn.functional as F

class ReducedConv(nn.Module):
    def __init__(self,input_layers,output_layers, input_dim, output_dim,kernel_size):
        super(ReducedConv, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        scale = 2
        self.kernel_size = self.input_dim*scale+ 3 - self.output_dim
        # scale = float(output_dim+kernel_size-3)/float(input_dim)
        self.ups = nn.Upsample(scale_factor = scale,mode = 'bilinear',align_corners=False )
        self.ref = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(input_layers,output_layers,self.kernel_size)
    def forward(self,x):
        assert x.shape[2]==self.input_dim, "original image is wrong size, %d"%(x.shape[2])
        x = self.conv(self.ref(self.ups(x)))
        assert x.shape[2]==self.output_dim, "not properly scaled, %d"%(x.shape[2])
        return x
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