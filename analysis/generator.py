import torch
import torch.nn as nn
import torch.nn.functional as F
NOISE_DIM = 10
EnergyDepositScale = 4000
MomentumScale = [30,30,100]
PointScale = [10,10]

class ReducedConv(nn.Module):
    def __init__(self,input_size,output_size, input_dim, output_dim,kernel_size):
        super(ReducedConv, self).__init__()
        scale = float(output_dim+kernel_size-3)/float(input_dim)
        self.ups = nn.Upsample(scale_factor = scale,mode = 'bilinear',align_corners=False )
        self.ref = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(input_size,output_size,kernel_size)
    def forward(self,x):
        return self.conv(self.ref(self.ups(x)))
#         return self.ref(self.ups(x))

class ResidualBlock(nn.Module):
    def __init__(self,input_size):
        super(ResidualBlock, self).__init__()        
        self.conv1 = nn.Conv2d(input_size,input_size,3,padding=1)
        self.conv2 = nn.Conv2d(input_size,input_size,3,padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)        
    def forward(self,xraw):
        x = F.leaky_relu(self.bn1(self.conv1(xraw)))
        x = F.leaky_relu(self.bn2(self.conv2(x))+xraw)
        return x

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim, MomentumScale, PointScale, EnergyScale):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim + 2 + 3, 256*4*4)
        self.resblock = ResidualBlock(16)
        self.resconv1 = ReducedConv(256,128,4,10,3)
        self.bn1 = nn.BatchNorm2d(128)
        self.resconv2 = ReducedConv(128,64,10,15,3)
        self.bn2 = nn.BatchNorm2d(64)
        self.resconv3 = ReducedConv(64,32,15,20,3)
        self.bn3 = nn.BatchNorm2d(32)
        self.resconv4 = ReducedConv(32,16,20,25,3)
        self.bn4 = nn.BatchNorm2d(16)        
        self.resconv5 = ReducedConv(16,1,25,30,3)
        self.dropout = nn.Dropout(p=0.2)
        self.finout = nn.Tanh()
        self.MomentumScale = MomentumScale
        self.PointScale = PointScale
        self.EnergyScale = EnergyScale
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        ParticleMomentum_ParticlePoint = torch.div(ParticleMomentum_ParticlePoint,torch.cat([self.MomentumScale,self.PointScale]))
        x = F.leaky_relu(self.fc1(
            torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        ))
        
        EnergyDeposit = x.view(-1, 256, 4, 4)
        
        EnergyDeposit = F.relu(self.bn1(self.resconv1(EnergyDeposit)))
        EnergyDeposit = F.relu(self.bn2(self.resconv2(EnergyDeposit)))
        EnergyDeposit = F.relu(self.bn3(self.resconv3(EnergyDeposit)))
        EnergyDeposit = F.relu(self.bn4(self.resconv4(EnergyDeposit)))

        EnergyDeposit = self.resconv5(EnergyDeposit)
        EnergyDeposit = torch.tanh(EnergyDeposit)
        EnergyDeposit = EnergyDeposit*EnergyDepositScale

        return EnergyDeposit