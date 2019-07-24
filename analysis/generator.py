import torch
import torch.nn as nn
import torch.nn.functional as F
import Label2Image
# from Label2Image import LabelToImages
from NetworkUtil import ReducedConv,ResidualBlock
# NOISE_DIM = 10


class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim, MomentumScale, PointScale, EnergyScale):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        # self.fc1 = nn.Linear(self.z_dim + 2 + 3, 256*4*4)
        # self.resblock = ResidualBlock(16)
        # self.resconv0 = ReducedConv(1+5,256,10,10,3)
        # self.resconv1 = ReducedConv(256,128,4,10,3)
        self.resconv1 = ReducedConv(1+5,256,self.z_dim,13,3)
        self.bn1 = nn.BatchNorm2d(256)
        self.resconv2 = ReducedConv(256,128,13,15,3)
        self.bn2 = nn.BatchNorm2d(128)
        self.resconv3 = ReducedConv(128,64,15,18,3)
        self.bn3 = nn.BatchNorm2d(64)
        self.resconv4 = ReducedConv(64,32,18,20,3)
        self.bn4 = nn.BatchNorm2d(32)
        self.resconv5 = ReducedConv(32,16,20,25,3)
        self.bn5 = nn.BatchNorm2d(16)                
        self.resconv6 = ReducedConv(16,1,25,30,3)
        # self.dropout = nn.Dropout(p=0.2)
        self.finout = nn.Tanh()
        self.MomentumScale = MomentumScale
        self.PointScale = PointScale
        self.EnergyScale = EnergyScale
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        ParticleMomentum_ParticlePoint = torch.div(ParticleMomentum_ParticlePoint,torch.cat([self.MomentumScale,self.PointScale]))
        LabelImages = Label2Image.LabelToImages(self.z_dim,self.z_dim,ParticleMomentum_ParticlePoint)
        z_image = z.view(-1,1,self.z_dim,self.z_dim)
        EnergyDeposit = torch.cat([z_image,LabelImages.cuda()],dim=1)
        # EnergyDeposit = x.view(-1, 256, 4, 4)
        
        EnergyDeposit = F.leaky_relu(self.bn1(self.resconv1(EnergyDeposit)),0.2)
        EnergyDeposit = F.leaky_relu(self.bn2(self.resconv2(EnergyDeposit)),0.2)
        EnergyDeposit = F.leaky_relu(self.bn3(self.resconv3(EnergyDeposit)),0.2)
        EnergyDeposit = F.leaky_relu(self.bn4(self.resconv4(EnergyDeposit)),0.2)
        EnergyDeposit = F.leaky_relu(self.bn5(self.resconv5(EnergyDeposit)),0.2)

        EnergyDeposit = self.resconv6(EnergyDeposit)
        EnergyDeposit = torch.tanh(EnergyDeposit)
        assert EnergyDeposit.shape[2]==30, 'Generated Image has wrong size: %d'%(EnergyDeposit.shape[2])
        EnergyDeposit = EnergyDeposit*self.EnergyScale

        return EnergyDeposit