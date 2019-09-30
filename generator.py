import torch
import torch.nn as nn
import torch.nn.functional as F
import Label2Image as Label2Image
# from Label2Image import LabelToImages
from NetworkUtil import ReducedConv,ResidualBlock
# import Normalization
# NOISE_DIM = 10
def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim, MomentumPointPDGScale,MomentumPointPDGOffset, EnergyScale,EnergyOffset,Nredconv_gen = 5,add_dense=False,use_resblock=False):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(6+self.z_dim*self.z_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, 256*4*4)
        self.bn_fc1 = nn.BatchNorm1d(self.fc1.out_features)
        self.bn_fc2 = nn.BatchNorm1d(self.fc2.out_features)
        
        # self.resblock = ResidualBlock(16)
        # self.resconv0 = ReducedConv(1+5,256,10,10,3)
        if add_dense:
            self.resconv1 = ReducedConv(256,128,4,10,3)
        else:
            self.resconv1 = ReducedConv(1+6,256,self.z_dim,10,3)
        # self.resconv1 = ReducedConv(256,128,4,10,3)
        self.resconv2 = ReducedConv(self.resconv1.out_channels,self.resconv1.out_channels//2,10,14,3)
        self.resconv3 = ReducedConv(self.resconv2.out_channels,self.resconv2.out_channels//2,14,18,3)
        self.resconv4 = ReducedConv(self.resconv3.out_channels,self.resconv3.out_channels//2,18,30,3)
        self.resconv5 = ReducedConv(self.resconv4.out_channels,1,30,30,3)
        # self.resconv6 = ReducedConv(8,1,26,30,3)
        self.bn1 = nn.BatchNorm2d(self.resconv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.resconv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.resconv3.out_channels)
        self.bn4 = nn.BatchNorm2d(self.resconv4.out_channels)
        # self.bn5 = nn.BatchNorm2d(self.resconv5.out_channels)
        self.resblock   = ResidualBlock(self.resconv4.out_channels)
        self.samesizerc = ReducedConv(self.resconv4.out_channels,self.resconv4.out_channels,30,30,3)
        self.bnrc       = nn.BatchNorm2d(self.samesizerc.out_channels)
        # self.dropout = nn.Dropout(p=0.2)
        self.finout = nn.Tanh()
        self.activation = nn.LeakyReLU(0.2)
        
        self.MomentumPointPDGScale = MomentumPointPDGScale
        self.MomentumPointPDGOffset = MomentumPointPDGOffset
        # self.PointScale    = PointScale
        self.EnergyScale   = EnergyScale
        self.EnergyOffset   = EnergyOffset
        self.Nredconv_gen  = Nredconv_gen
        self.add_dense = add_dense
        self.use_resblock = use_resblock
        
    def forward(self, z, ParticleMomentum_ParticlePoint_ParticlePDG):
        # ParticleMomentum_ParticlePoint = torch.div(ParticleMomentum_ParticlePoint,torch.cat([self.MomentumScale,self.PointScale]))
        ParticleMomentum_ParticlePoint_ParticlePDG = torch.div(ParticleMomentum_ParticlePoint_ParticlePDG-self.MomentumPointPDGOffset,self.MomentumPointPDGScale)
        # ParticleMomentum_ParticlePoint_ParticlePDG = Label2Image.LabelToImages(self.z_dim,self.z_dim,ParticleMomentum_ParticlePoint_ParticlePDG)
        EnergyDeposit = z.view(-1,1,self.z_dim,self.z_dim)
        
        if self.add_dense:
            EnergyDeposit = EnergyDeposit.view(-1,self.z_dim*self.z_dim)
            EnergyDeposit = torch.cat([EnergyDeposit,ParticleMomentum_ParticlePoint_ParticlePDG],dim=1)
            EnergyDeposit = self.activation(self.fc1(EnergyDeposit))
            EnergyDeposit = self.activation(self.fc2(EnergyDeposit))
            EnergyDeposit = self.activation(self.fc3(EnergyDeposit))
            EnergyDeposit = self.activation(self.fc4(EnergyDeposit))
            EnergyDeposit = EnergyDeposit.view(-1, 256, 4, 4)
        else:
            ParticleMomentum_ParticlePoint_ParticlePDG = Label2Image.LabelToImages(self.z_dim,self.z_dim,ParticleMomentum_ParticlePoint_ParticlePDG)
            # EnergyDeposit = z.view(-1,1,self.z_dim,self.z_dim)
            EnergyDeposit = torch.cat([EnergyDeposit,ParticleMomentum_ParticlePoint_ParticlePDG],dim=1)
            
        EnergyDeposit = self.activation(self.bn1(self.resconv1(EnergyDeposit)))
        EnergyDeposit = self.activation(self.bn2(self.resconv2(EnergyDeposit)))
        EnergyDeposit = self.activation(self.bn3(self.resconv3(EnergyDeposit)))
        EnergyDeposit = self.activation(self.bn4(self.resconv4(EnergyDeposit)))
        for i in range(self.Nredconv_gen):
            if self.use_resblock:
                EnergyDeposit = self.resblock(EnergyDeposit)
            else:
                EnergyDeposit = self.activation(self.bnrc(self.samesizerc(EnergyDeposit)))
            
        # EnergyDeposit = self.activation(self.bn5(self.resconv5(EnergyDeposit)))

        EnergyDeposit = self.resconv5(EnergyDeposit)
        EnergyDeposit = torch.tanh(EnergyDeposit)
        assert EnergyDeposit.shape[2]==30, 'Generated Image has wrong size: %d'%(EnergyDeposit.shape[2])
        EnergyDeposit = EnergyDeposit*self.EnergyScale+self.EnergyOffset

        return EnergyDeposit
    
    def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)
                