import torch
import torch.nn as nn
import torch.nn.functional as F
# NOISE_DIM = 10

class ModelD(nn.Module):
    def __init__(self, z_dim, MomentumScale, PointScale, EnergyScale):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1+z_dim, 16, 4, stride=2)#30->14
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(16, 32, 4)##14->11
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4)##11->8
        self.bn3 = nn.BatchNorm2d(64)        
        self.conv4 = nn.Conv2d(64, 128, 3)##8->6
        self.bn4 = nn.BatchNorm2d(128)                
        self.conv5 = nn.Conv2d(128, 1, 6)##6->4
        self.bn5 = nn.BatchNorm2d(256)
        
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.bn_fc4 = nn.BatchNorm1d(64)        
        self.MomentumScale = MomentumScale
        self.PointScale = PointScale
        self.EnergyScale = EnergyScale
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
#         EnergyDeposit = NormalizeImage(EnergyDeposit_raw)
        EnergyDeposit = EnergyDeposit/EnergyScale
        ParticleMomentum_ParticlePoint = torch.div(ParticleMomentum_ParticlePoint,torch.cat([self.MomentumScale,self.PointScale]))
        LabelImages = LabelToImages(IMAGE_DIM,IMAGE_DIM,ParticleMomentum_ParticlePoint)
        EnergyDeposit = torch.cat([EnergyDeposit,LabelImages.cuda()],dim=1)
        EnergyDeposit = F.leaky_relu(self.conv1(EnergyDeposit))
        EnergyDeposut = self.dropout(EnergyDeposit)        
        EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposut = self.dropout(EnergyDeposit)
        EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposut = self.dropout(EnergyDeposit)
        EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit))) # 32, 9, 9
        EnergyDeposut = self.dropout(EnergyDeposit)        
        EnergyDeposit = self.conv5(EnergyDeposit) # 32, 9, 9        
        return torch.sigmoid(EnergyDeposit)
