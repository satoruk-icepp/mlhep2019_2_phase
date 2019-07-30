import torch
import torch.nn as nn
import torch.nn.functional as F
import Label2Image
from NetworkUtil import ReducedConv,ResidualBlock
import Normalization
# NOISE_DIM = 10


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()

class ModelD(nn.Module):
    def __init__(self, cond_dim, MomentumPointPDGScale, EnergyScale, Nredconv_dis=3):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1+cond_dim, 16, 4, stride=2)#30->14
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 4)##14->11
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4)##11->8
        self.bn3 = nn.BatchNorm2d(64)        
        self.conv4 = nn.Conv2d(64, 128, 3)##8->6
        self.bn4 = nn.BatchNorm2d(128)                
        self.conv5 = nn.Conv2d(128, 1, 6)##6->1

        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.resblock = ResidualBlock(128)
        self.samesizerc = ReducedConv(128,128,6,6,3)
        self.MomentumPointPDGScale = MomentumPointPDGScale
        # self.PointScale = PointScale
        self.EnergyScale = EnergyScale
        self.Nredconv_dis = Nredconv_dis
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint_ParticlePDG):
#         EnergyDeposit = NormalizeImage(EnergyDeposit_raw)
        assert EnergyDeposit.shape[2]==30, 'Input Image has wrong size.'
        EnergyDeposit = EnergyDeposit/self.EnergyScale
        # ParticleMomentum_ParticlePoint = torch.div(ParticleMomentum_ParticlePoint,torch.cat([self.MomentumScale,self.PointScale]))
        # ParticleMomentum_ParticlePoint = GetNormalizedMomentumPoint(ParticleMomentum_ParticlePoint,self.MomentumScale,self.PointScale)
        ParticleMomentum_ParticlePoint_ParticlePDG = torch.div(ParticleMomentum_ParticlePoint_ParticlePDG,self.MomentumPointPDGScale)
        LabelImages = Label2Image.LabelToImages(EnergyDeposit.shape[2],EnergyDeposit.shape[3],ParticleMomentum_ParticlePoint_ParticlePDG)
        EnergyDeposit = torch.cat([EnergyDeposit,LabelImages],dim=1)
        EnergyDeposit = self.activation(self.conv1(EnergyDeposit))
        EnergyDeposut = self.dropout(EnergyDeposit)        
        EnergyDeposit = self.activation(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposut = self.dropout(EnergyDeposit)
        EnergyDeposit = self.activation(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposut = self.dropout(EnergyDeposit)
        EnergyDeposit = self.activation(self.bn4(self.conv4(EnergyDeposit))) # 32, 9, 9
        EnergyDeposut = self.dropout(EnergyDeposit)
        for ires in range(self.Nredconv_dis):
            # EnergyDeposit = self.samesizerc(EnergyDeposit)
            EnergyDeposit = self.resblock(EnergyDeposit)
        EnergyDeposit = self.conv5(EnergyDeposit) # 32, 9, 9
        EnergyDeposit = EnergyDeposit.view(EnergyDeposit.shape[0], -1)                
        return torch.sigmoid(EnergyDeposit)

    def weight_init(self, mean, std):
                for m in self._modules:
                    normal_init(self._modules[m], mean, std)