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
    def __init__(self, cond_dim, MomentumPointPDGScale, EnergyScale, Nredconv_dis=3, dropout_fraction=0.5):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1+cond_dim, 16, 4, stride=2)#30->14
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 4)##14->11
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(32, 64, 4)##11->8
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)        
        self.conv4 = nn.Conv2d(64, 128, 3)##8->6
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)                
        # self.conv5 = nn.Conv2d(128, 1, 6)##6->1

        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout = nn.Dropout(p=dropout_fraction)
        self.resblock = ResidualBlock(self.conv4.out_channels)
        self.samesizerc = ReducedConv(128,128,6,6,3)
        
        self.fc1 = nn.Linear(4608,2048)
        self.bn_fc1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features,self.fc1.out_features//2)
        self.bn_fc2 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features,self.fc2.out_features//2)
        self.bn_fc3 = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features,self.fc3.out_features//2)
        self.bn_fc4 = nn.BatchNorm1d(self.fc4.out_features)
        self.fc5 = nn.Linear(self.fc4.out_features,self.fc4.out_features//2)
        self.bn_fc5 = nn.BatchNorm1d(self.fc5.out_features)
        self.fc6 = nn.Linear(self.fc5.out_features,1)
        # self.fc6 = nn.Linear(4608,1)
        
        self.MomentumPointPDGScale = MomentumPointPDGScale
        self.EnergyScale = EnergyScale
        self.Nredconv_dis = Nredconv_dis
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint_ParticlePDG):
        assert EnergyDeposit.shape[2]==30, 'Input Image has wrong size.'
        EnergyDeposit = EnergyDeposit/self.EnergyScale
        ParticleMomentum_ParticlePoint_ParticlePDG = torch.div(ParticleMomentum_ParticlePoint_ParticlePDG,self.MomentumPointPDGScale)
        LabelImages = Label2Image.LabelToImages(EnergyDeposit.shape[2],EnergyDeposit.shape[3],ParticleMomentum_ParticlePoint_ParticlePDG)
        EnergyDeposit = torch.cat([EnergyDeposit,LabelImages],dim=1)
        EnergyDeposit = self.dropout(self.activation(self.conv1(EnergyDeposit)))
        EnergyDeposit = self.dropout(self.activation(self.bn2(self.conv2(EnergyDeposit))))
        EnergyDeposit = self.dropout(self.activation(self.bn3(self.conv3(EnergyDeposit))))
        EnergyDeposit = self.dropout(self.activation(self.bn4(self.conv4(EnergyDeposit)))) # 32, 9, 9
        for ires in range(self.Nredconv_dis):
            EnergyDeposit = self.dropout(self.resblock(EnergyDeposit))
        
        EnergyDeposit = EnergyDeposit.view(EnergyDeposit.shape[0], -1)
        EnergyDeposit = self.dropout(self.activation(self.fc1(EnergyDeposit))) # 32, 9, 9
        EnergyDeposit = self.dropout(self.activation(self.fc2(EnergyDeposit))) # 32, 9, 9
        EnergyDeposit = self.dropout(self.activation(self.fc3(EnergyDeposit))) # 32, 9, 9
        EnergyDeposit = self.dropout(self.activation(self.fc4(EnergyDeposit))) # 32, 9, 9
        EnergyDeposit = self.dropout(self.activation(self.fc5(EnergyDeposit))) # 32, 9, 9
        EnergyDeposit = self.fc6(EnergyDeposit) # 32, 9, 9
        return EnergyDeposit, torch.sigmoid(EnergyDeposit)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()