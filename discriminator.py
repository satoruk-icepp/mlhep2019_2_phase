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
    def __init__(self, cond_dim, MomentumPointPDGScale,MomentumPointPDGOffset, EnergyScale,EnergyOffset, Nredconv_dis=3, dropout_fraction=0.5, use_bn = True):
        super(ModelD, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(1+cond_dim, 16, 4, stride=2, padding=2)#30->16
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.ln1 = nn.LayerNorm([self.conv1.out_channels,16,16])
        
        self.conv2 = nn.Conv2d(16, 32, 5)##16->12
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.ln2 = nn.LayerNorm([self.conv2.out_channels,12,12])
        
        self.conv3 = nn.Conv2d(32, 64, 5)##12->8
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)        
        self.ln3 = nn.LayerNorm([self.conv3.out_channels,8,8])
        
        self.conv4 = nn.Conv2d(64, 128, 3)##8->6
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)                
        self.ln4 = nn.LayerNorm([self.conv4.out_channels,6,6])
        # self.conv5 = nn.Conv2d(128, 1, 6)##6->1

        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout = nn.Dropout(p=dropout_fraction)
        self.resblock = ResidualBlock(self.conv4.out_channels,use_bn)
        self.samesizerc = ReducedConv(128,128,6,6,3)
        
        self.fc1 = nn.Linear(4608+5,2048)
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
        self.MomentumPointPDGOffset = MomentumPointPDGOffset
        self.EnergyScale = EnergyScale
        self.EnergyOffset   = EnergyOffset
        self.Nredconv_dis = Nredconv_dis
        self.indices = torch.tensor([float(i) for i in range(30)]).cuda()
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint_ParticlePDG):
        assert EnergyDeposit.shape[2]==30, 'Input Image has wrong size.'
        EnergyDeposit_1d = EnergyDeposit.view(EnergyDeposit.shape[0],900)
        # MaxElement = torch.argmax(EnergyDeposit_1d,dim=1)
        SumElement = torch.sum(EnergyDeposit_1d,dim=1)
        SumElement = SumElement.view(EnergyDeposit.shape[0],1)
        ## project image and extract mean and variance
        XProj = EnergyDeposit.sum(dim=2).view(EnergyDeposit.shape[0],-1)
        YProj = EnergyDeposit.sum(dim=3).view(EnergyDeposit.shape[0],-1)
        XMean = torch.zeros(EnergyDeposit.shape[0],1).cuda()
        XVar  = torch.zeros(EnergyDeposit.shape[0],1).cuda()
        YMean = torch.zeros(EnergyDeposit.shape[0],1).cuda()
        YVar  = torch.zeros(EnergyDeposit.shape[0],1).cuda()
        if SumElement[i]<100:
            XMean[i] = 0
            XVar[i]  = 1
            YMean[i] = 0
            YVar[i]  = 1
        
        for i in range(EnergyDeposit.shape[0]):
            XMean[i] = ((self.indices * XProj[i]).sum()/SumElement[i]-14.5)/15.0
            XVar[i]  = XProj[i].norm()/SumElement[i]

        for i in range(EnergyDeposit.shape[0]):
            YMean[i] = ((self.indices * YProj[i]).sum()/SumElement[i]-14.5)/15.0
            YVar[i]  = YProj[i].norm()/SumElement[i]
        SumElement = SumElement/self.EnergyScale
        AdditionalProperties = torch.cat([SumElement,XMean,XVar,YMean,YVar],dim=1)
        
        EnergyDeposit = torch.div(EnergyDeposit-self.EnergyOffset,self.EnergyScale)
        ParticleMomentum_ParticlePoint_ParticlePDG = torch.div(ParticleMomentum_ParticlePoint_ParticlePDG-self.MomentumPointPDGOffset,self.MomentumPointPDGScale)
        ParticleMomentum_ParticlePoint_ParticlePDG = Label2Image.LabelToImages(EnergyDeposit.shape[2],EnergyDeposit.shape[3],ParticleMomentum_ParticlePoint_ParticlePDG)
        EnergyDeposit = torch.cat([EnergyDeposit,ParticleMomentum_ParticlePoint_ParticlePDG],dim=1)
        EnergyDeposit = self.dropout(self.activation(self.conv1(EnergyDeposit)))
        if self.use_bn:
            EnergyDeposit = self.dropout(self.activation(self.bn2(self.conv2(EnergyDeposit))))
            EnergyDeposit = self.dropout(self.activation(self.bn3(self.conv3(EnergyDeposit))))
            EnergyDeposit = self.dropout(self.activation(self.bn4(self.conv4(EnergyDeposit)))) # 32, 9, 9
            for ires in range(self.Nredconv_dis):
                EnergyDeposit = self.dropout(self.resblock(EnergyDeposit))
        else:
            EnergyDeposit = self.dropout(self.activation(self.ln2(self.conv2(EnergyDeposit))))
            EnergyDeposit = self.dropout(self.activation(self.ln3(self.conv3(EnergyDeposit))))
            EnergyDeposit = self.dropout(self.activation(self.ln4(self.conv4(EnergyDeposit)))) # 32, 9, 9
            for ires in range(self.Nredconv_dis):
                EnergyDeposit = self.dropout(self.resblock(EnergyDeposit))   
        
        EnergyDeposit = EnergyDeposit.view(EnergyDeposit.shape[0], -1)
        AdditionalProperties = AdditionalProperties.view(EnergyDeposit.shape[0], -1)
        EnergyDeposit = torch.cat([EnergyDeposit,AdditionalProperties],dim=1)
        # EnergyDeposit = torch.cat([EnergyDeposit,ParticleMomentum_ParticlePoint_ParticlePDG],dim=1)
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
                    