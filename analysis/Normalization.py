import torch

def GetNormalizedMomentumPoint(MomentumPoint,MomentumScale,PointScale):
    MomentumPoint_norm = torch.div(ParticleMomentum_ParticlePoint,torch.cat([MomentumScale,PointScale]))+0.5
    MomentumPoint_norm[:,:,2] -= 0.5
    return MomentumPoint_norm