#!/usr/bin/python
import sys
import numpy as np
from analysis.prd_score import compute_prd, compute_prd_from_embedding, _prd_to_f_beta
from prd import plot_pr_aucs,calc_pr_rec
from sklearn.metrics import auc
from generator import ModelGConvTranspose
from Regressor import Regressor,load_embedder
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import matplotlib.pyplot as plt
import os

NOISEIMAGE_DIM=10
NOISE_DIM=NOISEIMAGE_DIM**2
EnergyDepositScale = torch.tensor([4000]).float()
MomentumScale      = torch.tensor([30,30,100]).float()
PointScale         = torch.tensor([10,10]).float()
PDGScale         = torch.tensor([11]).float()
MomentumPointPDGScale = torch.cat([MomentumScale,PointScale,PDGScale])

def main():
    input_dir, output_dir = sys.argv[1:]
    embedder = load_embedder('./analysis/embedder.tp')
    
    # data_val = np.load(input_dir + '/data_val.npz', allow_pickle=True)
    # val_data_path_out = output_dir + '/data_val_prediction.npz'
    # 
    # data_test = np.load(input_dir + '/data_test.npz', allow_pickle=True)
    # test_data_path_out = output_dir + '/data_test_prediction.npz'
    
    data_train = np.load(input_dir + '/data_train.npz', allow_pickle=True)
    # test_data_path_out = output_dir + '/data_test_prediction.npz'
    
    generator_cpu = ModelGConvTranspose(z_dim=NOISEIMAGE_DIM, MomentumPointPDGScale = MomentumPointPDGScale,EnergyScale = EnergyDepositScale)
    # generator_cpu.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/gan_80.pt'))
    generator_cpu.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/gan_20.pt'))
    # generator_cpu.eval()
    
    # val
    data_size = 250
    EnergyDeposit_train     = torch.tensor(data_train['EnergyDeposit'][:data_size].reshape(-1,1,30,30)).float()
    ParticleMomentum_train  = torch.tensor(data_train['ParticleMomentum'][:data_size]).float()
    ParticlePoint_train     = torch.tensor(data_train['ParticlePoint'][:data_size, :2]).float()
    ParticlePDG_train       = torch.tensor(data_train['ParticlePDG'][:data_size].reshape(-1,1)).float()
    ParticleMomentum_ParticlePoint_ParticlePDG_train = torch.cat([ParticleMomentum_train, ParticlePoint_train, ParticlePDG_train], dim=1)
    calo_dataset_train = utils.TensorDataset(EnergyDeposit_train,ParticleMomentum_ParticlePoint_ParticlePDG_train)
    calo_dataloader_train = torch.utils.data.DataLoader(calo_dataset_train, batch_size=data_size, shuffle=False)

    with torch.no_grad():
        EnergyDeposit_train = []
        EnergyDeposit_train_truth = []
        for EnergyDeposit_train_batch,ParticleMomentum_ParticlePoint_ParticlePDG_train_batch in tqdm(calo_dataloader_train):
            noise = torch.randn(ParticleMomentum_ParticlePoint_ParticlePDG_train_batch.shape[0], NOISE_DIM)
            # print(ParticleMomentum_ParticlePoint_ParticlePDG_train_batch.shape)
            EnergyDeposit_train_gen_batch = generator_cpu(noise, ParticleMomentum_ParticlePoint_ParticlePDG_train_batch)
            EnergyDeposit_train.append(EnergyDeposit_train_gen_batch)
            data_real = embedder.get_encoding(torch.tensor(EnergyDeposit_train_batch).float().view(-1, 1, 30, 30)).detach().numpy()
            data_fake = embedder.get_encoding(torch.tensor(EnergyDeposit_train_gen_batch).float().view(-1, 1, 30, 30)).detach().numpy()
            precisions, recalls = calc_pr_rec(data_real, data_fake, num_clusters=100, num_runs=20)
            pr_aucs = plot_pr_aucs(precisions, recalls)
            plt.title('Num_clusters={}, num_runs={}, first third'.format(100, 20))
            plt.show()

    return 0

if __name__ == "__main__":
    main()
