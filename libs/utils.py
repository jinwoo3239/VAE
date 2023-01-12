import os

import torch
import torch.nn as nn
from torchvision.utils import save_image
import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ========== Loss function ==========

def loss_fnuction(x_recon, x, mu, z_log_var):
    '''
    Params
    x_recon = reconstructed x
    x = label x (input)
    '''
    bce_fn = nn.BCELoss(reduction='sum')
    bce_loss = bce_fn(x_recon, x)

    kld_loss = 0.5*torch.sum(mu**2 + torch.exp(z_log_var) - z_log_var -1)
    return bce_loss + kld_loss


# ========== Model&Result save&load ==========

def save_exp_result(model, setting:dict, result:dict):
    exp_name = setting['exp_name']

    result.update(setting)
    filename = f'./results/{exp_name}_{datetime.datetime.now()}'

    with open(filename + '.json', 'w') as f:
        json.dump(result, f)

    torch.save(model, filename + '.pt')




def plot_loss_figure(path, results, figsize=(5, 3)):
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(results['train_losses'])+1), results['train_losses'], marker='o', label='train')
    plt.plot(range(1, len(results['test_losses'])+1), results['test_losses'], marker='v', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    plt.legend()
    plt.savefig(path)


def generation_vae_image(model, batch_size, latent_dim, device):
    model.eval()
    with torch.no_grad():
        sample = torch.randn((batch_size, latent_dim)).to(device)
        output = model.decode(sample)
        output = torch.reshape(output, (batch_size, 28, 28)).cpu()
        return output

        

