# https://github.com/pytorch/examples/blob/main/vae/main.py

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from libs.model import VAE
from libs.utils import loss_fnuction
from libs.utils import save_exp_result
from libs.utils import plot_loss_figure
from libs.utils import generation_vae_image



# ========== Training ==========

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    
    train_loss = 0.0
    for data, _ in dataloader:
        data = data.to(device)
        data = torch.reshape(data, shape=(-1, 784))
        optimizer.zero_grad()

        recon_batch, mu, z_log_var = model(data)
        loss = loss_fn(recon_batch, data, mu, z_log_var)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(dataloader)
    return model, train_loss


def test(model, dataloader, loss_fn, device):
    model.eval()
    test_losses = 0.0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            data = torch.reshape(data, shape=(-1, 784))


            recon_batch, mu, z_log_var = model(data)
            loss = loss_fn(recon_batch, data, mu, z_log_var)

            test_losses += loss.item()
    test_losses /= len(dataloader)
    return test_losses


# ========== Training ==========

def main(args, device):

    # define the dataset and dataloader 

    train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    # train_dataset.data = train_dataset.data / 255.0
    # test_dataset.data = test_dataset.data / 255.0

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # define the model and training paramters

    model = VAE(args.hidden_dim, args.latent_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = loss_fnuction
    train_losses = []
    test_losses = []
    for epoch in range(args.epochs):
        ts = time.time()
        model, train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        test_loss = test(model, test_dataloader, loss_fn, device)
        te = time.time()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print('Epoch {}. LOSS(train/test) = ({:.2f}/{:.2f}), TIME={}'.format(
            epoch+1, train_loss, test_loss, te-ts
        ))

    result = {}
    result['train_losses'] = train_losses
    result['test_losses'] = test_losses
    return model, vars(args), result


if __name__ == '__main__':

    # Hyperparameters
    parser = argparse.ArgumentParser(description='VAE MNIST')
    parser.add_argument('--exp_name', default='VAE_basic_traning')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epochs', default=10)

    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--latent_dim', default=16)

    args = parser.parse_args()

    # SEED
    args.seed = 123
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.exp_name = 'VAE_model'
    args.latend_dim = 32
    args.epochs = 10

    ## device
    # if torch.cuda.is_available():
    #     device = torch.device('cpu')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')

    device = torch.device('cpu')

    model, setting, results = main(args, device)
    save_exp_result(model, setting, results)

    plot_loss_figure('./results/{}.png'.format(setting['exp_name']), results, figsize=(10, 3))
    
    output = generation_vae_image(model, 8, args.latent_dim, device)
    plt.figure(figsize=(8, 4))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(output[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show();
    plt.savefig('./results/sample_vae_gen_img.png')

