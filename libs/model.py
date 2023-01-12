import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Model define ==========

# args.hidden_dim
# args.latent_dim

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.flatten = nn.Flatten()

        self.encoder_fc1 = nn.Linear(784, hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_z_log_v = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, 784)

    def encode(self, inputs):
        x = self.flatten(inputs) # x = (batch_size, 28 * 28)
        x = F.relu(self.encoder_fc1(x))

        # mu, z_log_var = (batch_size, latent_dim)
        mu = self.encoder_mean(x)
        z_log_var = self.encoder_z_log_v(x) 
        return mu, z_log_var


    def reparameterize(self, mu, z_log_var):
        # z = mu + std * N(0, I)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = F.relu(self.decoder_fc1(z))
        x = torch.sigmoid(self.decoder_fc2(x))
        return x

    def forward(self, inputs):
        mu, z_log_var = self.encode(inputs)
        z = self.reparameterize(mu, z_log_var)
        output = self.decode(z)
        return output, mu, z_log_var



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
