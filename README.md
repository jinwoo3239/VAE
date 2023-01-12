# Basic VAE tutorial

This is an code implement of the paper [Auto-Encoding Variatioinal Bayes](https://arxiv.org/abs/1312.6114) and Codes are referenced in the [github of pytorch_example](https://github.com/pytorch/examples/tree/main/vae). The shared codes were written based on that reference, and they were slightly modified.


# Basic theory

```bash
def loss_function(x_recon, x, mu, z_log_var):
    '''
    Params
    ______
    x_recon: torch.Tensor, Reconstructed input with values 0~1
    x: torch.Tensor, Input

    mu: mean of latent variables
    z_log_var: log-variance of latent variables
    '''

    # Reconstruction loss
    # Assumption... Decoder has a bernoulli Distr.
    BCE = nn.BCELoss(reduction='sum')
    bce_loss = BCE(x_recon, x)

    # Regularization (KL Divergence)
    # Assumption... prior ~ N(0, I), posterior ~ Gaussian.
    kle_loss = 0.5 * torch.sum(mu**2 + torch.exp(z_log_var) - z_log_var +1)

    return bce_loss + z_log_var
```




# Getting Started

## Prerequisites
* `python` = 3.10.6
* [`pytorch`](https://pytorch.org) = 1.12.1
* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `argparse`


 The main.py script accepts the following arguments
```bash
optional arguments:
    --exp_name
    --batch_size
    --epochs
    --hidden_dim
    --latent_dim
```

