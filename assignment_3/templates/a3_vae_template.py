import argparse

import torch
import torch.nn as nn
import torch.distributions
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from datasets.bmnist import bmnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.h = nn.Linear(784, hidden_dim).to(device)
        self.z_mean = nn.Linear(hidden_dim, z_dim).to(device)
        self.z_std = nn.Linear(hidden_dim, z_dim).to(device)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        # print("input size:", input.size())
        hidden = self.h(input).relu()
        # print("encode hidden:", hidden.size())

        mean = self.z_mean(hidden)
        # print("mean hidden:", mean.size())

        std = self.z_std(hidden)
        # print("std hidden:", std.size())

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.h = nn.Linear(z_dim, hidden_dim).to(device)

        self.output = nn.Linear(hidden_dim, 784).to(device)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        # print("decode input:", input.size())
        hidden = self.h(input).relu()
        # print("decode hidden:", hidden.size())
        mean = self.output(hidden).sigmoid()
        # print("decode mean:", mean.size())

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, std = self.encoder(input)

        std = std.exp().sqrt()

        z = mean + std * torch.randn(size=(input.size()[0], self.z_dim)).to(device)

        out = self.decoder(z)

        reconstruction = nn.functional.binary_cross_entropy(out, input, reduction='sum')

        regularization = - 0.5 * torch.sum(1 + std - mean**2 - std.exp())

        return (reconstruction + regularization) / input.shape[0]

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        im_means = self.decoder(torch.randn(size=(n_samples, self.z_dim)).to(device))
        sampled_ims = im_means.bernoulli()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    average_epoch_elbo = 0

    for batch in data:
        batch = batch.view(-1, 784)

        loss = model(batch)

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_epoch_elbo += loss

    return average_epoch_elbo / batch.shape[0] / len(data)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):

        if epoch == 0:
            samples = model.sample(9)[0]
            save_image(samples.view(9, 1, 28, 28), "output_vae_epoch_{}.png".format(epoch), nrow=3)

        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        if epoch == ARGS.epochs/2 or epoch == ARGS.epochs - 1:
            samples = model.sample(9)[0]
            save_image(samples.view(9, 1, 28, 28), "output_vae_epoch_{}.png".format(epoch), nrow=3)
        
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
