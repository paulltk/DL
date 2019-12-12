import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(nn.Linear(args.latent_dim, 128),
                              nn.LeakyReLU(0.2),
                              nn.Linear(128, 256),
                              nn.BatchNorm1d(256),
                              nn.LeakyReLU(0.2),
                              nn.Linear(256, 512),
                              nn.BatchNorm1d(512),
                              nn.LeakyReLU(0.2),
                              nn.Linear(512, 1024),
                              nn.BatchNorm1d(1024),
                              nn.LeakyReLU(0.2),
                              nn.Linear(1024, 784),
                              nn.Tanh())

        self.model.to(device)

    def forward(self, z):

        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(nn.Linear(784, 512),
                              nn.LeakyReLU(0.2),
                              nn.Linear(512, 256),
                              nn.LeakyReLU(0.2),
                              nn.Linear(256, 1),
                              nn.Sigmoid())

        self.model.to(device)

    def forward(self, img):

        return self.model(img)

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # Setting loss to binary cross-entropy
    loss_function = nn.BCELoss().to(device)

    for epoch in range(args.n_epochs):
        print("Epoch", epoch)
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.view(-1, 784).to(device)

            # Train Discriminator
            # -------------------

            # generate images
            z = torch.randn(size=(imgs.shape[0], args.latent_dim)).to(device)
            gen_imgs = generator(z).detach()

            # predict for generated images
            gen_labels = torch.zeros(imgs.shape[0], 1).to(device)
            predictions = discriminator(gen_imgs)
            loss = loss_function(predictions, gen_labels)

            # predict for real images and calculate gradients
            real_labels = torch.ones(imgs.shape[0], 1).to(device)
            predictions = discriminator(imgs)
            loss += loss_function(predictions, real_labels)

            # calculate gradients and update weights
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            # Train Generator
            # ---------------

            # generate images
            z = torch.randn(size=(imgs.shape[0], args.latent_dim)).to(device)
            gen_imgs = generator(z)

            # predict and calculate gradients for generated images
            predictions = discriminator(gen_imgs)
            loss = loss_function(predictions, real_labels)

            # calculate gradients and update weights
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                gen_imgs = gen_imgs.view(gen_imgs.size()[0], 1, 28, 28)
                save_image(gen_imgs[:25], 'images/{}.png'.format(batches_done), nrow=5, normalize=True)
                pass

            if epoch == args.n_epochs - 1 and i % 10 == 0:
                z1 = torch.randn(1, args.latent_dim).to(device)
                z2 = torch.randn(1, args.latent_dim).to(device)
                diff = (z1 - z2) / 8
                z = z1

                for inter in range(8):
                    new_z = z1 - diff * (inter + 1)
                    z = torch.cat((z, new_z))

                inter_imgs = generator(z)
                inter_imgs = inter_imgs.view(inter_imgs.size()[0], 1, 28, 28)
                save_image(inter_imgs[:9], 'images/interpolated{}.png'.format(batches_done), nrow=9, normalize=True)


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    print("Creating models")
    generator = Generator()
    discriminator = Discriminator()
    print("Creating optimizers")
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    print("starting training")
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
