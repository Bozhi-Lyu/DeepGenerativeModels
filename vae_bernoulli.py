# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from flow import Flow, MaskedCouplingLayer


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MixedGaussianPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
            Number of components in the mixture.
        """
        super(MixedGaussianPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.K, self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(torch.ones(self.K,))
        comp = td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
        return td.MixtureSameFamily(mix, comp)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, type=None, device="cpu"):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        device: [str]
            The device to run the network on
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.type = type
        self.device = device

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample().to(self.device)  # Reparameterization trick

        # MOG manually compute KL divergence
        if self.type in ["mix", "flow"]:
            if self.type == "mix":
                log_prob_prior = self.prior().log_prob(z)
            elif self.type == "flow":
                log_prob_prior = self.prior.log_prob(z)
            log_prob_q = q.log_prob(z)
            kl = log_prob_q - log_prob_prior
            elbo = torch.mean(self.decoder(z).log_prob(x) - kl, dim=0)

        else:
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)

        #print(f'x: {x.shape}, z: {z.shape}, elbo: {elbo.shape}')
        #print(f'part1: {self.decoder(z).log_prob(x).shape}, part2: {td.kl_divergence(q, self.prior()).shape}')
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        if self.type == "flow":
            z = self.prior.sample(torch.Size([n_samples]))
        else:
            z = self.prior().sample(torch.Size([n_samples]))

        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    losses = []

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        running_loss = 0.0
        batch_itter = 0
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            running_loss += loss.item()
            batch_itter += 1
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), epoch=f"{epoch + 1}/{epochs}")
            progress_bar.update()
        print(f"\nEpoch {epoch+1}/{epochs} loss: {loss.item():.4f} avg. loss: {running_loss/batch_itter:.4f}")

    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default="gaussian", choices=['gaussian', 'mix', 'flow'], help='dimension of latent variable (default: %(default)s)')
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (thresshold < x).float().squeeze())
            ])),
                                                     batch_size=args.batch_size,
                                                     shuffle=True
                                                     )
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Load MNIST as grayscale and create data loaders
    #mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
    #                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
    #                                                batch_size=args.batch_size, shuffle=True)
    #mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
    #                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
    #                                                batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    K = 5
    if args.prior == "gaussian":
        prior = GaussianPrior(M)
    elif args.prior =="mix":
        prior = MixedGaussianPrior(M, K)
    elif args.prior == "flow":
        D = 28 * 28
        M=D
        base = GaussianPrior(D)
        # Define transformations
        transformations = []
        num_transformations = 5
        num_hidden = 8

        for i in range(num_transformations):
            mask = torch.rand(28 * 28)
            scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        prior = Flow(base, transformations).to(device)


    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    # Change to mog_prior to use mixture of gaussians
    model = VAE(prior, decoder, encoder, type=args.prior, device=device).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Evaluate model
        model.eval()
        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                loss = model(x)
                print(f'Loss: {loss.item()}')
                break

    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Plot samples from the approximate posterior with their corresponding colour coded by class
        # If latent dimensions is larger than 2, use PCA to reduce dimensionality onto the first two principal components
        model.eval()
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).mean
                break

        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z.cpu().numpy())
        plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y, cmap='tab10')
        plt.colorbar()
        plt.show()



