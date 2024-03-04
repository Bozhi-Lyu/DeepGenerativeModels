# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
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
import numpy as np
import seaborn as sns
sns.set_theme()


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

class MoGPrior(nn.Module):
    def __init__(self, M, num_components = 10):
        """
        Define a mixture of Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] Dimension of the latent space.
        num_components: [int] Number of Gaussian components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.num_components = num_components

        self.means = nn.Parameter(torch.rand(self.num_components, self.M))
        self.log_stds = nn.Parameter(torch.randn(self.num_components, self.M))
        self.weights = nn.Parameter(torch.ones(self.num_components))
        self.softmax = nn.Softmax(dim=0)
        # print("weights.size(): ", self.weights.size())

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        stds = torch.exp(self.log_stds)
        weights = self.softmax(self.weights)
        mix = td.Categorical(probs=weights)
        comp = torch.distributions.Independent( td.Normal(self.means, stds), 1)
        prior = td.MixtureSameFamily(mix, comp)

        return prior


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
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

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
        z = q.rsample()
        if isinstance(self.prior, GaussianPrior):
            KL_qp = td.kl_divergence(q, self.prior())
        else:
            KL_qp = q.log_prob(z) - self.prior().log_prob(z)
        elbo = torch.mean(self.decoder(z).log_prob(x) - KL_qp, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    losses = []

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
    
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob
    torch.manual_seed(1234)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'plot1', 'plot2'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model_sg_M2.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='model_sg_M2.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='sg', choices=['sg', 'MoG'])
    
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'sg':
        prior = GaussianPrior(M)
    elif args.prior == 'MoG':
        prior = MoGPrior(M)

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
    model = VAE(prior, decoder, encoder).to(device)

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
        model.eval()
        elbos = []
        sum_elbo = 0

        with torch.no_grad():
            for batch in mnist_test_loader:
                images = batch[0]
                elbo = model.elbo(images)
                elbos.append(elbo)
                sum_elbo += elbo.item()

        print("Averaged elbo:", sum_elbo/(len(elbos)))

    elif args.mode == 'plot1':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        zlist = []
        labellist = []
        merged_data = []

        with torch.no_grad():
            for batch in mnist_test_loader:
                merged_data.append(batch[0])

                t = model.encoder(batch[0])
                samples = t.sample()
                zlist.append(samples)
                labellist.append(batch[1])
            
        z = torch.cat(zlist)
        merged_data = torch.cat(merged_data, dim=0)

        print("z.size(): ", z.size())
        print("merged_data.size(): ", merged_data.size())

        labels = torch.cat(labellist)
        pca = PCA(n_components=2)
        reduced_z = pca.fit_transform(z)
        print("reduced_z.size(): ", reduced_z.shape)

        z1 = reduced_z[:, 0]
        z2 = reduced_z[:, 1]

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange', 'gray']
        for i in range(10):
            plt.scatter(z1[labels == i], z2[labels == i], c=colors[i], label=str(i), s=3)
        plt.legend()

        plt.savefig("plot1_" + args.samples)       

    elif args.mode == 'plot2':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # zlist = []
        labellist = []
        merged_data = []
        plot_range = [-20, 20]

        with torch.no_grad():
            for batch in mnist_test_loader:
                merged_data.append(batch[0])

            merged_data = torch.cat(merged_data, dim=0)

            Posteriordist = model.encoder.forward(merged_data).mean
            x_post, y_post = Posteriordist[:, 0], Posteriordist[:, 1]

            x = torch.linspace(plot_range[0], plot_range[1], 200)
            y = torch.linspace(plot_range[0], plot_range[1], 200)
            X, Y = torch.meshgrid(x, y)
            coordi = torch.stack((X.flatten(), Y.flatten()), dim=1).reshape(-1, 2)
            print("coordi.size():", coordi.size())        
            print("x_post.size(): ", x_post.size())
            print("Posteriordist.size(): ", Posteriordist.size())

            X = coordi[:, 0]
            Y = coordi[:, 1]

            prob_density = model.prior.forward().log_prob(
            torch.tensor(np.column_stack((X, Y)))).reshape(x.size(0), -1)        


        print("prob_density:", prob_density.size())

        

        plt.contourf(x.detach().numpy(), y.detach().numpy(), prob_density.detach().numpy())  
        plt.scatter(x_post.detach().numpy(), y_post.detach().numpy(), s=1, label='Posterior Samples')
        plt.colorbar()
        plt.xlabel('M1')
        plt.ylabel('M2')
        plt.xlim([plot_range[0], plot_range[1]])
        plt.ylim([plot_range[0], plot_range[1]])
        plt.savefig("plot2_" + args.samples)


