# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.3 (2024-02-11)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb
# - https://github.com/VincentStimper/normalizing-flows/tree/master

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
import matplotlib.pyplot as plt
from fid import calculate_fid_given_paths

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)
        


    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        
        # The masked coupling layer
        x = z * self.mask + (1 - self.mask) * (z * torch.exp(self.scale_net(self.mask * z)) + self.translation_net(self.mask * z))
        
        # The log determinant of the Jacobian
        log_det_J = torch.sum((1 - self.mask) * (self.scale_net(self.mask * z)), dim=1)
        #log_det_J = torch.zeros(z.shape[0])
        
        return x, log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        
        # The inverse of the masked coupling layer. 
        z = self.mask * x + (1 - self.mask) * (x - self.translation_net(self.mask * x)) * torch.exp(-self.scale_net(self.mask * x))
        
        # The log determinant of the Jacobian of the inverse transformation is the negative of the log determinant of the Jacobian of the forward transformation
        log_det_J = -torch.sum((1 - self.mask) * (self.scale_net(self.mask * z)), dim=1)
        
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, n_samples=1):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        if type(n_samples) is int:
            z = self.base().sample(torch.Size([n_samples]))
        else:
            z = self.base().sample(n_samples)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
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


    for epoch in range(epochs):
        data_iter = iter(data_loader)
        running_loss = 0.0
        batch_itter = 0
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            running_loss += loss.item()
            batch_itter += 1
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        print(f"\nEpoch {epoch+1}/{epochs} loss: {loss.item():.4f} avg. loss: {running_loss/batch_itter:.4f}")


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'plot', 'fid'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--masktype', type=str, default='random', choices=['random', 'chequerboard'], help='masking pattern (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    """ 
    # Generate the data
    n_data = 10000000
    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    train_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
    """
    # Load the MNIST dataset

    mnist_train = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255), 
                    transforms.Lambda(lambda x: x.flatten())
                    ])
                   )
    
    mnist_test = datasets.MNIST('data/', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255), 
                    transforms.Lambda(lambda x: x.flatten())
                    ]) 
                   )
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    #D = next(iter(train_loader)).shape[1]
    D = 28*28
    base = GaussianBase(D)

    # Define transformations
    transformations =[]
    
    # The checkerboard mask
    base_mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
    
    
    num_transformations = 5
    num_hidden = 8

    # Make a mask that is 1 for the first half of the features and 0 for the second half
    #mask = torch.zeros((D,))
    #mask[D//2:] = 1



    for i in range(num_transformations):
        if args.masktype == "random":
            mask = torch.rand(28*28)

        elif args.masktype == "chequerboard":
            if i%2 == 0:
                mask = (1-base_mask) # Flip the mask
            else:
                mask = base_mask
        #scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        # Add tanh activation at the end of the scale_net
        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        #transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        #transformations.append(RandMaskedCouplingLayer(scale_net, translation_net, D))
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    # Define flow model
    model = Flow(base, transformations).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)
        """
    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,))).cpu() 
        # Plot the density of the toy data and the model samples
        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)
        plt.savefig(args.samples)
        plt.close()
    """
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.masktype + "_" + args.samples)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

        # Evaluate model
        model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                loss = model(x)
                print(f'Loss: {loss.item()}')
                break

    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

        # Plot samples from the approximate posterior with their corresponding colour coded by class
        # If latent dimensions is larger than 2, use PCA to reduce dimensionality onto the first two principal components
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                z = model.encoder(x).mean
                break

        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z.cpu().numpy())
        plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y, cmap='tab10')
        plt.colorbar()
        plt.show()

    elif args.mode == 'fid':
        transform = transforms.Compose([
            transforms.Resize(299),  # Resize to Inception v3's input size
            transforms.Grayscale(num_output_channels=3),  # Make sure to have 3 channels like Inception expects
            transforms.ToTensor(),
            ])
        
        real_data = datasets.MNIST('data/', train=True, download=True, transform=transform)

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            sample_data = (model.sample(100)).cpu()
        
        # transform the samples back to the original shape 28*28        
        sample_data = sample_data.view(-1, 1, 28, 28)

        # transform the samples to be consistent with the real data
        sample_data = F.interpolate(sample_data, size=(299, 299), mode='bilinear', align_corners=False)

        sample_data_rgb = sample_data.repeat(1, 3, 1, 1)

        fid_value = calculate_fid_given_paths(sample_data_rgb, real_data, 32, args.device, dims=64)
        print(f"FID: {fid_value}")