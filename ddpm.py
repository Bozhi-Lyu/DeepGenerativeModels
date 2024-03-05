# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import unet 

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        
        batch_size, input_dim = x.shape
        
        # Sample t ~ Uniform({1, ..., T})
        t = torch.randint(1, self.T + 1, (batch_size, 1)) # size (batch_size, 1)
        
        alpha_t = self.alpha_cumprod[t-1] # size (batch_size, 1)
        
        # Sample epsilon ~ N(0, I)
        epsilon = torch.randn_like(x) # size (batch_size, input_dim)
        
        # Compute x_t
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * epsilon # size (batch_size,)
        
        # Predict epsilon_0 (noise prediction)
        epsilon_0_pred = self.network(x_t, t-1)
        
        # Negative ELBO, sum over the input dimensions representing the nefative ELBO for each data point in the batch
        neg_elbo = ((epsilon - epsilon_0_pred)**2).sum(dim=1) #size (batch_size,)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        
        
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)
        
        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, 0, -1):
            ### Implement the remaining of Algorithm 2 here ###
            t_tensor = torch.full((shape[0],1), t, dtype=torch.float).to(self.alpha.device)
            z = torch.randn(shape).to(self.alpha.device) if t > 1 else torch.zeros(shape).to(self.alpha.device)
            sigma_t = torch.sqrt(self.beta[t]).unsqueeze(0) 
            epsilon_theta = self.network(x_t, t_tensor)
            x_t = (1 / torch.sqrt(self.alpha[t])) * (x_t - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon_theta) + sigma_t * z
        
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)

            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.mean()}")
    
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'sample-minst', 'fid'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    
    if args.data == 'mnist':
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x + torch.rand(x.shape)/255), 
                                  transforms.Lambda(lambda x: (x-0.5)*2.0), transforms.Lambda(lambda x: x.flatten())]
                                )
        train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data/', train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        # Get the dimension of the dataset
        D = next(iter(train_loader))[0].shape[1]
    else:
        # Generate the data
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        # Get the dimension of the dataset
        D = next(iter(train_loader)).shape[1] # torch.Size([10000, 2])

    # Define the network
    num_hidden = 64
    #network = FcNetwork(D, num_hidden)
    network = unet.Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,D))).cpu() 

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

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
    
    elif args.mode == 'sample-mnist':
        import matplotlib.pyplot as plt
        import numpy as np
        from torchvision.utils import save_image

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((100,D))).cpu() 
        
        # transform the samples back to the original shape 28*28        
        samples = (samples + 1) / 2
        samples = samples.view(-1, 1, 28, 28)
        
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axs.flat):  # axs.flat is used to iterate over a 2D array as if it's flat
            if i < len(samples):  # Check to avoid IndexError in case the number of samples is less than 15
                ax.imshow(samples[i].squeeze(), cmap='gray')  # Display the ith sample
                ax.axis('off')  # Hide the axis

        plt.savefig(args.samples)
        plt.close()
    
    elif args.mode == 'fid':
        import numpy as np
        import torchvision.transforms as TF
        from scipy import linalg
        from torch.nn.functional import adaptive_avg_pool2d
        from pytorch_fid.inception import InceptionV3
        # Revised code from https://github.com/mseitzer/pytorch-fid?tab=readme-ov-file

        def get_activations(dataset, model, batch_size=50, dims=2048, device='cpu',
                            num_workers=1):
            """Calculates the activations of the pool_3 layer for all images.

            Params:
            -- files       : List of image files paths
            -- model       : Instance of inception model
            -- batch_size  : Batch size of images for the model to process at once.
                            Make sure that the number of samples is a multiple of
                            the batch size, otherwise some samples are ignored. This
                            behavior is retained to match the original FID score
                            implementation.
            -- dims        : Dimensionality of features returned by Inception
            -- device      : Device to run calculations
            -- num_workers : Number of parallel dataloader workers

            Returns:
            -- A numpy array of dimension (num images, dims) that contains the
            activations of the given tensor when feeding inception with the
            query tensor.
            """
            model.eval()

            if batch_size > len(dataset):
                print(('Warning: batch size is bigger than the data size. '
                    'Setting batch size to data size'))
                batch_size = len(dataset)

            #dataset = ImagePathDataset(files, transforms=TF.ToTensor())
            dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=num_workers)

            pred_arr = np.empty((len(dataset), dims))

            start_idx = 0

            for batch in tqdm(dataloader):
                if torch.is_tensor(batch):
                    batch = batch.to(device)
                else:
                    batch = batch[0].to(device)
                with torch.no_grad():
                    pred = model(batch)[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[start_idx:start_idx + pred.shape[0]] = pred

                start_idx = start_idx + pred.shape[0]

            return pred_arr


        def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
            """Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            Stable version by Dougal J. Sutherland.

            Params:
            -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
            -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
            -- sigma1: The covariance matrix over activations for generated samples.
            -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.

            Returns:
            --   : The Frechet Distance.
            """

            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)

            sigma1 = np.atleast_2d(sigma1)
            sigma2 = np.atleast_2d(sigma2)

            assert mu1.shape == mu2.shape, \
                'Training and test mean vectors have different lengths'
            assert sigma1.shape == sigma2.shape, \
                'Training and test covariances have different dimensions'

            diff = mu1 - mu2

            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
                print(msg)
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)
            
            return (diff.dot(diff) + np.trace(sigma1)
                    + np.trace(sigma2) - 2 * tr_covmean)


        def calculate_activation_statistics(dataset, model, batch_size=50, dims=2048,
                                            device='cpu', num_workers=1):
            """Calculation of the statistics used by the FID.
            Params:
            -- files       : List of image files paths
            -- model       : Instance of inception model
            -- batch_size  : The images numpy array is split into batches with
                            batch size batch_size. A reasonable batch size
                            depends on the hardware.
            -- dims        : Dimensionality of features returned by Inception
            -- device      : Device to run calculations
            -- num_workers : Number of parallel dataloader workers

            Returns:
            -- mu    : The mean over samples of the activations of the pool_3 layer of
                    the inception model.
            -- sigma : The covariance matrix of the activations of the pool_3 layer of
                    the inception model.
            """
            act = get_activations(dataset, model, batch_size, dims, device, num_workers)
            mu = np.mean(act, axis=0)
            sigma = np.cov(act, rowvar=False)
            print(f'mu.shape: {mu.shape}, sigma.shape: {sigma.shape}')
            return mu, sigma


        def compute_statistics_of_path(dataset, model, batch_size, dims, device,
                                    num_workers=1):

            m, s = calculate_activation_statistics(dataset, model, batch_size,
                                                    dims, device, num_workers)

            return m, s


        def calculate_fid_given_paths(sample_data, real_data, batch_size, device, dims, num_workers=1):
            """Calculates the FID of two paths"""

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

            model = InceptionV3([block_idx]).to(device)

            m1, s1 = compute_statistics_of_path(sample_data, model, batch_size,
                                                dims, device, num_workers)
            m2, s2 = compute_statistics_of_path(real_data, model, batch_size,
                                                dims, device, num_workers)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
            
            return fid_value

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
            sample_data = (model.sample((100,D))).cpu() 

        # transform the samples back to the original shape 28*28        
        sample_data = (sample_data + 1) / 2
        sample_data = sample_data.view(-1, 1, 28, 28)

        # transform the samples to be consistent with the real data
        sample_data = F.interpolate(sample_data, size=(299, 299), mode='bilinear', align_corners=False)
        sample_data_rgb = sample_data.repeat(1, 3, 1, 1)
        
        fid_value = calculate_fid_given_paths(sample_data_rgb, real_data, 32, args.device, dims=64)
        print(f"FID: {fid_value}")
                
                
                
                
                