from collections import OrderedDict

from torchvision.transforms import Compose, Resize, ToTensor

import wandb
import nrrd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        # torch.manual_seed("THIS IS A GREAT SEED")
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_scan_tensor(sidelength)
        self.pixels = img.flatten()
        self.coords = get_mgrid(sidelength, 3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

def get_scan_tensor(sidelenght):
    data, headers = nrrd.read("ASOCA_001_0000.nrrd")
    transform = Compose([
        ToTensor(),
        Resize(sidelenght),
    ])
    # data = torch.from_numpy(data)
    # data = data.float()
    # data = data.resize((sidelenght, sidelenght, sidelenght))
    # data = (data - data.min()) / (data.max() - data.min())
    # return data
    return transform(data)


def psnr(im1, im2, max_pixel_value=1):
    return 20 * torch.log10(max_pixel_value / mse(im1, im2))

def mse(im1, im2):
    return F.mse_loss(im1, im2)

# wandb.login()

epochs = 500
learning_rate = 1e-4

scan = ImageFitting(512)
dataloader = DataLoader(scan, batch_size=1, pin_memory=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.to(device), ground_truth.to(device)


# run = wandb.init(
#     project="SIREN_scans",
#     name="SIREN_scan",
#     config={
#         "epochs": epochs,
#         "learning rate": learning_rate,
#     },
# )

model = Siren(in_features=3, out_features=1, hidden_features=256,
              hidden_layers=3, outermost_linear=True).to(device)

optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())



for epoch in range(epochs):
    model_output = model(model_input)

    loss = mse(ground_truth, model_output)
    psnr = psnr(ground_truth, model_output)

    optim.zero_grad()
    loss.backward()
    optim.step()

    # run.log({
    #     "loss": loss,
    #     "psnr": psnr,
    # })

    if not epoch % 10:
        print(f"Epoch {epoch}: loss = {loss:.4f}, psnr = {psnr:.4f}")

# run.finish()

