import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow

import glob
import train

device = torch.device("cpu")

output_folder = 'output/'
latest_model_path = glob.glob("output/glow_model_*.pth")[-1]

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)

test_mnist = train.MyMNIST(train=False, download=False)
image_shape = (32, 32, 1)
num_classes = 10
batch_size = 512

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(latest_model_path))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model):
    with torch.no_grad():
        if hparams['y_condition']:
            y = torch.eye(num_classes)
            y = y.repeat(batch_size // num_classes + 1)
            y = y[:32, :].to(device) # number hardcoded in model for now
        else:
            y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()

images = sample(model)
grid = make_grid(images[:30], nrow=6).permute(1,2,0)

plt.figure(figsize=(10,10))
plt.imshow(grid)
plt.axis('off')
plt.savefig('output.png')

