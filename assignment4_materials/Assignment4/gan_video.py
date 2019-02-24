import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from gan.utils import show_images, sample_noise, deprocess_img
import copy

import numpy as np
import cv2

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from gan.train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from gan.models import Discriminator, Generator


batch_size = 128
scale_size = 128  # We resize the images to 64x64 for training

celeba_root = 'celeba_data'

NOISE_DIM = 100
NUM_EPOCHS = 50
learning_rate = 0.0002

D = Discriminator().to(device)
G = Generator(noise_dim=NOISE_DIM).to(device)

D.load_state_dict(torch.load('mydiscriminator.pt'))
G.load_state_dict(torch.load('mygenerator.pt'))

step_size = 0.2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('faces.avi', fourcc, 20.0, (128, 128))


vec1 = torch.from_numpy(2 * np.random.random((1, NOISE_DIM)) - 1).to(device).float()
vec2 = torch.from_numpy(2 * np.random.random((1, NOISE_DIM)) - 1).to(device).float()
step_size = 20

plt.figure()

vec = sample_noise(1, NOISE_DIM).to(device)
orig_vec = copy.deepcopy(vec)

num_frames = 1000

for i in range(num_frames):
    
    iteration = i % step_size
    
    if (iteration == 0 and i != 0):
        vec1 = vec2
        if (i == num_frames - step_size):
            vec2 = orig_vec
        else:
            vec2 = torch.from_numpy(2 * np.random.random((1, NOISE_DIM)) - 1).to(device).float()
            
        print(i, num_frames - step_size)
        print(vec2 == orig_vec)

    curr_vec = vec1 + (vec2 - vec1) * (iteration / step_size)
    
    image = G(curr_vec).cpu().data.numpy()[0,:,:,:]
    image = deprocess_img(image)
    image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
    image = (image*256).astype(np.uint8)
    image = np.flip(image, 2)
    out.write(image)
    
cap.release()
out.release()
