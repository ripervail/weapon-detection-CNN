import torch
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import Saliency
from PIL import Image
from convnet import *
from ResNets import *
from utils.dynamiccentrecrop import DynamicCenterCrop

# Load model to test
model = ConvNet2()
model_name = "4-7_ConvNet2.100"
file_path = "saved_models/"
model.load_state_dict(torch.load(file_path + f"{model_name}.pth"))
# Set into evaluation mode
model.eval()

# Load the image
# with open(f"model_training_stats/{model_name}_correct_pred.pkl", "rb") as f:
#     correct_pred = pickle.load(f)
# with open(f"model_training_stats/{model_name}_wrong_pred.pkl", "rb") as f:
#     wrong_pred = pickle.load(f)

# Define the transformation for the image
transform = transforms.Compose([
    DynamicCenterCrop(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()])

def get_random_pic(modifier1, modifier2):
    '''
    - modifier1: "train" or "test"
    - modifier2: "norm" or "weap"
    '''
    file_path = f"data/{modifier1}/{modifier2}/"
    files = os.listdir(file_path)
    random_file = file_path + random.choice(files)
    return random_file

img_path = "rembg-data/train/weap/a015032m_040922_threat_3346_140.png"
img_tensor = transform(Image.open(img_path))
print(img_path)

# Create a batch of size 1
batch = img_tensor.unsqueeze(0)

# Compute the saliency map
saliency = Saliency(model)

# Get the saliency attribution
attribution = saliency.attribute(batch, target=0)

# Convert the tensor to a numpy array
attribution_np = attribution.squeeze(0).cpu().detach().numpy()

# Rescale the attribution to be between 0 and 1
attribution_np = (attribution_np - attribution_np.min()) / (attribution_np.max() - attribution_np.min())

# Plot the original image and the saliency map side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_tensor.permute(1, 2, 0))
axs[0].axis("off")
axs[0].set_title("Original Image")
axs[1].imshow(attribution_np.transpose(1, 2, 0), cmap="gray")
axs[1].axis("off")
axs[1].set_title("Saliency Map")

plt.savefig(f"figures/SMAP_{model_name}.png")