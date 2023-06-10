import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import tqdm
import pickle

# Initialize data augmentation functions to transform the images
resize = transforms.Resize(size=(224,224))
hflip = transforms.RandomHorizontalFlip(p=0.5)
colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

rotate = transforms.RandomRotation(degrees=15)

# Separate transformers for train and test set
# train_transforms = transforms.Compose([resize, hflip, vflip, rotate, transforms.ToTensor()])
# Mean and std determined from training data
#transformations
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),                                
                                       transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

# Image paths
train_image_path = "data/train"
test_image_path  = "data/test"

# Make the datasets
print(f"{datetime.now()}: Creating datasets...")
train_ds = ImageFolder(root=train_image_path,
                       transform=train_transforms)
test_ds = ImageFolder(root=test_image_path,
                      transform=test_transforms)

# Print some info
print(f"Training dataset: {len(train_ds)} samples")
print(f"Test dataset: {len(test_ds)} samples")

# Create DataLoaders
print(f"{datetime.now()}: Creating DataLoaders...")
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=32)
test_dl  = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=32)

def make_train_step(model, optimizer, loss_fn):
    def train_step(x,y):
        #make prediction
        yhat = model(x)
        #enter train mode
        model.train()
        #compute loss
        loss = loss_fn(yhat,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #optimizer.cleargrads()

        return loss
    return train_step

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = models.resnet18(pretrained=True)

# Load the ResNet
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

#freeze all params
for params in model.parameters():
    params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)

#loss
loss_fn = nn.BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(model.fc.parameters()) 

#train step
train_step = make_train_step(model, optimizer, loss_fn)

losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 20
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

for epoch in range(n_epochs):
    epoch_loss = 0
    for i ,data in tqdm(enumerate(train_dl), total = len(train_dl)): #iterate ove batches
        x_batch , y_batch = data
        x_batch = x_batch.to(device) #move to gpu
        y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
        y_batch = y_batch.to(device) #move to gpu


        loss = train_step(x_batch, y_batch)
        epoch_loss += loss/len(train_dl)
        losses.append(loss)
        
    epoch_train_losses.append(epoch_loss)
    print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

    #validation doesnt requires gradient
    with torch.no_grad():
        cum_loss = 0
        val_accuracy = 0.0
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device)

            #model to eval mode
            model.eval()

            yhat = model(x_batch)
            val_loss = loss_fn(yhat,y_batch)
            cum_loss += loss/len(test_dl)
            val_losses.append(val_loss.item())

            probs = torch.sigmoid(yhat) # Use sigmoid since our final layer does not have sigmoid
            preds = (probs >= 0.5).long() # If >= 0.5, convert to 1. Else 0
            val_accuracy += (preds == y_batch).sum().item()


    epoch_test_losses.append(cum_loss)
    print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))
    print(f"Accuracy: {val_accuracy / len(test_ds)}")

# Save the model
torch.save(model.state_dict(), f"saved_models/resnet18_rembg.pth")