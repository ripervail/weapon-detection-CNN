import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from torch.autograd import Variable
from convnet import *
from ResNets import *
from utils.dynamiccentrecrop import DynamicCenterCrop
import pickle

# Initialize data augmentation functions to transform the images
resize = transforms.Resize(size=(128,128))
hflip = transforms.RandomHorizontalFlip(p=0.5)
colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
rotate = transforms.RandomRotation(degrees=15)

# Separate transformers for train and test set
# Mean and std determined from training data
train_transforms = transforms.Compose([
    # DynamicCenterCrop(),
    resize, 
    hflip, colorjitter, rotate, # Extras
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.0989, 0.0841, 0.0835], # Background removed
    #                      std=[0.1668, 0.1419, 0.1387])])
    transforms.Normalize(mean=[0.4327, 0.4083, 0.3829], # Standard
                         std=[0.2189, 0.2137, 0.2104])])
test_transforms = transforms.Compose([
    # DynamicCenterCrop(),
    resize, 
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.0989, 0.0841, 0.0835], # Background removed
    #                      std=[0.1668, 0.1419, 0.1387])])
    transforms.Normalize(mean=[0.4327, 0.4083, 0.3829], # Standard
                         std=[0.2189, 0.2137, 0.2104])])

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

# Model initialization
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model = baseResNet().to(device)
# model = ConvNet2().to(device)
# model = ResNet([2, 2, 2, 2]).to(device)
model = ResNet([1,1,1,1]).to(device)

# Optmizer and loss function
optimizer = Adam(model.parameters(),lr=0.0001, weight_decay=0.0001)
# loss_function = nn.CrossEntropyLoss()
loss_function = nn.BCEWithLogitsLoss()

num_epochs = 100
train_count = len(train_ds)
test_count  = len(test_ds)

# Model training
best_accuracy=0.0
# Record loss and accuracy
train_stats_loss = []
test_stats_loss  = []
train_stats_acc  = []
test_stats_acc   = []

for epoch in range(num_epochs):

    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_dl):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
    
        optimizer.zero_grad()

        outputs=model(images)
        loss=loss_function(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.cpu().data*images.size(0)
        # _,prediction=torch.max(outputs.data,1)
        
        # train_accuracy+=int(torch.sum(prediction==labels.data))
        # Round probabilities to 0 or 1
        probs = torch.sigmoid(outputs) # Use sigmoid since our final layer does not have sigmoid
        preds = (probs >= 0.5).long() # If >= 0.5, convert to 1. Else 0
        train_accuracy += (preds == labels.unsqueeze(1)).float().sum().item() # Record number of correct
        
    train_accuracy /= train_count
    train_loss /= train_count
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy = 0.0
    test_loss = 0.0

    with torch.no_grad():

        # Save file paths of correct or wrongly predicted images
        correct_pred = []
        wrong_pred = []

        for i, (images,labels) in enumerate(test_dl):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())
                
            outputs=model(images)
            loss=loss_function(outputs,labels.float().unsqueeze(1))
            test_loss+= loss.cpu().data*images.size(0)
            # _,prediction=torch.max(outputs.data,1)
            # test_accuracy+=int(torch.sum(prediction==labels.data))
            probs = torch.sigmoid(outputs) # Use sigmoid since our final layer does not have sigmoid
            preds = (probs >= 0.5).long() # If >= 0.5, convert to 1. Else 0
            test_accuracy += (preds == labels.unsqueeze(1)).float().sum().item()

            # Record correct or wrong predictions
            for j in range(len(preds)):
                if preds[j] == labels[j]:
                    correct_pred.append(test_ds.imgs[i * test_dl.batch_size + j][0])
                else:
                    wrong_pred.append(test_ds.imgs[i * test_dl.batch_size + j][0])

    test_accuracy /= test_count
    test_loss /= test_count 
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Epoch: ' + str(epoch) + ' Train Loss: ' + "{:.4f}".format(train_loss.item()) + ', Test Loss: ' + "{:.4f}".format(test_loss.item()) + ', Train Accuracy: ' + "{:.4f}".format(train_accuracy) + ', Test Accuracy: ' + "{:.4f}".format(test_accuracy))

    # Record stats for plotting
    train_stats_loss.append(train_loss.item())
    train_stats_acc.append(train_accuracy)
    test_stats_loss.append(test_loss.item())
    test_stats_acc.append(test_accuracy)

# Save each stats
now = datetime.now()
day = now.day
month = now.month
model_name = model.get_name()
file_dir = "model_training_stats"

with open(f"{file_dir}/{month}-{day}_{model_name}.{num_epochs}_1_train_loss.pkl", "wb") as f:
    pickle.dump(train_stats_loss, f)
with open(f"{file_dir}/{month}-{day}_{model_name}.{num_epochs}_1_train_acc.pkl", "wb") as f:
    pickle.dump(train_stats_acc, f)
with open(f"{file_dir}/{month}-{day}_{model_name}.{num_epochs}_1_test_loss.pkl", "wb") as f:
    pickle.dump(test_stats_loss, f)
with open(f"{file_dir}/{month}-{day}_{model_name}.{num_epochs}_1_test_acc.pkl", "wb") as f:
    pickle.dump(test_stats_acc, f)

# Save the model
torch.save(model.state_dict(), f"saved_models/{month}-{day}_{model_name}.{num_epochs}.pth")