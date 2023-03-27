import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from torch.autograd import Variable
from convnet import *

# Initialize data augmentation functions to transform the images
resize = transforms.Resize(size=(128,128))
hflip = transforms.RandomHorizontalFlip(p=0.5)
vflip = transforms.RandomVerticalFlip(p=0.5)
rotate = transforms.RandomRotation(degrees=15)

# Separate transformers for train and test set
train_transforms = transforms.Compose([resize, hflip, vflip, rotate, transforms.ToTensor()])
test_transforms = transforms.Compose([resize, transforms.ToTensor()])

# Image paths
train_image_path = "/illumina/scratch/deep_learning/jneo1/cs5242_project/weapon-detection-CNN/data/train"
test_image_path  = "/illumina/scratch/deep_learning/jneo1/cs5242_project/weapon-detection-CNN/data/test"

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

def visualize_batch(batch, batch_size, classes, dataset_type):
    '''
    Visualize sample images from training and test batches
    '''
	# Initialize a figure
    fig = plt.figure("{} batch".format(dataset_type),
                     figsize=(batch_size, batch_size))
	
    # Loop over the batch size
    for i in range(0, batch_size):
		# Create subplot
        ax = plt.subplot(2, 4, i + 1)
		# Grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
        image = batch[0][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255.0).astype("uint8")
        # Grab the label id and get the label from the classes list
        idx = batch[1][i]
        label = classes[idx]
        # Show the image along with the label
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
	
    plt.tight_layout()
    plt.show()

# Model initialization
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = ConvNet().to(device)

# Optmizer and loss function
optimizer = Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 10
train_count = len(train_ds)
test_count = len(test_ds)

# Model training and saving best model

best_accuracy=0.0

for epoch in range(num_epochs):
    
    print(f"{datetime.now()}: Epoch {epoch}")

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
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_dl):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_count
    
    print('Epoch: ' + str(epoch) + ' Train Loss: ' + "{:.4f}".format(train_loss.item()) + ' Train Accuracy: ' + "{:.4f}".format(train_accuracy) + ' Test Accuracy: ' + "{:.4f}".format(test_accuracy))
