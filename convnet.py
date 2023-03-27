import torch.nn as nn

# Basic convolutional model
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,128,128)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,128,128)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,128,128)
        self.relu1=nn.ReLU()
        #Shape= (256,12,128,128)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,64,64)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,64,64)
        self.relu2=nn.ReLU()
        #Shape= (256,20,64,64)
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,64,64)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,64,64)
        self.relu3=nn.ReLU()
        #Shape= (256,32,64,64)
        
        self.fc=nn.Linear(in_features=64 * 64 * 32,out_features=num_classes)
              
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)

        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        # Above output will be in matrix form, with shape (256,32,64,64)
        output=output.view(-1,32*64*64)
        output=self.fc(output)
        return output