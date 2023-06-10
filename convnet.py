import torch.nn as nn

# Basic convolutional model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,128,128)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,128,128)
        self.bn1=nn.BatchNorm2d(num_features=8)
        #Shape= (256,12,128,128)
        self.relu1=nn.LeakyReLU(0.01)
        #Shape= (256,12,128,128)
        
        self.pool1=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,64,64)
        
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,64,64)
        # self.relu2=nn.LeakyReLU(0.01)
        #Shape= (256,20,64,64)

        # Extra
        # self.pool2=nn.MaxPool2d(kernel_size=2)
        
        # self.conv3=nn.Conv2d(in_channels=12,out_channels=16,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,64,64)
        self.bn3=nn.BatchNorm2d(num_features=16)
        #Shape= (256,32,64,64)
        self.relu3=nn.LeakyReLU(0.01)
        #Shape= (256,32,64,64)
        
        # Old = 64*64*32
        self.dropout = nn.Dropout(p=0.5)
        self.fc=nn.Linear(in_features=16*16*16, out_features=1)
              
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool1(output)

        output=self.conv2(output)
        # output=self.relu2(output)
        # output=self.pool2(output)

        # output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        
        output=self.dropout(output)
        # Above output will be in matrix form, with shape (256,32,64,64)
        output=output.view(-1,16*16*16)
        output=self.fc(output)
        return output
    
    def get_name(self):
        return type(self).__name__
    
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.25)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout(p=0.25)
        
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=256)
        self.dropoutfc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropoutfc(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_name(self):
        return type(self).__name__