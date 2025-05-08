import torch
import torch.nn as nn
def _approximated_ReLU(x):
    return 0.117071 * x**2 + 0.5 * x + 0.375373

class Square(torch.nn.Module):
    def forward(self, x):
        return x**2

class ApproxReLU(torch.nn.Module):
    def forward(self, x):
        return _approximated_ReLU(x)

class Flatten(torch.nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class M1(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(M1, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=3, padding=0)
        self.Square1 = Square()
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(9 * 9 * 8, 64)
        self.Square2 = Square()
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square2(out)
        out = self.FC2(out)
        return out

class M2(torch.nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.Conv2 = torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(192, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        return out

class M3(torch.nn.Module):
    def __init__(self):
        super(M3, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.ApproxReLU1 = ApproxReLU()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(1014, 120)
        self.ApproxReLU2 = ApproxReLU()
        self.FC2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.ApproxReLU1(out)
        out = self.AvgPool1(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.ApproxReLU2(out)
        out = self.FC2(out)
        return out

class M4(torch.nn.Module):
    def __init__(self, hidden=84, output=10):
        super(M4, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        self.Conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        self.Conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.Square3 = Square()
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(120, hidden)
        self.Square4 = Square()
        self.FC2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Conv3(out)
        out = self.Square3(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square4(out)
        out = self.FC2(out)
        return out

class M5(torch.nn.Module):
    def __init__(self, output=10):
        super(M5, self).__init__()
        # L1 Image shape=(?, 32, 32, 1)
        #    Conv     -> (?, 30, 30, 16)
        #    Pool     -> (?, 15, 15, 16)
        self.Conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 15, 15, 16)
        #    Conv     -> (?, 12, 12, 64)
        #    Pool     -> (?, 6, 6, 64)
        self.Conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 6, 6, 64)
        #    Conv     -> (?, 4, 4, 128)
        #    Pool     -> (?, 1, 1, 128)
        self.Conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.Square3 = Square()
        self.AvgPool3 = torch.nn.AvgPool2d(kernel_size = 4)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(1*1*128, output)


    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Conv3(out)
        out = self.Square3(out)
        out = self.AvgPool3(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        return out


class M6(torch.nn.Module):
    def __init__(self):
        super(M6, self).__init__()
        self.Conv1 = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.Square1 = Square()        
        self.Conv2 = torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(128, 32)
        self.Square2 = Square()
        self.FC2 = torch.nn.Linear(32, 5)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.Conv2(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square2(out)
        out = self.FC2(out)
        return out
        
        
#class M7(torch.nn.Module):
    #def __init__(self):
        #super(M7, self).__init__()
        #self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=0)
        #self.ApproxReLU1 = ApproxReLU()
        #self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=0)
        #self.ApproxReLU2 = ApproxReLU()
        #self.conv3 = torch.nn. Conv2d(64, 128, kernel_size=3, padding=0)
        #self.ApproxReLU3 = ApproxReLU()
        #self.pool = torch.nn.AvgPool2d(2, 2)
        #self.fc1 = torch.nn.Linear(128*26*26, 1024)
        #self.ApproxReLU4 = ApproxReLU()
        #self.fc2 = torch.nn.Linear(1024, 1024)
        #self.ApproxReLU5 = ApproxReLU()
        #self.fc3 = torch.nn.Linear(1024, 2)

    #def forward(self, x):
        #x = self.conv1(x)
        #x = self.ApproxReLU1(x)
        #x = self.pool(x)
        #x = self.conv2(x)
        #x = self.ApproxReLU2(x)
        #x = self.pool(x)
        #x = self.conv3(x)
        #x = self.ApproxReLU3(x)
        #x = self.pool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.ApproxReLU4(x)
        #x = self.fc2(x)
        #x = self.ApproxReLU5(x)
        #x = self.fc3(x)
        #return x
        
#class M7(torch.nn.Module):
    #def __init__(self):
        #super(M7, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)  # Reduced number of filters
        #self.ApproxReLU1 = ApproxReLU()
        #self.pool = nn.AvgPool2d(2, 2)

        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)  # Reduced number of filters
        #self.ApproxReLU2 = ApproxReLU()

        #self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=0)  # Reduced number of filters
        #self.ApproxReLU3 = ApproxReLU()

        #self.fc1 = nn.Linear(64 * 24 * 24, 512)  # Corrected input size for the flattened layer
        #self.ApproxReLU4 = ApproxReLU()

        #self.fc2 = nn.Linear(512, 128)  # Reduced number of neurons
        #self.ApproxReLU5 = ApproxReLU()

        #self.fc3 = nn.Linear(128, 2)  # Output layer

    #def forward(self, x):
        #x = self.conv1(x)
        #x = self.ApproxReLU1(x)
        #x = self.pool(x)

        #x = self.conv2(x)
        #x = self.ApproxReLU2(x)
        #x = self.pool(x)

        #x = self.conv3(x)
        #x = self.ApproxReLU3(x)
        #x = self.pool(x)

        #x = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.ApproxReLU4(x)
        #x = self.fc2(x)
        #x = self.ApproxReLU5(x)
        #x = self.fc3(x)
        #return x
        


#class M7(nn.Module):
    #def __init__(self):
        #super(M7, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        #self.ApproxReLU1 = ApproxReLU()
        #self.pool = nn.AvgPool2d(2, 2)  # Pooling layer added
        
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        #self.ApproxReLU2 = ApproxReLU()
        
        #self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        #self.ApproxReLU3 = ApproxReLU()
        
        #self.pool_before_fc1 = nn.AvgPool2d(4, 4)  # Additional pooling layer
        
        #self.fc1 = nn.Linear(64 * 6 * 6, 512)  # Reduced input size for fc1
        #self.ApproxReLU4 = ApproxReLU()
        
        #self.fc2 = nn.Linear(512, 128)
        #self.ApproxReLU5 = ApproxReLU()
        
        #self.fc3 = nn.Linear(128, 2)

    #def forward(self, x):
        #x = self.conv1(x)
        #x = self.ApproxReLU1(x)
        #x = self.pool(x)
        
        #x = self.conv2(x)
        #x = self.ApproxReLU2(x)
        #x = self.pool(x)
        
        #x = self.conv3(x)
        #x = self.ApproxReLU3(x)
        #x = self.pool(x)
        
        #x = self.pool_before_fc1(x)  # Applying additional pooling before fc1
        #x = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.ApproxReLU4(x)
        #x = self.fc2(x)
        #x = self.ApproxReLU5(x)
        #x = self.fc3(x)
        #return x
        


# class M7(nn.Module):
#     def __init__(self):
#         super(M7, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Adding padding=1
#         self.relu1 = ApproxReLU()
#         self.pool1 = nn.AvgPool2d(2, 2)
        
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Adding padding=1
#         self.relu2 = Square()
#         self.pool2 = nn.AvgPool2d(2, 2)
        
#         self.avgpool_before_fc1 = nn.AvgPool2d(56)  # AvgPool layer with kernel size 56
        
#         self.fc1 = nn.Linear(32, 128)  # Adjusting input size for the flattened layer
#         self.relu3 = Square()
        
#         self.fc2 = nn.Linear(128, 2)  # Output layer

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
        
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
        
#         x = self.avgpool_before_fc1(x)  # Applying avg pooling before fc1
        
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu3(x)
        
#         x = self.fc2(x)
#         return x

# class M7(nn.Module):
#     def __init__(self):
#         super(M7, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.relu1 = ApproxReLU()
#         self.pool1 = nn.AvgPool2d(2, 2)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.relu2 = Square()
#         self.pool2 = nn.AvgPool2d(2, 2)

#         self.avgpool_before_fc1 = nn.AvgPool2d(10)  # AvgPool layer with kernel size 10

#         self.fc1 = nn.Linear(32 * 5 * 5, 128)  # Adjusting input size for the flattened layer
#         self.relu3 = Square()

#         self.fc2 = nn.Linear(128, 2)  # Output layer

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)

#         x = self.avgpool_before_fc1(x)  # Applying avg pooling before fc1

#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu3(x)

#         x = self.fc2(x)
#         return x


#New model using 100 by 100 image

class M7(nn.Module):
    def __init__(self):
        super(M7, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = ApproxReLU()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = Square()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.avgpool_before_fc1 = nn.AvgPool2d(5)  # AvgPool layer with kernel size 5

        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # Adjusting input size for the flattened layer
        self.relu3 = Square()

        self.fc2 = nn.Linear(128, 2)  # Output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.avgpool_before_fc1(x)  # Applying avg pooling before fc1

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x


