#import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn import preprocessing
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image


austin_df = pd.read_csv('austin_zones.csv')
austin_df = austin_df.drop('Unnamed: 0', axis = 1)
austin_df.head()

img_names = austin_df['image_name']

dim = (150, 150)
img_df = []

for i, img in enumerate(img_names):
    visual_img = image.load_img(('../'+img), target_size=dim)
    img_array = image.img_to_array(visual_img)
    img_df.append(img_array)

img_array = np.array(img_df)

labels = austin_df['zone']
X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size=.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train / 255
X_test = X_test / 255



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #1st conv block
        self.conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)

        #2nd conv block
        self.conv2 = nn.Conv2d(20, 50, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(2, 2)

        #3rd conv block
        self.conv3 = nn.Conv2d(50, 100, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(100)

        #4th conv block
        self.conv4 = nn.Conv2d(100, 200, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(200)


        self.conv5 = nn.Conv2d(200, 300, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(300)

        self.conv6 = nn.Conv2d(300, 400, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(400)

        self.conv7 = nn.Conv2d(400, 600, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(600)

        #dropout and fully connected layer
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(194400, 4)

    #forward propogation
    def forward(self, x):
    	#1st conv block
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        #2nd conv block
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        # print(x.shape)

        #3rd conv block
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        #4th conv block
        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)

        x = F.relu(self.conv6(x))
        x = self.bn6(x)

        x = F.relu(self.conv7(x))
        x = self.bn7(x)

        # print(x.shape)
        #dropout and fully connected layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)
        x = self.fc1(x)
        return x



net = Net()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('param count:', count_parameters(net))

#set device to gpu if one exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
	net = nn.DataParallel(net)
net.to(device)

#cross entropy loss w SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

#convert numpy arrays to tensors, encode labels
X_train = torch.from_numpy(X_train)
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = le.transform(y_test)
y_test = torch.from_numpy(y_test)

#create dataloaders for all 3 datasets
train_set = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_set, batch_size = 32, shuffle=True)

test_set = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_set, batch_size = 32)

for epoch in range(32):
	#set loss for each epoch to 0
	running_loss = 0.0

	#loop through training set in batches
	for i, data in enumerate(train_dataloader):
		#get input data and labels
		inputs, labels = data

		#labels need to be LongTensor type for some reason
		labels = labels.type(torch.LongTensor)

		#inputs need to be floats
		inputs = inputs.float()

		#reshape inputs as (channels, batch size, 20, 32)
		#(20, 32) at end should be constant due to our preprocessing of the
		#MFCCs
		inputs = torch.reshape(inputs, (3, inputs.shape[0], 150, 150))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

		#zero the gradients in optimizer
		optimizer.zero_grad()

		#create predictions, calculate loss, backward propogation, then
		#update weights of model
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		#add loss
		running_loss += loss.item()

		#print update every 32 batches
		if i % 32 == 31:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.3f}')
			running_loss = 0.0

print('Finished training')

######Testing#######
correct = 0
total = 0 

#use no_grad as we are testing
with torch.no_grad():

	#loop through test data
	for data in test_dataloader:
		#gets inputs and labels
		inputs, labels = data 

		#convert labels to long tensors
		labels = labels.type(torch.LongTensor)

		#convert inputs to float
		inputs = inputs.float()
		
		#reshape inputs as (channels, batch size, 20, 32)
		inputs = torch.reshape(inputs, (3, inputs.shape[0], 150, 150))
		inputs = torch.permute(inputs, (1, 0, 2, 3))

		#send data to gpus
		inputs = inputs.to(device)
		labels = labels.to(device)

		#predict outputs
		outputs = net(inputs)

		#calculate class with highest probability
		_, predicted = torch.max(outputs.data, 1)

		#update total and correct counts
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
print(f'Total count: {total}, correct: {correct}')

PATH = os.getcwd()
file_name = 'nn02_model.pt'
torch.save(net.state_dict(), os.path.join(PATH, file_name))