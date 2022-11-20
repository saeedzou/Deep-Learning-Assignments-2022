import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch 
import matplotlib.pyplot as plt
import string
from torchvision import transforms
from torch import optim
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# create a custom dataset class to load the data from q3_train.csv and q3_test.csv
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data.iloc[idx, 1:].values.astype('float32')
        y = self.data.iloc[idx, 0]
        if self.transform:
            x = self.transform(x)
        return x, y

# create a custom transform class to normalize the data
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

# define a transform to normalize the data
transform = transforms.Compose([Normalize(0.5, 0.5)])

# Load q3_train.csv and shuffle the data and split it into train and validation sets
trainset = CustomDataset('./DL_HW2/Data/HW2_data/Q3_train.csv', transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [int(0.8*len(trainset)), int(0.2*len(trainset))])
testset = CustomDataset('./DL_HW2/Data/HW2_data/Q3_test.csv', transform=transform)
# create dataloaders for train and validation and test sets
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# define an MLP model with 2 hidden layers
# there are 25 classes in the dataset

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# define the loss function
criterion = nn.CrossEntropyLoss()



# define a function to train the model and return the training and validation losses and accuracies
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (output.argmax(1) == y).type(torch.float).mean().item()
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_acc += (output.argmax(1) == y).type(torch.float).mean().item()
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        train_accs.append(train_acc/len(train_loader))
        val_accs.append(val_acc/len(val_loader))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(epoch+1, train_loss/len(train_loader), val_loss/len(val_loader), train_acc/len(train_loader), val_acc/len(val_loader)))
    return train_losses, val_losses, train_accs, val_accs

# define a function to test the model and return the test loss and accuracy
def test_model(model, test_loader, criterion):
    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == y).type(torch.float).mean().item()
    print('Test Loss: {:.6f} \tTest Accuracy: {:.6f}'.format(test_loss/len(test_loader), test_acc/len(test_loader)))
    return test_loss/len(test_loader), test_acc/len(test_loader)

# Choose the best learning rate for Adam optimizer
learning_rates = [1e-5, 3e-5,1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for lr in learning_rates:
    model = MLP(784, 256, 25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)
    train_losses.append(train_loss[1])
    val_losses.append(val_loss[1])
    train_accs.append(train_acc[1])
    val_accs.append(val_acc[1])

# choose the best learning rate
best_lr_Adam = learning_rates[np.argmax(val_accs)]
print('Best learning rate: {}'.format(best_lr_Adam))

# # choose best learning rate for SGD optimizer
# learning_rates = [1e-5, 3e-5,1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []

# for lr in learning_rates:
#     model = MLP(784, 256, 25).to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)
#     train_losses.append(train_loss[1])
#     val_losses.append(val_loss[1])
#     train_accs.append(train_acc[1])
#     val_accs.append(val_acc[1])
# # choose the best learning rate
# best_lr_SGD = learning_rates[np.argmax(val_accs)]
# print('Best learning rate: {}'.format(best_lr_SGD))



# define the model as model_Adam and save the best model based on the validation accuracy
model_Adam = MLP(784, 256, 25).to(device)
optimizer = optim.Adam(model_Adam.parameters(), lr=best_lr_Adam)
train_losses_Adam, val_losses_Adam, train_accs_Adam, val_accs_Adam = train_model(model_Adam, train_loader, val_loader, criterion, optimizer, epochs=25)
test_loss_Adam, test_acc_Adam = test_model(model_Adam, test_loader, criterion)
torch.save(model_Adam.state_dict(), './model_Adam.pt')

# # define the model as model_SGD and save the best model based on the validation accuracy
# model_SGD = MLP(784, 256, 25).to(device)
# optimizer = optim.SGD(model_SGD.parameters(), lr=best_lr_SGD)
# train_losses_SGD, val_losses_SGD, train_accs_SGD, val_accs_SGD = train_model(model_SGD, train_loader, val_loader, criterion, optimizer, epochs=25)
# test_loss_SGD, test_acc_SGD = test_model(model_SGD, test_loader, criterion)
# torch.save(model_SGD.state_dict(), './model_SGD.pt')

# # define a new model like MLP but with dropout
# class MLP_dropout(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP_dropout, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(p=0.5)
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # choose best learning rate for Adam optimizer for MLP_dropout
# learning_rates = [1e-5, 3e-5,1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []

# for lr in learning_rates:
#     model = MLP_dropout(784, 256, 25).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)
#     train_losses.append(train_loss[1])
#     val_losses.append(val_loss[1])
#     train_accs.append(train_acc[1])
#     val_accs.append(val_acc[1])
# # choose the best learning rate
# best_lr_Adam_dropout = learning_rates[np.argmax(val_accs)]
# print('Best learning rate: {}'.format(best_lr_Adam_dropout))

# # train the model with the best learning rate and save the best model based on the validation accuracy
# model_Adam_dropout = MLP_dropout(784, 256, 25).to(device)
# optimizer = optim.Adam(model_Adam_dropout.parameters(), lr=best_lr_Adam_dropout)
# train_losses_Adam_dropout, val_losses_Adam_dropout, train_accs_Adam_dropout, val_accs_Adam_dropout = train_model(model_Adam_dropout, train_loader, val_loader, criterion, optimizer, epochs=50)
# test_loss_Adam_dropout, test_acc_Adam_dropout = test_model(model_Adam_dropout, test_loader, criterion)
# torch.save(model_Adam_dropout.state_dict(), './model_Adam_dropout.pt')

# # plot the training and validation loss curves for MLP_SGD, MLP_Adam, MLP_Adam_dropout
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses_SGD, label='train loss SGD')
# plt.plot(val_losses_SGD, label='val loss SGD')
# plt.plot(train_losses_Adam, label='train loss Adam')
# plt.plot(val_losses_Adam, label='val loss Adam')
# plt.plot(train_losses_Adam_dropout, label='train loss Adam dropout')
# plt.plot(val_losses_Adam_dropout, label='val loss Adam dropout')
# plt.legend()
# plt.xlabel('Epochs')

# # plot the training and validation accuracy curves for MLP_SGD, MLP_Adam, MLP_Adam_dropout
# plt.figure(figsize=(10, 5))
# plt.plot(train_accs_SGD, label='train acc SGD')
# plt.plot(val_accs_SGD, label='val acc SGD')
# plt.plot(train_accs_Adam, label='train acc Adam')
# plt.plot(val_accs_Adam, label='val acc Adam')
# plt.plot(train_accs_Adam_dropout, label='train acc Adam dropout')
# plt.plot(val_accs_Adam_dropout, label='val acc Adam dropout')
# plt.legend()
# plt.xlabel('Epochs')

# Choose the best model among MLP_SGD, MLP_Adam, MLP_Adam_dropout based on the validation accuracy
best_model = model_Adam
best_model.load_state_dict(torch.load('./model_Adam.pt'))

