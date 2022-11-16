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

# define an MLP model that achieves more than 75% accuracy on the test set
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


# define the model
model_Adam = MLP(784, 256, 25).to(device)


# define the optimizer
optimizer = optim.Adam(model_Adam.parameters(), lr=0.001)
# train the model
train_losses_Adam, val_losses_Adam, train_accs_Adam, val_accs_Adam = train_model(model_Adam, train_loader, val_loader, criterion, optimizer, epochs=10)

# define a new model with SGD optimizer and train it and test it
model_SGD = MLP(784, 256, 25)
model_SGD.to(device)
optimizer_SGD = optim.SGD(model_SGD.parameters(), lr=0.001)
train_losses_SGD, val_losses_SGD, train_accs_SGD, val_accs_SGD = train_model(model_SGD, train_loader, val_loader, criterion, optimizer_SGD, epochs=10)

# plot the training and validation losses and accuracies for both optimizers
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_Adam, label='Adam_training_loss')
plt.plot(train_losses_SGD, label='SGD_training_loss')
plt.plot(val_losses_Adam, label='Adam_validation_loss')
plt.plot(val_losses_SGD, label='SGD_validation_loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(train_accs_Adam, label='Adam_training_accuracy')
plt.plot(train_accs_SGD, label='SGD_training_accuracy')
plt.plot(val_accs_Adam, label='Adam_validation_accuracy')
plt.plot(val_accs_SGD, label='SGD_validation_accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# compare the test losses and accuracies for both optimizers
test_loss_Adam, test_acc_Adam = test_model(model_Adam, test_loader, criterion)
test_loss_SGD, test_acc_SGD = test_model(model_SGD, test_loader, criterion)
print('Adam Test Loss: {:.6f} \tAdam Test Accuracy: {:.6f}'.format(test_loss_Adam, test_acc_Adam))


# define a new model like the previous one but with dropout
class MLP_dropout(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# define the model and train it and test it with Adam optimizer and plot the training and validation losses and accuracies
model_Adam_dropout = MLP_dropout(784, 256, 25).to(device)
optimizer = optim.Adam(model_Adam_dropout.parameters(), lr=3e-4)
train_losses_Adam_dropout, val_losses_Adam_dropout, train_accs_Adam_dropout, val_accs_Adam_dropout = train_model(model_Adam_dropout, train_loader, val_loader, criterion, optimizer, epochs=10)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_Adam_dropout, label='Adam_training_loss')
plt.plot(val_losses_Adam_dropout, label='Adam_validation_loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(train_accs_Adam_dropout, label='Adam_training_accuracy')
plt.plot(val_accs_Adam_dropout, label='Adam_validation_accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
print('Adam Test Loss and Accuracy with dropout') 
test_loss_Adam_dropout, test_acc_Adam_dropout = test_model(model_Adam_dropout, test_loader, criterion)





