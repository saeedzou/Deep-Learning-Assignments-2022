import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import transforms
from torchvision.transforms import Lambda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Load fashion MNIST dataset from torchvision
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# create dataloaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# plot one random image from each class
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    indx = np.random.choice(np.where(np.array(trainset.targets) == i)[0])
    ax[i//5, i%5].imshow(trainset[indx][0].squeeze(), cmap='gray')
    ax[i//5, i%5].set_title(classes[i])
plt.show()

# use nn.Module to create linear layer from scratch
# use he initialization for weights and zero initialization for bias
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2/in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

# use nn.Module to create ReLU activation from scratch
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(min=0)
# use nn.Module to create softmax activation from scratch
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

# implement a class named model that uses the above classes to create a model
# the model should have 3 linear layers with 128, 64, and 10 neurons respectively
# the model should have 3 ReLU activation layers
# the model should have a softmax activation layer
# do not use nn.Sequential
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

# create a model object
model = Model()
# move the model to GPU if available
model.to(device)

# define cross entropy loss from scratch
def cross_entropy(pred, target):
    return -torch.log(pred[range(target.shape[0]), target]).mean()

# define accuracy from scratch
def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean()

# define training loop without using torch.optim
def train(model, train_loader, test_loader, epochs, lr):
    # save train and test losses and accuracies
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    # set model to train mode
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # move data and target to GPU if available
            data, target = data.to(device), target.to(device)
            # zero the parameter gradients
            model.zero_grad()
            # forward pass
            output = model(data)
            # calculate loss
            loss = cross_entropy(output, target)
            # backward pass
            loss.backward()
            # update weights
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
            # print training loss every 100 batches
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # set model to evaluation mode
        model.eval()
        # calculate validation loss and accuracy
        val_loss, val_acc = 0, 0

        for data, target in test_loader:
            # move data and target to GPU if available
            data, target = data.to(device), target.to(device)
            # forward pass
            output = model(data)
            # calculate loss
            val_loss += cross_entropy(output, target).item()
            # calculate accuracy
            val_acc += accuracy(output, target).item()
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        print('Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(val_loss, val_acc))
        # set model back to train mode
        model.train()
        # save train and test losses and accuracies
        train_losses.append(loss.item())
        train_accs.append(accuracy(output, target).item())
        test_losses.append(val_loss)
        test_accs.append(val_acc)
    return train_losses, train_accs, test_losses, test_accs

# train the model and plot train and test losses and accuracies
train_losses, train_accs, test_losses, test_accs = train(model, train_loader, test_loader, 10, 0.01)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(train_losses, label='train')
ax[0].plot(test_losses, label='test')
ax[0].set_title('Loss')
ax[0].legend()
ax[1].plot(train_accs, label='train')
ax[1].plot(test_accs, label='test')
ax[1].set_title('Accuracy')
ax[1].legend()
plt.show()



