import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch 

# load international_matches.csv into a dataframe
df = pd.read_csv('./DL_HW2/Data/HW2_data/international_matches.csv')

# display the last 10 rows of the dataframe
print(df.tail(10))



# create a new column called home_team_result
# if home_team_score > away_team_score, home_team_result = 1
# # if home_team_score == away_team_score, home_team_result = 0
# # if home_team_score < away_team_score, home_team_result = -1
home_team_result = df['home_team_score'] - df['away_team_score']
home_team_result = pd.DataFrame(np.sign(home_team_result), columns=['home_team_result'])
# set home_team_result = 2 if home_team_result = -1
home_team_result[home_team_result == -1] = 2
# # scatter plot with home_team_fifa_rank on the x-axis and away_team_fifa_rank on the y-axis with the color of home_team_result
# scatter = plt.scatter(x=df['home_team_fifa_rank'], y=df['away_team_fifa_rank'], c=home_team_result, s=5)
# plt.xlabel("home_team_fifa_rank")
# plt.ylabel("away_team_fifa_rank")
# plt.legend(handles=scatter.legend_elements()[0],
# labels=['Win', 'Draw', 'Lose'])
# plt.show()

# remove the rows where both home_team_total_fifa_points and away_team_total_fifa_points are 0
columns = ["home_team_fifa_rank", "away_team_fifa_rank", "home_team_total_fifa_points", "away_team_total_fifa_points"]
df = df[columns]
# concatenate the home_team_result column to the dataframe
df = pd.concat([df, home_team_result], axis=1)
drop_indx = df[(df["away_team_total_fifa_points"] == 0) & (df["home_team_total_fifa_points"] == 0)].index
df = df.drop(drop_indx)
# # split the data into training and testing sets with a 75/25 split
train = df.sample(frac=0.75, random_state=0)
test = df.drop(train.index)
# split the training and testing sets into features and labels
train_features = train[columns]
train_labels = train['home_team_result']
test_features = test[columns]
test_labels = test['home_team_result']
# # normalize the data
# train_features = (train_features - train_features.mean()) / train_features.std()
# test_features = (test_features - test_features.mean()) / test_features.std()
# # convert the data into tensors
train_features = torch.tensor(train_features.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels.values, dtype=torch.float32)
test_features = torch.tensor(test_features.values, dtype=torch.float32)
test_labels = torch.tensor(test_labels.values, dtype=torch.float32)
# # plot heatmap of the correlation matrix
# corr = df.corr()
# sns.heatmap(corr, annot=True)
# plt.show()

# # use nn.module to create a neural network with 3 hidden layers and relu activation function
# # to predict home_team_result
# # the hidden layers have 10, 20, and 8 neurons respectively
# # the output layer has 3 neuron
# # the input layer has 4 neurons
# # the loss function is cross entropy loss
# # the optimizer is Adam
# # the learning rate is 0.001
# # the batch size is 32
# # the number of epochs is 10
# # the accuracy is calculated on the test set
# # the accuracy is printed after each epoch
# # the loss is printed after each epoch
# # the accuracy and loss are plotted after training

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 8)
        self.fc4 = nn.Linear(8, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
# define a new model with better accuracy on the test set
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 8)
        self.fc4 = nn.Linear(8, 3)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
batch_size = 32
num_epochs = 10
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    # train and append loss and accuracy
    net.train()
    for i in range(0, len(train_features), batch_size):
        optimizer.zero_grad()
        output = net(train_features[i:i+batch_size])
        loss = criterion(output, train_labels[i:i+batch_size].long())
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    net.eval()
    output = net(train_features)
    _, predicted = torch.max(output, 1)
    correct = (predicted == train_labels).sum().item()
    train_acc.append(correct / len(train_features))
    
    # test
    net.eval()
    with torch.no_grad():
        output = net(test_features)
        loss = criterion(output, test_labels.long())
        test_loss.append(loss.item())
        pred = torch.argmax(output, dim=1)
        acc = (pred == test_labels).sum().item() / len(test_labels)
        test_acc.append(acc)

    # print
    print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss.item(), acc))

# plot
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# predict iran's chances of winning the world cup
net.eval()
with torch.no_grad():
    iran = torch.tensor(([[20, 16, 1564.61, 1627.48],
    [20, 5, 1564.61, 1728.47],
    [20, 19, 1564.61, 1569.82]]))
    output = net(iran)
    pred = torch.argmax(output, dim=1)
    print(pred)
    # iran_vs_usa = torch.tensor([20, 16, 1564.61, 1627.48])
    # iran_vs_england = torch.tensor([20, 5, 1564.61, 1728.47])
    # iran_vs_wales = torch.tensor([20, 19, 1564.61, 1569.82])
    # output_usa = net(iran_vs_usa)
    # print("Iran vs USA: ", torch.argmax(output_usa, dim=1))
    # output_england = net(iran_vs_england)
    # output_wales = net(iran_vs_wales)
    
    # print("Iran vs England: ", torch.argmax(output_england, dim=1))
    # print("Iran vs Wales: ", torch.argmax(output_wales, dim=1))








