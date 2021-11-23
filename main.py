import torch
import torch.optim as optim
import numpy as np
from torch import nn
from netwok import Net
from data_reader import read_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
net = Net().to(device)

x_train, y_train = read_data("data/2")
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()
x_train, y_train = x_train.to(device), y_train.to(device)

x_test = x_train
y_test = y_train

batches = 5
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)
learning_rate = 0.1
epochs = 1000

# optimizer = torch.optim.Adam(net.parameters(), learning_rate)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = np.empty(len(x_train_batches))
    for batch in range(len(x_train_batches)):
        optimizer.zero_grad()
        outputs = net(x_train_batches[batch])
        loss = criterion(outputs, y_train_batches[batch])
        loss.backward()
        optimizer.step()

        running_loss[batch] = loss.item()

    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss.mean()))

# print("accuracy = %s" % net.accuracy(x_test, y_test))
print("accuracy = %s" % torch.mean(torch.eq(torch.max(net(x_test), 1).indices, y_test).float()))
