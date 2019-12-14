import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 5, kernel_size=2, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
#        self.layer2 = nn.Sequential(
#                nn.Conv2d(5, 10, kernel_size=2, stride=1, padding=2),
#                nn.ReLU(),
#                nn.MaxPool2d(kernel_size=2, stride=2)
#                )
        self.dropout = nn.Dropout(p=.1)
        self.fc1 = nn.Linear(3594, 500)
        self.fc2 = nn.Linear(500, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, x, y):
        x = self.layer1(x)
#        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = torch.cat((x,y),1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
dqn = DQN()

criterion = nn.MSELoss()
optimizer = optim.Adam(dqn.parameters(), lr=0.001) #, momentum=0.9)


def batch_split(X, y, batch_size):
    dataset = []
    for i in range(int(len(X)/batch_size)):
        if (i+1)*batch_size < len(X):
            to_append = X[i*batch_size:(i+1)*batch_size][:]
            smally = y[i*batch_size:(i+1)*batch_size][:]
            to_append = np.reshape(to_append, (batch_size, 2, 512))
            dataset.append((to_append, X[i*batch_size:(i+1)*batch_size][:], smally))
        else:
            smallX = X[i*batch_size:][:]
            smally = y[i*batch_size:][:]
            smallX = np.reshape(smallX, (batch_size, 2, 512))
            dataset.append((smallX, X[i*batch_size:][:], smally))
#            #dataset.append(([X[i*batch_size:(i+1)*batch_size][:512].reshape(512,1), X[i*batch_size:(i+1)*batch_size][0][512:].reshape(512,1)], y[i*batch_size:(i+1)*batch_size][:]))
#            dataset.append(([X[i*batch_size:(i+1)*batch_size][:512], X[i*batch_size:(i+1)*batch_size][512:]], y[i*batch_size:(i+1)*batch_size][:]))
#        else:
#            dataset.append(([X[i*batch_size:][:512], X[i*batch_size:][512:]], y[i*batch_size:][:]))

    return dataset


y = np.load('/Users/alexnam/doodle_img_comparison/preprocessing/data/pixel_features/vgg_moredata_y_train.npy', allow_pickle=True)
X = np.load('/Users/alexnam/doodle_img_comparison/preprocessing/data/pixel_features/vgg_moredata_X_train.npy', allow_pickle=True) 
split = int(.8*len(X))
X_train = X[:split][:]
y_train = y[:split][:]
X_val = X[split:][:]
y_val = y[split:][:]
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_val2 = X_val
X_val1 =np.reshape(X_val, (len(X_val), 2, 512))

for epoch in tqdm(range(20)):
    running_loss = 0.0
    trainloader = batch_split(X_train, y_train, batch_size = 10)
    for i, data in enumerate(trainloader, 0):
        inputs, inputs2, labels = data
        optimizer.zero_grad()
        # 1, 2, 512
#        import pdb; pdb.set_trace()
        inputs = torch.from_numpy(inputs).unsqueeze(1)
        outputs = dqn(Variable(inputs), Variable(torch.from_numpy(inputs2)))
        loss = criterion(outputs.float(), Variable(torch.from_numpy(labels.reshape(10,1))).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        if i % 2000 == 1999:
            dqn.eval()
            y_pred = dqn(Variable(torch.from_numpy(X_val1).unsqueeze(1)), Variable(torch.from_numpy(X_val2)))
            loss = criterion(y_pred.float(), Variable(torch.from_numpy(y_val.reshape(len(y_val),1))).float())
            print('validation-loss', loss.item(), end='\n')
            dqn.train()
        if i == len(trainloader) - 1:
            dqn.eval()
            y_pred = dqn(Variable(torch.from_numpy(X_val1).unsqueeze(1)), Variable(torch.from_numpy(X_val2)))
            pred = y_pred.data.numpy()
            accuracy = 0.0
            for j in range(len(y_val)):
                if pred[j] > 0.5 and y_val[j] == 1: accuracy += 1
                elif pred[j] < 0.5 and y_val[j] == 0: accuracy += 1
            print('prediction accuracy ', accuracy/len(y_val))
            dqn.train()
print('Finished Training')

PATH = './matcher_cnn_L2loss_layers_concatenate_1conv_with_linear_adam.pth'
torch.save(dqn.state_dict(), PATH)
