from dataset import HistoryPoints
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=5, hidden_size=100 ,num_layers=10, dropout=0.2)
        self.fc1 = nn.Linear(100, 50, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(50, 1, bias=True)
        self.act2 = nn.ReLU(inplace=True)


    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc1(output)
        output = self.act1(output)
        output = self.fc2(output)
        output = self.act2(output)
        return output


    def computeLoss(self, logit, y):
        lossFunc = nn.MSELoss()
        return lossFunc(logit, y)
