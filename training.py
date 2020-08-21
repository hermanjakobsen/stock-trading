from dataset import getTrainAndTestData, convertToDataLoader
from model import Model
import torch
import torch.optim as optim

Epoch = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

xTrain, yTrain, xTest, yTest, yTestUnscaled = getTrainAndTestData("AAPL_daily.csv", 0.9)

trainDataLoader = convertToDataLoader(xTrain, yTrain, 32, True, True)
testDataLoader = convertToDataLoader(xTest, yTest, 32, False, False)

model = Model()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(Epoch):
    print(f'Epoch: {epoch}')
    print(f'\tTraining...')

    for x, y in trainDataLoader:
        x = x.to(device=device)
        y = y.to(device=device)

        logit = model(x)

        print(logit.shape)

        loss =  model.computeLoss(logit, y)
        #accuracy = (logit.argmax(dim=1)==y).float().mean()

        #optimizer.zero_grad()

        #loss.backward()
 
        #optimizer.step()


        




