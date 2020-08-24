from dataset import getTrainAndTestData, convertToDataLoader
from model import Model
import torch
import torch.optim as optim

Epoch = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

xTrain, yTrain, xTest, yTest, yTestUnscaled = getTrainAndTestData("AAPL_daily.csv", 0.9)

trainDataLoader = convertToDataLoader(xTrain, yTrain, batchSize=32, shuffle=True, dropLast=True)
testDataLoader = convertToDataLoader(xTest, yTest, batchSize=32, shuffle=False, dropLast=False)

model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(Epoch):
    print(f'Epoch: {epoch}')
    print(f'\tTraining...')
    model.train()

    for x, y in trainDataLoader:
        x = x.to(device=device)
        y = y.to(device=device)

        logit = model(x)
        loss =  model.computeLoss(logit[:,-1,:], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'\tTesting...')
    with torch.no_grad():
        model.eval()
        testLoss = 0.

        for x, y in testDataLoader:

            x = x.to(device=device)
            y = y.to(device=device)

            logit = model(x)
            loss =  model.computeLoss(logit[:,-1,:], y)
            testLoss += loss

    print(f'Test loss for epoch {epoch}: {testLoss}')


