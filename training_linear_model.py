import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import preprocessing
from math import sqrt
from model import LinearNet
from dataset import filterDataFrame

def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        l2_reg: float,
) -> torch.nn.Module:
    """
    Train model using mini-batch SGD
    After each epoch, we evaluate the model on validation data

    :param net: initialized neural network
    :param train_loader: DataLoader containing training set
    :param n_epochs: number of epochs to train
    :param lr: learning rate (default: 0.001)
    :param l2_reg: L2 regularization factor (default: 0)
    :return: torch.nn.Module: trained model.
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Train Network
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:

            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            
            # Compute cost function
            batch_mse = criterion(outputs, labels)
            
            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + l2_reg * reg_loss

            # Backward propagation to compute gradient
            cost.backward()
            
            # Update parameters using gradient
            optimizer.step()
        
        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)
        print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
        
    return net

random_seed = 12345
df = filterDataFrame(pd.read_csv('AAPL_daily.csv'))

print(list(df.columns) )

fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# Choke valve opening
ax[0, 0].plot(df['1. open'], label='open')
ax[0, 0].plot(df['4. close'], '--r', label='close')
ax[0, 0].legend()

# Total flow through choke valve
ax[0, 1].plot(df['2. high'], label='high')
ax[0, 1].legend()

# Diff pressure over choke valve
ax[1, 0].plot(df['3. low'], label='low')
ax[1, 0].legend()

# Fractions
ax[1, 1].plot(df['5. volume'], label='volume')
ax[1, 1].legend()

print(df['4. close'])

#plt.show()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_set = df.iloc[4800:5225]

train_val_set = df.copy().drop(test_set.index)

val_set = train_val_set.sample(frac=0.1, replace=False, random_state=random_seed)

train_set = train_val_set.copy().drop(val_set.index)

n_points = len(train_set) + len(val_set) + len(test_set)
print(f'{len(df)} = {len(train_set)} + {len(val_set)} + {len(test_set)} = {n_points}')

normaliser_train = preprocessing.MinMaxScaler()
normaliser_val = preprocessing.MinMaxScaler()
normaliser_test = preprocessing.MinMaxScaler()

train_set = normaliser_train.fit_transform(train_set)
test_set = normaliser_test.fit_transform(test_set)
val_set = normaliser_val.fit_transform(val_set)


INPUT_COLS = [0, 1, 2, 4]
OUTPUT_COLS = [3]

x_train = torch.from_numpy(train_set[:, INPUT_COLS]).to(torch.float).to(device)
y_train = torch.from_numpy(train_set[:, OUTPUT_COLS]).to(torch.float).to(device)

x_val = torch.from_numpy(val_set[:, INPUT_COLS]).to(torch.float).to(device)
y_val = torch.from_numpy(val_set[:, OUTPUT_COLS]).to(torch.float).to(device)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_set), shuffle=False)

layers = [len(INPUT_COLS), 50, 50, len(OUTPUT_COLS)]
net = LinearNet(layers).to(device)

print(f'Layers: {layers}')
print(f'Number of model parameters: {net.get_num_parameters()}')
# print(6*50 + 50 + 50*50 + 50 + 50 * 1 + 1)


# # Train the model
# 
# Almost there. We only need to set some important hyper-parameters before we start the training. The number of epochs to train, the learning rate, and the L2 regularization factor.

# In[71]:


n_epochs = 20
lr = 0.001
l2_reg = 0.001  # 10
net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg)


# # Evaluate the model on validation data

# In[72]:


# Predict on validation data
pred_val = net(x_val)

# Compute MSE, MAE and MAPE on validation data
print('Error on validation data')

mse_val = torch.mean(torch.pow(pred_val - y_val, 2))
print(f'MSE: {mse_val.item()}')

mae_val = torch.mean(torch.abs(pred_val - y_val))
print(f'MAE: {mae_val.item()}')

mape_val = 100*torch.mean(torch.abs(torch.div(pred_val - y_val, y_val)))
print(f'MAPE: {mape_val.item()} %')

x_test = torch.from_numpy(test_set[:,INPUT_COLS]).to(torch.float).to(device)
y_test = torch.from_numpy(test_set[:,OUTPUT_COLS]).to(torch.float).to(device)

# Make prediction
pred_test = net(x_test)

# Compute MSE, MAE and MAPE on test data
print('Error on test data')

mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
print(f'MSE: {mse_test.item()}')

mae_test = torch.mean(torch.abs(pred_test - y_test))
print(f'MAE: {mae_test.item()}')

mape_test = 100*torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
print(f'MAPE: {mape_test.item()} %')

y_test = y_test.to('cpu')
pred_test = pred_test.to('cpu')

plt.figure(figsize=(16, 9))
plt.plot(y_test.numpy(), label='True close')
plt.plot(pred_test.detach().numpy(), label='Estimated close')
plt.legend()
plt.show()