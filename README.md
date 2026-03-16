# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.


  

## Design Steps


1.Data Preprocessing: Load the stock dataset, normalize the closing prices, and create sequential input data for the model.

2.Model Design: Build an RNN model with hidden layers to learn temporal patterns from the sequential stock price data.

3.Training and Prediction: Train the model using MSE loss and Adam optimizer, then use it to predict future stock prices and compare with actual values.


## Program
#### Name: KUKKADAPU CHARAN TEJ
#### Register Number: 212224040167
Include your code here:


```Python 
# Define RNN Model
class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel,self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50,1)

    def forward(self,x):

        out,_ = self.rnn(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out



model = RNNModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model

num_epochs = 20
train_losses = []

for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss/len(train_loader)

    train_losses.append(avg_loss)

    print(f'Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.6f}')
print("Name: Kukkadapu Charan Tej")
print("Register Number: 212224040167")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
model.eval()

with torch.no_grad():

    predicted = model(x_test_tensor.to(device)).cpu().numpy()

    actual = y_test_tensor.cpu().numpy()
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

print("Name: Kukkadapu Charan Tej")
print("Register Number: 212224040167")

plt.figure(figsize=(10,6))

plt.plot(actual_prices,label='Actual Price')
plt.plot(predicted_prices,label='Predicted Price')

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction using RNN")

plt.legend()
plt.show()

print("Predicted Price:", predicted_prices[-1])
print("Actual Price:", actual_prices[-1])







```

## Output

### True Stock Price, Predicted Stock Price vs time


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/dcec43d3-750c-4f88-9aab-a34d55690839" />



### Predictions 

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/dfd951dd-ec69-4c4f-865e-5f31f25c8fbb" />


## Result
The RNN model was successfully trained on historical stock price data and was able to predict future stock prices with reasonable accuracy.
