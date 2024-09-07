import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import pandas as pd
import numpy as np

# Tạo dữ liệu giả định với các tọa độ x, y, z cho các quỹ đạo khác nhau
num_samples = 1000
time = np.linspace(0, 10, num_samples)

# Quỹ đạo thẳng
x_straight = time
y_straight = 2 * time
z_straight = 0.5 * time

# Quỹ đạo parabol
x_parabola = time
y_parabola = time**2
z_parabola = 0.5 * time**2

# Quỹ đạo sin
x_sin = np.sin(time)
y_sin = np.sin(2 * time)
z_sin = np.sin(0.5 * time)

# Tạo dataframe và lưu vào file CSV
data = pd.DataFrame({
    'x': np.concatenate([x_straight, x_parabola, x_sin]),
    'y': np.concatenate([y_straight, y_parabola, y_sin]),
    'z': np.concatenate([z_straight, z_parabola, z_sin])
})

data.to_csv('trajectories.csv', index=False)
# Load and normalize the dataset
data = pd.read_csv('trajectories.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['x', 'y', 'z']])

# Prepare the sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader for training and testing
train_size = int(0.8 * len(X_tensor))
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
input_size = 3
hidden_size = 50
num_layers = 2
output_size = 3

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
model.train()

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(targets.numpy())

predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(actuals)

# Calculate errors
mae = np.mean(np.abs(predictions - actuals))
mse = np.mean((predictions - actuals) ** 2)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
