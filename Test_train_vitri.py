# lstm_prediction.py

import pybullet as p
import pybullet_data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Hàm tạo dữ liệu trong PyBullet
def generate_data():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    object_id = p.loadURDF("r2d2.urdf")
    positions = []

    for step in range(1000):
        # Di chuyển vật thể theo các quỹ đạo khác nhau
        x = step * 0.01
        y = np.sin(x)
        z = 0
        p.resetBasePositionAndOrientation(object_id, [x, y, z], [0, 0, 0, 1])
        position = p.getBasePositionAndOrientation(object_id)[0]
        positions.append(position)

    p.disconnect()
    return np.array(positions)


# Tạo và chuẩn bị dữ liệu
data = generate_data()
train_data = torch.tensor(data[:-1], dtype=torch.float32)
target_data = torch.tensor(data[1:], dtype=torch.float32)
dataset = TensorDataset(train_data, target_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Định nghĩa mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# Thiết lập thông số mô hình
model = LSTMModel(input_size=3, hidden_size=50, output_size=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 1000
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(1), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# Hàm dự đoán vị trí tương lai
def predict_future_positions(model, input_data, future_steps):
    model.eval()
    predictions = []
    input_seq = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    for _ in range(future_steps):
        with torch.no_grad():
            output = model(input_seq)
            predictions.append(output.squeeze(0).numpy())
            input_seq = output

    predictions = np.array(predictions)
    predictions = np.squeeze(predictions)  # Loại bỏ chiều thừa
    print(
        f"Shape of predictions after conversion: {predictions.shape}")  # Kiểm tra kích thước mảng sau khi loại bỏ chiều thừa
    return predictions
# Dự đoán 100 bước tiếp theo từ dữ liệu ban đầu
predicted_positions = predict_future_positions(model, data[-1:], future_steps=100)
print(f"Shape of predicted_positions: {predicted_positions.shape}")  # Kiểm tra kích thước mảng dự đoán


# Vẽ đồ thị so sánh giữa tọa độ thực tế và dự đoán
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Các hàm và định nghĩa model của bạn ở đây

def plot_comparison(actual_data, predicted_data, steps_to_plot=100):
    actual_x = actual_data[-steps_to_plot:, 0]
    actual_y = actual_data[-steps_to_plot:, 1]
    actual_z = actual_data[-steps_to_plot:, 2]

    predicted_x = predicted_data[:, 0]
    predicted_y = predicted_data[:, 1]
    predicted_z = predicted_data[:, 2]

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(range(steps_to_plot), actual_x, label='Thực tế', color='b')
    plt.plot(range(steps_to_plot), predicted_x, label='Dự đoán', color='r', linestyle='--')
    plt.title('So sánh tọa độ X theo thời gian')
    plt.xlabel('Thời gian (step)')
    plt.ylabel('Tọa độ X')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(steps_to_plot), actual_y, label='Thực tế', color='b')
    plt.plot(range(steps_to_plot), predicted_y, label='Dự đoán', color='r', linestyle='--')
    plt.title('So sánh tọa độ Y theo thời gian')
    plt.xlabel('Thời gian (step)')
    plt.ylabel('Tọa độ Y')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(range(steps_to_plot), actual_z, label='Thực tế', color='b')
    plt.plot(range(steps_to_plot), predicted_z, label='Dự đoán', color='r', linestyle='--')
    plt.title('So sánh tọa độ Z theo thời gian')
    plt.xlabel('Thời gian (step)')
    plt.ylabel('Tọa độ Z')
    plt.legend()

    plt.tight_layout()
    plt.savefig("comparison_plot.png")  # Lưu biểu đồ thành file PNG
    plt.close()  # Đóng biểu đồ sau khi lưu

# Gọi hàm train model và plot_comparison


# Gọi hàm vẽ đồ thị
plot_comparison(data, predicted_positions, steps_to_plot=1000)
