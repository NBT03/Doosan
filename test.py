import numpy as np

def get_vector(q_near, q_new):
    # Tính vector giữa hai điểm trong không gian khớp
    vector = np.array(q_new) - np.array(q_near)
    return vector

# Ví dụ
q_near = [0.5, 0.3, -0.1, 0.2, 0.4, -0.5]  # Giá trị khớp hiện tại
q_new = [0.7, 0.4, -0.2, 0.3, 0.5, -0.3]  # Giá trị khớp mới

vector = get_vector(q_near, q_new)
print("Vector giữa q_near và q_new:", vector)
