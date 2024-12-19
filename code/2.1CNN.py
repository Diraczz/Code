import torch.optim as optim
import os
from xlstm_function import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载与预处理
def data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_val = np.load('x_test.npy')
    y_val = np.load('y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    return x_train, y_train, x_val, y_val


x_train, y_train, x_val, y_val = data()
print(x_train.shape)
print('------------------')
print(y_train.shape)
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


input_size = x_train.shape[1]
cnn_output_size = 13


# 创建模型
cnn_model = CNNFeatureExtractor(input_size, cnn_output_size).to(device)
#xlstm_model = xLSTMModule(cnn_output_size, xlstm_head_size, xlstm_num_heads, output_size).to(device)

# 第一步：训练CNN部分
best_model_path = train_cnn(cnn_model, train_loader, val_loader, epochs=500)

# 第二步：生成CNN特征数据，使用最优模型
train_features, train_targets = generate_cnn_features(cnn_model, train_loader, best_model_path)


'''
# 转换为DataLoader
feature_train_dataset = TensorDataset(train_features, train_targets)
feature_train_loader = DataLoader(feature_train_dataset, batch_size=batch_size, shuffle=True)

# 第三步：训练xLSTM部分
train_xlstm(xlstm_model, feature_train_loader, epochs=200)'''



