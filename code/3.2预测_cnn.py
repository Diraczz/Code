import matplotlib.pyplot as plt

from xlstm_function import *
def predict(model_cnn, x_input, device):
    with torch.no_grad():
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
        cnn_features = model_cnn(x_tensor)
    return cnn_features.cpu().numpy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
input_size = 104
cnn_output_size = 13

# 初始化融合模型
model_cnn = CNNFeatureExtractor(input_size, cnn_output_size).to(device).to(torch.float32)
for levels in [1,2,3,4,5,6,7]:
    # 加载权重
    model_cnn.load_state_dict(torch.load('checkpoints_cnn/cnn-epoch-303-vloss-9.164e-06.pth'))
    model_cnn.eval()

    npy_file = f'100ps_{levels/10}.npy'
    x_data = np.load(npy_file)
    predictions=[]
    for i in range(0,x_data.shape[0]):
        x_pred=x_data[i,:].reshape(1,x_data.shape[1],1)
        predictions.append(predict(model_cnn, x_pred, device))
    np.save(f'cnn的预测结果_100ps_{levels}.npy', predictions)