import numpy as np

features_100ps = np.zeros((10001, 104))

lam=15
gamma=275
temperature=155

def time1(t,k):
    #return 1 / (1 + 15 * np.exp(-0.02 * (70 * t + 4 * k - 1)))
    return 1/14*(np.tanh(t/4+k/35)+13 / (1 + 15 * np.exp(-0.02 * (40*t + 4 * k - 1))))
    #return  1 / (1 + 15 * np.exp(-0.02 * (70 * t + 4 * k - 1)))
for levels in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
    """能级"""
    for i in range(features_100ps.shape[0]):
        features_100ps[i, 0] = levels
        features_100ps[i, 1] = lam / 100
        features_100ps[i, 2] = gamma / 1000
        features_100ps[i, 3] = temperature / 1000
        for j in range(4,104):
            features_100ps[i, j] = time1(i / 100, j - 4)

    np.save(f'100ps_{levels}.npy', features_100ps)



'''data=np.load(r'G:\读研\研三工作\工作二程序AIQD-XLSTM版\测试集上预测数据\energy_levels_population_30_169_286.npy')
data2=np.load('cnn的预测结果.npy')

print(data2.shape)
#plt.show()
def time(t,k):
    #return 1 / (1 + 15 * np.exp(-0.02 * (70*t + 4 * k - 1)))
    return np.tanh(t/2+k/4)

t=np.linspace(0,3000,301)
k=np.linspace(0,5,9)
f_k=[]
for i in k:
    f_k.append(time(t/1000,i))

data_new=np.zeros((data.shape[0]*7,13))
data_new[:,0]=data2[:,-1,0]
print(data_new.shape)
for i in range(int(data_new.shape[0]/301)):
    data_new[301 * i:301 * (i + 1), 1] = f_k[0]
    data_new[301 * i:301 * (i + 1), 2] = f_k[1]
    data_new[301 * i:301 * (i + 1), 3] = f_k[2]
    data_new[301 * i:301 * (i + 1), 4] = f_k[3]
    data_new[301 * i:301 * (i + 1), 5] = f_k[4]
    data_new[301 * i:301 * (i + 1), 6] = f_k[5]
    data_new[301 * i:301 * (i + 1), 7] = f_k[6]
    data_new[301 * i:301 * (i + 1), 8] = f_k[7]
    data_new[301 * i:301 * (i + 1), 9] = f_k[8]
    data_new[301 * i:301 * (i + 1), 10] = data[:,0]/max(data[:,0])
    data_new[301 * i:301 * (i + 1), 11] = data[:, 1]/max(data[:,1])
    data_new[301 * i:301 * (i + 1), 12] = data[:, 2]/max(data[:,2])

np.save('test_xlstm_input.npy',data_new)'''
