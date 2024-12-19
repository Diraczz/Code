import matplotlib.pyplot as plt
import numpy as np
for levels in [1,2,3,4,5,6,7]:

    pred=np.load(f'cnn的预测结果_100ps_{levels}.npy')
    real=np.load('G:\读研\研三工作\工作二程序AIQD-XLSTM版\训练集内的预测数据\energy_levels_population_15_155_275.npy')
    #real2=np.load(r'G:\读研\研三工作\工作二程序AIQD-XLSTM版\all\energy_levels_population_14.0_150.0_272.0.npy')


    rho1_pred= pred[:, -1, 0]
    rho1_real=real[:,levels+2]
    plt.plot( np.linspace(0, 1, 101)[::7],rho1_pred[0:101][::7],'ro', markersize=5)
    plt.plot( np.linspace(1, 7, 601)[::20],rho1_pred[100:701][::20],'ro', markersize=5,label='Predicted')
    plt.plot(np.linspace(0,7,701),rho1_real[0:701],label='Real')
    plt.xlabel(r't/fs')
    plt.ylabel(r'$\rho_{%d%d}$' % (levels, levels))
    plt.title(f'lam={15},gamma={275},temperature={155}')
    plt.legend()
    plt.grid()
    plt.savefig(f"levels{levels},7ps.eps")
    plt.show()
    plt.close()

'''    plt.plot( np.linspace(0, 100, 10001)[::100],rho1_pred[::100],'ro', markersize=4,label='Predicted')
    plt.plot(np.linspace(0,100,10001),rho1_real,label='Real')
    plt.xlabel(r't/fs')
    plt.ylabel(r'$\rho_{%d%d}$' % (levels, levels))
    plt.title(f'lam={15},gamma={275},temperature={155}')
    plt.legend()
    plt.grid()
    plt.savefig(f"levels{levels},all.eps")
    plt.close()'''
