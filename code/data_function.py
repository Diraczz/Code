
from qutip import (
    Options,
    Qobj,
    basis,
    expect,
    liouvillian,
)
from qutip.nonmarkov.heom import (
    HEOMSolver,
    DrudeLorentzBath,
)
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np
import glob
import os
"""
            考虑到各个能级的布居在1500fs(0.15ps)内波动较大，因此将0.05ps内的时间特征的函数定义为：
                                1/(1 + 15 Exp[-0.02 (t + 5 k - 1)])
                                    将0.15ps之后的时间特征的函数定义为：
                                1/(1 + 15 Exp[-0.06 (t + 5 k - 1)])
"""
def dataset(times,timesteps,train_or_test,t1,k1):
    # System Hamiltonian:
    Hsys = 3e10 * 2 * np.pi * Qobj([
        [200, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
        [-87.7, 320, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
        [-5.9, 8.2, -53.5, 110, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 270, 81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0, 81.1, 420, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 230],
    ])

    def time(t,k):
        #return 1 / (1 + 15 * np.exp(-0.02 * (t1*t + k1 * k - 1)))
        return 1/14*(np.tanh(t/t1+k/k1)+13 / (1 + 15 * np.exp(-0.02 * (40*t + 4 * k - 1))))

    def cot(x):
        """ Vectorized cotangent of x. """
        return 1 / np.tan(x)

    def J0(energy, lam, gamma):
        """ Under-damped brownian oscillator spectral density. """
        return 2 * lam * gamma * energy / (energy**2 + gamma**2)

    def J0_dephasing(lam, gamma):
        """ Under-damped brownian oscillator dephasing probability.
            This returns the limit as w -> 0 of J0(w) * n_th(w, T) / T.
        """
        return 2 * lam * gamma / gamma**2

    def n_th(energy, T):
        """ The average occupation of a given energy level at temperature T. """
        return 1 / (np.exp(energy / T) - 1)

    def run_simulation(lam1, temperature, gamma1):
        # Parameters
        lam = lam1 * 3e10 * 2 * np.pi
        gamma = 1 / (gamma1 * 1e-15)
        T = temperature * 0.6949 * 3e10 * 2 * np.pi

        # 时间范围（单位为秒）
        tlist = np.linspace(0, times*1e-12, timesteps)  # 总时间点 0 到 3 ps
        time_fs = tlist * 1e15  # 转换为飞秒（fs）

        # We start the excitation at site 1:
        rho0 = basis(7, 0) * basis(7, 0).dag()

        # HEOM solver options:
        options = Options(nsteps=10000, store_states=True, max_step=1000)
        NC = 3  # Reduced precision for faster results
        Nk = 0

        Q_list = []
        baths = []
        Ltot = liouvillian(Hsys)
        for m in range(7):
            Q = basis(7, m) * basis(7, m).dag()
            Q_list.append(Q)
            baths.append(
                DrudeLorentzBath(
                    Q, lam=lam, gamma=gamma, T=T, Nk=Nk, tag=str(m)
                )
            )
            _, terminator = baths[-1].terminator()
            Ltot += terminator

        HEOMMats = HEOMSolver(Hsys, baths, NC, options=options)

        # Run HEOM simulation
        outputFMO_HEOM = HEOMMats.run(rho0, tlist)

        # Collect populations
        populations = []
        for m in range(7):
            Q = basis(7, m) * basis(7, m).dag()
            pop = np.real(expect(outputFMO_HEOM.states, Q))
            populations.append(pop)

        # 转换为 NumPy 数组
        populations_array = np.array(populations).T  # 转置使每列对应一个能级

        # 计算附加的 100 列 time1 和 time2
        extra_columns = []
        for idx, t in enumerate(time_fs):
            extra_columns.append([time(t / 1000, k) for k in np.linspace(0, 99, 100)])
        extra_columns = np.array(extra_columns)

        # 将 lam, gamma, T 添加到每行的前三列
        params = np.array([[lam1, gamma1, temperature]] * populations_array.shape[0])

        # 合并 lam, gamma, T，populations_array 和 extra_columns
        result_array = np.hstack((params, populations_array, extra_columns))



        file_name = f"{train_or_test}/energy_levels_population_{lam1}_{temperature}_{gamma1}.npy"
        np.save(file_name, result_array)




    if train_or_test == "train":
        lam_range = np.linspace(14, 28, 15)
        temp_range = np.linspace(150, 164, 15)
        gamma_range = np.linspace(270, 284, 15)
    elif train_or_test == "test":
        lam_range = np.linspace(28, 30, 3)
        temp_range = np.linspace(164, 170, 7)
        gamma_range = np.linspace(284, 290, 7)

    Parallel(n_jobs=12)(
        delayed(run_simulation)(lam1, temperature, gamma1)
        for lam1 in lam_range
        for temperature in temp_range
        for gamma1 in gamma_range
    )

    '''Parallel(n_jobs=12)(
        delayed(run_simulation)(lam1,temperature,gamma1)
        for lam1 in np.linspace(28,30,3)
        for temperature in np.linspace(164,170,7)
        for gamma1 in np.linspace(284,290,7)
    )'''

    '''Parallel(n_jobs=12)(
        delayed(run_simulation)(lam1,temperature,gamma1)
        for lam1 in [15]
        for temperature in [155]
        for gamma1 in [275]
    )'''

def datamaker(test_or_train):
    # 加载所有文件
    all_files = {}
    datapath = f"{test_or_train}"
    for files in glob.glob(datapath + '/*.np[yz]'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
    file_count = len(all_files)

    if test_or_train=='train':
        rows_per_file = 301
    else:
        rows_per_file = 1001

    """能级 1"""
    data_x1 = np.zeros((rows_per_file * file_count, 104))
    data_y1 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 全局索引初始化

    for file_name in tqdm(all_files.keys(), desc="Processing Level 1"):
        for i in range(rows_per_file):
            data_x1[global_idx, 0] = 0.1  # 标记能级 1
            data_x1[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x1[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x1[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y1[global_idx, 0] = all_files[file_name][i, 3]
            data_y1[global_idx, 1] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 4])
            data_y1[global_idx, 2] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 5])
            data_y1[global_idx, 3] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 6])
            data_y1[global_idx, 4] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 7])
            data_y1[global_idx, 5] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 8])
            data_y1[global_idx, 6] = abs(all_files[file_name][i, 3] - all_files[file_name][i, 9])
            data_y1[global_idx, 7] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 4])
            data_y1[global_idx, 8] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 5])
            data_y1[global_idx, 9] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 6])
            data_y1[global_idx, 10] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 7])
            data_y1[global_idx, 11] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 8])
            data_y1[global_idx, 12] = abs(all_files[file_name][i, 3] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x1[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 2"""
    data_x2 = np.zeros((rows_per_file * file_count, 104))
    data_y2 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 2"):
        for i in range(rows_per_file):
            data_x2[global_idx, 0] = 0.2  # 标记能级 2
            data_x2[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x2[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x2[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y2[global_idx, 0] = all_files[file_name][i, 4]
            data_y2[global_idx, 1] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 3])
            data_y2[global_idx, 2] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 5])
            data_y2[global_idx, 3] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 6])
            data_y2[global_idx, 4] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 7])
            data_y2[global_idx, 5] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 8])
            data_y2[global_idx, 6] = abs(all_files[file_name][i, 4] - all_files[file_name][i, 9])
            data_y2[global_idx, 7] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 3])
            data_y2[global_idx, 8] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 5])
            data_y2[global_idx, 9] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 6])
            data_y2[global_idx, 10] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 7])
            data_y2[global_idx, 11] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 8])
            data_y2[global_idx, 12] = abs(all_files[file_name][i, 4] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x2[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 3"""
    data_x3 = np.zeros((rows_per_file * file_count, 104))
    data_y3 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 3"):
        for i in range(rows_per_file):
            data_x3[global_idx, 0] = 0.3
            data_x3[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x3[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x3[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y3[global_idx, 0] = all_files[file_name][i, 5]
            data_y3[global_idx, 1] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 3])
            data_y3[global_idx, 2] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 4])
            data_y3[global_idx, 3] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 6])
            data_y3[global_idx, 4] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 7])
            data_y3[global_idx, 5] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 8])
            data_y3[global_idx, 6] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 9])
            data_y3[global_idx, 7] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 3])
            data_y3[global_idx, 8] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 4])
            data_y3[global_idx, 9] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 6])
            data_y3[global_idx, 10] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 7])
            data_y3[global_idx, 11] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 8])
            data_y3[global_idx, 12] = abs(all_files[file_name][i, 5] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x3[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 4"""
    data_x4 = np.zeros((rows_per_file * file_count, 104))
    data_y4 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 4"):
        for i in range(rows_per_file):
            data_x4[global_idx, 0] = 0.4
            data_x4[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x4[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x4[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y4[global_idx, 0] = all_files[file_name][i, 6]
            data_y4[global_idx, 1] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 3])
            data_y4[global_idx, 2] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 4])
            data_y4[global_idx, 3] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 5])
            data_y4[global_idx, 4] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 7])
            data_y4[global_idx, 5] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 8])
            data_y4[global_idx, 6] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 9])
            data_y4[global_idx, 7] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 3])
            data_y4[global_idx, 8] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 4])
            data_y4[global_idx, 9] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 5])
            data_y4[global_idx, 10] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 7])
            data_y4[global_idx, 11] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 8])
            data_y4[global_idx, 12] = abs(all_files[file_name][i, 6] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x4[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 5"""
    data_x5 = np.zeros((rows_per_file * file_count, 104))
    data_y5 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 5"):
        for i in range(rows_per_file):
            data_x5[global_idx, 0] = 0.5
            data_x5[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x5[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x5[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y5[global_idx, 0] = all_files[file_name][i, 7]
            data_y5[global_idx, 1] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 3])
            data_y5[global_idx, 2] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 4])
            data_y5[global_idx, 3] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 5])
            data_y5[global_idx, 4] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 6])
            data_y5[global_idx, 5] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 8])
            data_y5[global_idx, 6] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 9])
            data_y5[global_idx, 7] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 3])
            data_y5[global_idx, 8] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 4])
            data_y5[global_idx, 9] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 5])
            data_y5[global_idx, 10] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 6])
            data_y5[global_idx, 11] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 8])
            data_y5[global_idx, 12] = abs(all_files[file_name][i, 7] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x5[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 6"""
    data_x6 = np.zeros((rows_per_file * file_count, 104))
    data_y6 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 6"):
        for i in range(rows_per_file):
            data_x6[global_idx, 0] = 0.6
            data_x6[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x6[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x6[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y6[global_idx, 0] = all_files[file_name][i, 8]
            data_y6[global_idx, 1] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 3])
            data_y6[global_idx, 2] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 4])
            data_y6[global_idx, 3] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 5])
            data_y6[global_idx, 4] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 6])
            data_y6[global_idx, 5] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 7])
            data_y6[global_idx, 6] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 9])
            data_y6[global_idx, 7] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 3])
            data_y6[global_idx, 8] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 4])
            data_y6[global_idx, 9] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 5])
            data_y6[global_idx, 10] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 6])
            data_y6[global_idx, 11] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 7])
            data_y6[global_idx, 12] = abs(all_files[file_name][i, 8] + all_files[file_name][i, 9])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x6[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    """能级 7"""
    data_x7 = np.zeros((rows_per_file * file_count, 104))
    data_y7 = np.zeros((rows_per_file * file_count, 13))
    global_idx = 0  # 重置全局索引

    for file_name in tqdm(all_files.keys(), desc="Processing Level 7"):
        for i in range(rows_per_file):
            data_x7[global_idx, 0] = 0.7
            data_x7[global_idx, 1] = all_files[file_name][i, 0] / 100
            data_x7[global_idx, 2] = all_files[file_name][i, 1] / 1000
            data_x7[global_idx, 3] = all_files[file_name][i, 2] / 1000

            data_y7[global_idx, 0] = all_files[file_name][i, 9]
            data_y7[global_idx, 1] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 3])
            data_y7[global_idx, 2] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 4])
            data_y7[global_idx, 3] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 5])
            data_y7[global_idx, 4] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 6])
            data_y7[global_idx, 5] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 7])
            data_y7[global_idx, 6] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 8])
            data_y7[global_idx, 7] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 3])
            data_y7[global_idx, 8] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 4])
            data_y7[global_idx, 9] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 5])
            data_y7[global_idx, 10] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 6])
            data_y7[global_idx, 11] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 7])
            data_y7[global_idx, 12] = abs(all_files[file_name][i, 9] + all_files[file_name][i, 8])

            for j in range(4, 104):  # 处理第 4 列到第 103 列
                data_x7[global_idx, j] = all_files[file_name][i, j + 6]

            global_idx += 1  # 更新全局索引

    '''       
    """能级3"""
    data_x3=np.zeros((3000*file_count,104))
    data_y3=np.zeros((3000*file_count,7))
    for file_name in all_files.keys():
        for i in np.linspace(0,3000,3001):
            i = int(i)
            data_x3[i, 0] = 0.3
            data_x3[i, 1] = all_files[file_name][i, 0]
            data_x3[i, 2] = all_files[file_name][i, 1]
            data_x3[i, 3] = all_files[file_name][i, 2]

            data_y3[i, 0] = all_files[file_name][i, 5]
            data_y3[i, 1] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 4])
            data_y3[i, 2] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 6])
            data_y3[i, 3] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 7])
            data_y3[i, 4] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 8])
            data_y3[i, 5] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 9])
            data_y3[i, 6] = abs(all_files[file_name][i, 5] - all_files[file_name][i, 10])
            for j in np.linspace(4,103,100):
                j=int(j)
                data_x3[i, j] = all_files[file_name][i, j]
    """能级4"""
    data_x4=np.zeros((3000*file_count,104))
    data_y4=np.zeros((3000*file_count,7))
    for file_name in all_files.keys():
        for i in np.linspace(0,3000,3001):
            i = int(i)
            data_x4[i, 0] = 0.4
            data_x4[i, 1] = all_files[file_name][i, 0]
            data_x4[i, 2] = all_files[file_name][i, 1]
            data_x4[i, 3] = all_files[file_name][i, 2]

            data_y4[i, 0] = all_files[file_name][i, 6]
            data_y4[i, 1] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 4])
            data_y4[i, 2] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 5])
            data_y4[i, 3] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 7])
            data_y4[i, 4] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 8])
            data_y4[i, 5] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 9])
            data_y4[i, 6] = abs(all_files[file_name][i, 6] - all_files[file_name][i, 10])
            for j in np.linspace(4,103,100):
                j=int(j)
                data_x4[i, j] = all_files[file_name][i, j]
    """能级5"""
    data_x5=np.zeros((3000*file_count,104))
    data_y5=np.zeros((3000*file_count,7))
    for file_name in all_files.keys():
        for i in np.linspace(0,3000,3001):
            i = int(i)
            data_x5[i, 0] = 0.5
            data_x5[i, 1] = all_files[file_name][i, 0]
            data_x5[i, 2] = all_files[file_name][i, 1]
            data_x5[i, 3] = all_files[file_name][i, 2]

            data_y5[i, 0] = all_files[file_name][i, 7]
            data_y5[i, 1] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 4])
            data_y5[i, 2] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 5])
            data_y5[i, 3] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 6])
            data_y5[i, 4] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 8])
            data_y5[i, 5] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 9])
            data_y5[i, 6] = abs(all_files[file_name][i, 7] - all_files[file_name][i, 10])
            for j in np.linspace(4,103,100):
                j=int(j)
                data_x5[i, j] = all_files[file_name][i, j]
    """能级6"""
    data_x6=np.zeros((3000*file_count,104))
    data_y6=np.zeros((3000*file_count,7))
    for file_name in all_files.keys():
        for i in np.linspace(0,3000,3001):
            i = int(i)
            data_x6[i, 0] = 0.6
            data_x6[i, 1] = all_files[file_name][i, 0]
            data_x6[i, 2] = all_files[file_name][i, 1]
            data_x6[i, 3] = all_files[file_name][i, 2]

            data_y6[i, 0] = all_files[file_name][i, 8]
            data_y6[i, 1] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 4])
            data_y6[i, 2] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 5])
            data_y6[i, 3] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 6])
            data_y6[i, 4] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 7])
            data_y6[i, 5] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 9])
            data_y6[i, 6] = abs(all_files[file_name][i, 8] - all_files[file_name][i, 10])
            for j in np.linspace(4,103,100):
                j=int(j)
                data_x6[i, j] = all_files[file_name][i, j]
    """能级7"""
    data_x7=np.zeros((3000*file_count,104))
    data_y7=np.zeros((3000*file_count,7))
    for file_name in all_files.keys():
        for i in np.linspace(0,3000,3001):
            i = int(i)
            data_x7[i, 0] = 0.7
            data_x7[i, 1] = all_files[file_name][i, 0]
            data_x7[i, 2] = all_files[file_name][i, 1]
            data_x7[i, 3] = all_files[file_name][i, 2]

            data_y7[i, 0] = all_files[file_name][i, 9]
            data_y7[i, 1] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 4])
            data_y7[i, 2] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 5])
            data_y7[i, 3] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 6])
            data_y7[i, 4] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 7])
            data_y7[i, 5] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 8])
            data_y7[i, 6] = abs(all_files[file_name][i, 9] - all_files[file_name][i, 10])
            for j in np.linspace(4,103,100):
                j=int(j)
                data_x7[i, j] = all_files[file_name][i, j]'''
    data_x = np.vstack((data_x1, data_x2, data_x3, data_x4, data_x5, data_x6, data_x7))
    data_y = np.vstack((data_y1, data_y2, data_y3, data_y4, data_y5, data_y6, data_y7))
    np.save(f'x_{test_or_train}', data_x)
    np.save(f'y_{test_or_train}', data_y)