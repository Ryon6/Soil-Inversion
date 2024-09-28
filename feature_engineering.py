"""
TODO: feature_select 重构：过滤法，包裹法，嵌入法
TODO: 噪声波段去除
TODO: PCA降维
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from dwt import wavelet_denoising
from load_data import load_mining_region_data
from model.machine_learning import ml_model_test


def feature_select(X, y, dim=20, method='rf'):
    """
    特征选择方法
    :param method:
    :param X: 特征数据集
    :param y: 目标变量
    :param dim: 要选择的特征数量
    :return: 选择后的特征数据和对应的索引
    """
    # X = (X - X.min()) / (X.max() - X.min())
    # 计算互信息
    if method == 'mi':
        scores = mutual_info_regression(X, y)
        selected_indices = np.argsort(scores)[-dim:]
    elif method == 'rf':
        # 训练随机森林模型，获取特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        print(rf.score(X, y))
        scores = rf.feature_importances_
        # plt.plot(scores)
        # plt.show()
        selected_indices = np.argsort(scores)[-dim:]
    elif method == 'LASSO':
        lasso = Lasso()  # alpha参数控制正则化的强度
        lasso.fit(X, y)
        print(lasso.score(X, y))
        # 获取特征系数
        scores = lasso.coef_
        selected_indices = np.argsort(scores)[-dim:]
    elif method == 'person':
        Y1 = pd.Series(y)
        scores = [pd.Series(X[:, i]).corr(Y1) for i in range(X.shape[1])]
        selected_indices = np.argsort(scores)[-dim:]
    else:
        return

    # 对互信息进行排序，并获取最高的dim个索引，选择对应的特征数据
    selected_x = X[:, selected_indices]

    return selected_x, selected_indices


def first_order_differential(hsi, wavelengths, axis=1):
    """
    :param hsi: 高光谱图像，shape = (n, B), (B,h,w)
    :param wavelengths: 波段中间波长信息
    :param axis: 指定差分的维度，0表示行（波段）
    :return: 变换结果
    """
    # 计算波长间隔
    delta_lambda = np.diff(wavelengths)
    diff_hsi = np.diff(hsi, axis=axis)
    shape = np.ones(hsi.ndim, dtype=np.int16)
    shape[axis] = diff_hsi.shape[axis]

    delta_lambda = np.reshape(delta_lambda, shape)
    # 计算一阶微分
    first_diff = diff_hsi / (2 * delta_lambda)

    return first_diff


def feature_select_test(X, y, method='mi', models=None, dims=range(3, 42, 2), plot=False):
    # 评估特征数量为dims的r2值
    r2_list = []
    indices_list = []
    for dim in dims:
        feature, indices = feature_select(X, y, dim, method=method)
        r2_tmp = []
        for i in range(5):
            rmse_values, r2_values = ml_model_test(feature, y, models=models, plot=False)
            r2_tmp.append(max(r2_values))
        r2_list.append(np.mean(r2_tmp))
        indices_list.append(indices)

    if plot:
        # 绘制折线图
        plt.plot(dims, r2_list, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('R^2 Score')
        plt.title('R^2 Score vs Number of Features')
        plt.xticks(dims)
        plt.grid(True)
        plt.show()

    # 找出最优的特征数量
    optimal_dim = dims[np.argmax(r2_list)]
    print(f'Optimal number of features: {optimal_dim}')
    print(f'Optimal R-square: {np.max(r2_list)}')
    print(f'Optimal R-square: {indices_list[np.argmax(r2_list)]}')


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T
    y = zn_content

    # 光谱微分变换
    X = first_order_differential(samples_spectral, wavelengths, axis=1)
    img_array = first_order_differential(img_array, wavelengths, axis=0)

    # 离散小波变换
    X = wavelet_denoising(X.T, 'db4', 5).T
    img_array = wavelet_denoising(img_array, 'db4', 5)

    # para_search(samples_spectral, som_content)
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=4000, alpha=0.001,
                                     learning_rate_init=0.001),
    }
    # models = {'Lasso': Lasso()}
    # models = {'SVR': SVR(C=8, epsilon=0.001, gamma=0.01)}
    models = {'RF': RandomForestRegressor()}
    # feature, indices = feature_select(X, y, 20, method='rf')
    feature_select_test(X, y, method='rf', models=models, dims=range(4, 42), plot=True)
    # indices = [286,7,105,133,8,290,138,190,195,127,193,295,166,291,117]
    # X = X[:, indices]
    # r2_list = []
    # for i in range(10):
    #     rmse_values, r2_values = ml_model_test(X, som_content, models=models, plot=False)
    #     r2_list.append(r2_values)
    #     print(r2_values)
    # print(np.mean(r2_list))


if __name__ == '__main__':
    main()
