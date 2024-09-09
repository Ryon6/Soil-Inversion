"""
TODO: 光谱微分变换，特征波段选择
"""
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from load_data import load_mining_region_data, load_cultivated_land_data
from model.machine_learning import ml_model_test
import matplotlib.pyplot as plt
from sklearn.svm import SVR


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
        scores = rf.feature_importances_
        selected_indices = np.argsort(scores)[-dim:]
    elif method == 'LASSO':
        lasso = Lasso(alpha=0.2, max_iter=5000)  # alpha参数控制正则化的强度
        lasso.fit(X, y)
        # 获取特征系数
        scores = lasso.coef_
        selected_indices = np.argsort(np.abs(scores))[-dim:]
    else:
        return

    # 对互信息进行排序，并获取最高的dim个索引，选择对应的特征数据
    selected_x = X[:, selected_indices]

    return selected_x, selected_indices


def first_order_differential(hsi, wavelengths):
    """
    :param hsi: 高光谱图像
    :param wavelengths: 波段中间波长信息
    :return: 变换结果
    """
    delta_lambda = np.diff(wavelengths)
    first_diff = (hsi[:, 1:] - hsi[:, :-1]) / (2 * delta_lambda)

    return first_diff

def delete_all_zero(img_array, samples_spectral, wavelengths):
    # 检查全为0的波段，并剔除
    zero_bands = []
    for i in range(img_array.shape[0]):
        if np.all(img_array[i] == 0):
            zero_bands.append(i)
    img_array = np.delete(img_array, zero_bands, axis=0)

    samples_spectral = np.delete(samples_spectral, zero_bands, axis=1)
    wavelengths = np.delete(wavelengths, zero_bands)
    return img_array, samples_spectral, wavelengths


def second_order_differential(hsi):
    """

    光谱二阶微分变换
    :param hsi: 高光谱图像
    :return:变换结果
    """
    return hsi


def feature_select_test(X, y, method='mi', models=None, dims=range(3, 42, 2), plot=False):
    # 评估特征数量为dims的r2值
    r2_list = []
    for dim in dims:
        feature, _ = feature_select(X, y, dim, method=method)
        rmse_values, r2_values = ml_model_test(feature, y, models=models, plot=False)
        r2_list.append(max(r2_values))

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


def para_search(X, y):
    from sklearn.model_selection import GridSearchCV
    svr = SVR()
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 8],
        'epsilon': [0.1, 0.01, 0.001],
        # 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 0.01, 0.1, 0.05, 0.001]
    }
    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='r2')

    # 执行网格搜索
    grid_search.fit(X, y)
    # 打印最佳参数和最佳得分
    print(grid_search.best_params_)
    print(grid_search.best_score_)


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T

    # 检查全为0的波段，并剔除
    zero_bands = []
    for i in range(img_array.shape[0]):
        if np.all(img_array[i] == 0):
            zero_bands.append(i)
    img_array = np.delete(img_array, zero_bands, axis=0)

    samples_spectral = np.delete(samples_spectral, zero_bands, axis=1)
    wavelengths = np.delete(wavelengths, zero_bands)

    X = first_order_differential(samples_spectral, wavelengths)

    # para_search(samples_spectral, som_content)
    models = None
    # models = {'SVR': SVR(C=8, epsilon=0.001, gamma=0.01)}
    # models = {'RF': RandomForestRegressor()}
    feature_select_test(X, som_content, method='LASSO', models=models, dims=range(1, 42, 2), plot=True)


if __name__ == '__main__':
    main()
