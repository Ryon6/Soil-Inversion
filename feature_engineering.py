"""
TODO: 光谱微分变换，特征波段选择
"""
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

def feature_select(X, y, dim=20, method='rf'):
    """
    特征选择方法
    :param X: 特征数据集
    :param y: 目标变量
    :param dim: 要选择的特征数量
    :return: 选择后的特征数据和对应的索引
    """
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

    else: return

    # 对互信息进行排序，并获取最高的dim个索引，选择对应的特征数据
    selected_x = X[:, selected_indices]

    return selected_x, selected_indices

def first_order_differential(hsi, wavelengths):
    """
    :param hsi: 高光谱图像
    :param wavelengths: 波段中间波长信息
    :return: 变换结果
    """
    delta_lambda = wavelengths[1] - wavelengths[0]
    first_diff = (hsi[1:] - hsi[:-1]) / (2 * delta_lambda)
    first_diff = first_diff[1:]

    return first_diff



def second_order_differential(hsi):
    """

    光谱二阶微分变换
    :param hsi: 高光谱图像
    :return:变换结果
    """
    return hsi

if __name__ == '__main__':
    from load_data import load_mining_region_data, load_cultivated_land_data
    from model.machine_learning import ml_model_test
    import matplotlib.pyplot as plt

    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T

    # 评估特征数量从3到41的r2值
    dims = range(3, 42, 2)
    r2_list = []
    for dim in dims:
        feature, _ = feature_select(samples_spectral, som_content, dim, method='rf')
        rmse_values, r2_values = ml_model_test(feature, som_content, plot=False)
        r2_list.append(max(r2_values))

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
