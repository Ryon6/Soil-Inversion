"""
异常值去除 ->
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from dwt import wavelet_denoising
from feature_engineering import first_order_differential, feature_select
from load_data import load_mining_region_data, load_cultivated_land_data
from model.machine_learning import ml_model_test


def plot():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = som_content

    # 光谱微分变换
    X = first_order_differential(X, wavelengths, axis=1)
    img_array = first_order_differential(img_array, wavelengths, axis=0)

    # 离散小波变换
    X = wavelet_denoising(X.T, 'db4', 5).T
    img_array = wavelet_denoising(img_array, 'db4', 5)

    indices = [286, 7, 105, 133, 8, 290, 138, 190, 195, 127, 193, 295, 166, 291, 117]
    X = X[:, indices]
    hsi = img_array[indices, :, :]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = MLPRegressor()

    model.fit(X, y)

    hsi_shape = hsi.shape
    hsi = np.reshape(hsi, [hsi_shape[0], -1])
    hsi = scaler.transform(hsi.T)

    y_pred = model.predict(hsi)
    y_pred = np.reshape(y_pred, [hsi_shape[1], hsi_shape[2]])
    plt.imshow(y_pred, cmap='viridis')  # 选择合适的颜色映射
    plt.colorbar(fraction=0.046, pad=0.04)  # 设置colorbar比例和位置

    # 获取y_pred的最小值和最大值
    vmin, vmax = np.min(y_pred), np.max(y_pred)
    # 设置colorbar范围，可以根据实际情况调整
    plt.clim(vmin, vmax)

    plt.title('dim={}'.format(15))
    plt.show()



if __name__ == '__main__':
    output_folder = 'output'

    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = zn_content

    # 光谱微分变换
    X = first_order_differential(X, wavelengths, axis=1)
    hsi = first_order_differential(img_array, wavelengths, axis=0)

    # 离散小波变换
    X = wavelet_denoising(X.T, 'db4', 4).T
    hsi = wavelet_denoising(hsi, 'db4', 4)

    X, indices = feature_select(X, y, 3, method='rf')
    hsi = hsi[indices, :, :]
    # plt.plot(hsi[:,208,578])
    # plt.plot(X[0,:])
    # plt.show()

    # models = {
    #     'AdaBoostRegressor': AdaBoostRegressor(),
    #     'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=4000, alpha=0.001,
    #                                  learning_rate_init=0.001),
    #     'RandomForestRegressor': RandomForestRegressor(n_estimators=500)
    # }
    models = {
        'AdaBoostRegressor': AdaBoostRegressor(),
        'MLPRegressor': MLPRegressor(),
        'RandomForestRegressor': RandomForestRegressor()
    }

    # indices = [286, 7, 105, 133, 8, 290, 138, 190, 195, 127, 193, 295, 166, 291, 117]
    # X = samples_spectral[:, indices]
    # hsi = img_array[indices, :, :]

    y_pred = ml_model_test(X, y, models=models, hsi=hsi, plot=True)

    plt.imshow(y_pred, cmap='viridis')  # 选择合适的颜色映射
    plt.colorbar(fraction=0.046, pad=0.04)  # 设置colorbar比例和位置

    # 获取y_pred的最小值和最大值
    vmin, vmax = np.min(y_pred), np.max(y_pred)
    # 设置colorbar范围，可以根据实际情况调整
    plt.clim(vmin, vmax)

    plt.title('dim={}'.format(15))
    plt.show()

    # plt.savefig(os.path.join(output_folder, f"dim_{15}.png"), dpi=300)
    # plt.close()

    # for dim in range(4, 41, 1):
    #     print('特征波段数', dim)
    #     # 特征选择
    #     X, indices = feature_select(samples_spectral, y, dim, method='LASSO')
    #     hsi = img_array[indices, :, :]
    #
    #     models = {
    #         # 'SVR': SVR(C=8, epsilon=0.001, gamma=0.01),
    #         'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=1000, alpha=0.001,
    #                                      learning_rate_init=0.1),
    #         # 'RandomForestRegressor': RandomForestRegressor(max_depth=10, max_features='log2', min_samples_leaf=1,
    #         #                                                min_samples_split=15, n_estimators=500)
    #     }
    #
    #     y_pred = ml_model_test(X, y, models=models, hsi=hsi, plot=False)
    #
    #     plt.imshow(y_pred, cmap='viridis')  # 选择合适的颜色映射
    #     plt.colorbar(fraction=0.046, pad=0.04)  # 设置colorbar比例和位置
    #
    #     # 获取y_pred的最小值和最大值
    #     vmin, vmax = np.min(y_pred), np.max(y_pred)
    #     # 设置colorbar范围，可以根据实际情况调整
    #     plt.clim(vmin, vmax)
    #
    #     plt.title('dim={}'.format(dim))
    #     plt.savefig(os.path.join(output_folder, f"dim_{dim}.png"), dpi=300)
    #     plt.close()
