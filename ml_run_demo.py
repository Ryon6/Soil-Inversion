import numpy as np
import matplotlib.pyplot as plt

from load_data import load_cultivated_land_data, load_mining_region_data
from model.machine_learning import ml_model_test
from feature_engineering import feature_select, first_order_differential

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T
    y = zn_content

    # 光谱微分变换
    samples_spectral = first_order_differential(samples_spectral, wavelengths, axis=1)
    img_array = first_order_differential(img_array, wavelengths, axis=0)

    for dim in range(1, 41, 1):
        print('特征波段数', dim)
        # 特征选择
        X, indices = feature_select(samples_spectral, y, dim, method='LASSO')
        hsi = img_array[indices, :, :]

        models = {
            # 'SVR': SVR(C=8, epsilon=0.001, gamma=0.01),
            'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=1000, alpha=0.001,
                                         learning_rate_init=0.1),
            # 'RandomForestRegressor': RandomForestRegressor(max_depth=10, max_features='log2', min_samples_leaf=1,
            #                                                min_samples_split=15, n_estimators=500)
        }

        y_pred = ml_model_test(X, y, models=models, hsi=hsi, plot=False)

        plt.imshow(y_pred, cmap='viridis')  # 选择合适的颜色映射
        plt.colorbar(fraction=0.046, pad=0.04)  # 设置colorbar比例和位置

        # 获取y_pred的最小值和最大值
        vmin, vmax = np.min(y_pred), np.max(y_pred)
        # 设置colorbar范围，可以根据实际情况调整
        plt.clim(vmin, vmax)

        plt.title('dim={}'.format(dim))
        plt.show()
