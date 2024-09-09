import numpy as np

from load_data import load_cultivated_land_data, load_mining_region_data
from model.machine_learning import ml_model_test
from feature_engineering import feature_select, first_order_differential
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T

    y = som_content
    samples_spectral = first_order_differential(samples_spectral, wavelengths)
    samples_spectral, indices = feature_select(samples_spectral, y, 40, method='LASSO')

    his_shape = img_array.shape
    img_array = np.reshape(img_array,[his_shape[0],-1]).T
    img_array = first_order_differential(img_array, wavelengths)
    img_array = img_array[:, :]

    models = {
        'SVR': SVR(C=8, epsilon=0.001, gamma=0.01),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(20, 100, 20), max_iter=1000, alpha=0.001,
                                     learning_rate_init=0.05),
        'RandomForestRegressor': RandomForestRegressor(max_depth=10, max_features='log2', min_samples_leaf=1,
                                                       min_samples_split=15, n_estimators=500)
    }

    ml_model_test(samples_spectral, y, models=models, hsi=img_array, plot=False)
