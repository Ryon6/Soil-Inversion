import numpy as np

from load_data import load_mining_region_data
from dwt import wavelet_denoising
from feature_engineering import first_order_differential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = zn_content

    # 光谱微分变换
    X = first_order_differential(X, wavelengths, axis=1)
    img_array = first_order_differential(img_array, wavelengths, axis=0)

    # 离散小波变换
    # X = wavelet_denoising(X.T, 'db4', 4).T
    # img_array = wavelet_denoising(img_array, 'db4', 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.feature_selection import RFECV, RFE
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt

    # score = []
    # for i in range(1, 50, 4):
    #     x_wrapper = RFE(RandomForestRegressor(), n_features_to_select=i, step=4).fit_transform(X, y)
    #     once = cross_val_score(RandomForestRegressor(), x_wrapper, y, cv=5).mean()
    #     score.append(once)
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(1, 50, 4), score)
    # plt.xticks(range(1, 50, 4))
    # plt.show()


    # 假设X是特征矩阵，y是目标变量
    # 创建线性回归模型
    estimator = Lasso()
    # estimator = DecisionTreeRegressor()

    # # 创建RFECV对象
    # rfecv = RFECV(estimator=estimator, verbose=0, step=1, cv=KFold(3), scoring='r2', min_features_to_select=8)
    #
    # # 拟合数据
    # rfecv.fit(X, y)
    # X_RFECV = rfecv.transform(X)
    #
    # from ml_run_demo import ml_model_test
    # models = {
    #     'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 200, 100), max_iter=4000, alpha=0.001,
    #                                  learning_rate_init=0.001),
    # }
    # print(ml_model_test(X, y, models=models))
    # print(ml_model_test(X_RFECV, y, models=models))
    #
    # # 查看最佳特征数量
    # print("Optimal number of features : %d" % rfecv.n_features_)

    # import matplotlib.pyplot as plt
    # X_RFECV = rfecv.transform(X)
    #
    # print(np.where(rfecv.support_))
    # plt.plot(X[0, :] / np.max(X[0, :]))
    # plt.plot(list(rfecv.ranking_ / 305))
    # plt.show()




if __name__ == '__main__':
    main()
