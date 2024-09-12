from load_data import load_mining_region_data


# from sklearn.neural_network import MLPRegressor
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.svm import SVR
# from sklearn.linear_model import Lasso
# img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
# X = samples_spectral.T
# y = zn_content
#
# # 替换SVC为MLPRegressor
# model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
# model = SVR()
# model = Lasso()
#
# # RFECV配置
# rfecv = RFECV(estimator=model,  # 基学习器
#               min_features_to_select=2,  # 最小特征数
#               step=1,  # 每步移除的特征数
#               cv=StratifiedKFold(2),  # 交叉验证
#               scoring='neg_mean_squared_error',  # 评分函数
#               verbose=0,
#               n_jobs=1)
#
# # 拟合数据
# rfecv.fit(X, y)
#
# # 获取特征选择后的数据
# X_RFECV = rfecv.transform(X)
#
# # 打印结果
# print("RFECV特征选择结果——————————————————————————————————————————————————")
# print("有效特征个数 : %d" % rfecv.n_features_)
# print("全部特征等级 : %s" % list(rfecv.ranking_))

def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = zn_content

    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.ensemble import RandomForestRegressor

    # 假设X是特征矩阵，y是目标变量
    # 创建线性回归模型
    estimator = Lasso()

    # 创建RFECV对象
    rfecv = RFECV(estimator=estimator, step=1, cv=KFold(3), scoring='r2')

    # 拟合数据
    rfecv.fit(X, y)

    # 查看最佳特征数量
    print("Optimal number of features : %d" % rfecv.n_features_)

    # 获取特征重要性
    print("Feature ranking: %s" % list(rfecv.ranking_))

    import matplotlib.pyplot as plt
    X_RFECV = rfecv.transform(X)
    print(X_RFECV.shape)
    plt.plot(X[0, :])
    plt.plot(X_RFECV[0, :])
    plt.show()


if __name__ == '__main__':
    main()
