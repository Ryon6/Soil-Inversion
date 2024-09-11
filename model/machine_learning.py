"""
用于测试机器学习模型效果
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def ml_model_test(X, y, hsi=None, models=None, plot=False, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if models is None:
        models = {
            # 'LinearRegression': LinearRegression(),
            # 'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=5, min_samples_split=2, max_features='sqrt',
            #                                                min_samples_leaf=1, splitter='best'),
            'MLPRegressor': MLPRegressor(hidden_layer_sizes=(20, 100, 20), max_iter=1000, alpha=0.001,
                                         learning_rate_init=0.05),
            # 'RandomForestRegressor': RandomForestRegressor(max_depth=10, max_features='log2', min_samples_leaf=1,
            #                                                min_samples_split=15, n_estimators=500)
        }

    # 训练并评估模型
    results = {}
    best_model_name = None
    best_r2 = -100
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[name] = {'rmse': rmse, 'r2': r2}
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # 提取MSE和R-squared数据
    rmse_values = [result['rmse'] for result in results.values()]
    r2_values = [result['r2'] for result in results.values()]

    if hsi is not None:
        hsi_shape = hsi.shape
        hsi = np.reshape(hsi, [hsi_shape[0], -1])

        # 测试集上表现最优的模型
        model = models[best_model_name]
        model.fit(X_train, y_train)
        print('最优模型', model.__class__)
        print('最优r2', best_r2)
        print('最优RMSE', np.max(rmse_values))
        hsi = scaler.transform(hsi.T)

        y_pred = model.predict(hsi)
        y_pred = np.reshape(y_pred, [hsi_shape[1], hsi_shape[2]])
        return y_pred

    if plot:
        # 设置matplotlib显示中文
        # plt.rcParams['font.sans-serif'] = ['AR PL UKai CN']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # 创建子图
        fig, ax1 = plt.subplots()

        # 绘制MSE折线图
        color = 'tab:blue'
        ax1.set_xlabel('model')
        ax1.set_ylabel('MSE', color=color)
        ax1.plot(list(models.keys()), rmse_values, color=color, label='MSE')
        ax1.tick_params(axis='y', labelcolor=color)

        # 创建第二个y轴，绘制R-squared折线图
        ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
        color = 'tab:red'
        ax2.set_ylabel('R-squared', color=color)
        ax2.plot(list(models.keys()), r2_values, color=color, label='squared')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.show()

    return rmse_values, r2_values


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
