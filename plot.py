import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

from dwt import wavelet_denoising
from feature_engineering import first_order_differential
from load_data import load_mining_region_data, load_cultivated_land_data


def relevance(X, y, wavelengths, y_label):
    # X1 = pd.Series(X)
    Y1 = pd.Series(y)
    scores = mutual_info_regression(X, y)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    scores = rf.feature_importances_
    scores = [pd.Series(X[:, i]).corr(Y1) for i in range(X.shape[1])]

    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    x_max = X[np.argmax(y), :]
    x_min = X[np.argmin(y), :]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('wavelengths/nm')
    ax1.set_ylabel('reflectance')
    ax1.plot(wavelengths, x_max, color='tab:blue', label='highest {}'.format(y_label))
    ax1.plot(wavelengths, x_min, color='tab:red', label='lowest {}'.format(y_label))
    plt.legend(loc='upper right', bbox_to_anchor=(0.8, 1))

    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    ax2.set_ylabel('mutual information')
    ax2.plot(wavelengths, scores, color='black', label='relevance')

    plt.legend(loc='upper left', bbox_to_anchor=(0.1, 1))
    plt.show()


def transform(X, wavelengths, index=0):
    # 光谱微分变换
    X_diff = first_order_differential(X, wavelengths, axis=1)

    # 离散小波变换
    X_diff_denoised = wavelet_denoising(X.T, 'db4', 4).T

    # 创建一个具有3个子图的图
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # 绘制原始光谱
    axs[0].plot(wavelengths, X[index, :])
    axs[0].set_title('original spectra')
    axs[0].set_xlabel('wavelengths')
    axs[0].set_ylabel('reflectance')

    # 绘制微分变换曲线
    axs[1].plot(wavelengths[0:-1], X_diff[index, :])
    axs[1].set_title('first_order_differential')
    axs[1].set_xlabel('wavelengths')
    axs[1].set_ylabel('reflectance')

    # 绘制降噪后的微分变换曲线
    axs[2].plot(wavelengths[0:-1], X_diff_denoised[index, :])
    axs[2].set_title('discrete wavelet transform')
    axs[2].set_xlabel('wavelengths')
    axs[2].set_ylabel('reflectance')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图像
    plt.show()

def main():
    img_array, samples_spectral, salt_content, som_content, wavelengths = load_cultivated_land_data(
        need_wavelengths=True)
    # img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = som_content
    # relevance(X, y, wavelengths)

    # 光谱微分变换
    X_diff = first_order_differential(X, wavelengths, axis=1)

    # 离散小波变换
    X_denoised = wavelet_denoising(X.T, 'db4', 3).T

    relevance(X_denoised[:, 0:-1], som_content, wavelengths, 'SOM content')
    relevance(X_denoised[:, 0:-1], salt_content, wavelengths, 'salt content')


if __name__ == '__main__':
    main()
