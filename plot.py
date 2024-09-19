import matplotlib.pyplot as plt

from dwt import wavelet_denoising
from feature_engineering import first_order_differential
from load_data import load_mining_region_data


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    X = samples_spectral.T
    y = som_content

    # 光谱微分变换
    X_diff = first_order_differential(X, wavelengths, axis=1)

    # 离散小波变换
    X_diff_denoised = wavelet_denoising(X_diff.T, 'db4', 5).T

    index = 4
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


if __name__ == '__main__':
    main()
