# from feature_engineering import first_order_differential
import matplotlib.pyplot as plt
import numpy as np
import pywt

from load_data import load_mining_region_data


def wavelet_denoising(img, wavelet='db4', level=3):
    """
    使用小波变换对高光谱图像进行降噪（批量处理）

    Args:
        img: 高光谱图像数据，形状为(波段数, 高, 宽)
        wavelet: 小波基函数，默认为'db4'
        level: 分解层数

    Returns:
        降噪后的高光谱图像
    """

    if img.ndim == 3:
        # 获取图像尺寸
        bands, height, width = img.shape
        # Reshape图像，将每个像素点的光谱作为一个向量
        signals = img.reshape(bands, -1)
    elif img.ndim == 2:
        signals = img

    # 对所有像素点的光谱进行小波变换
    coeffs = pywt.wavedec(signals, wavelet, level=level, axis=0)

    # 设置阈值
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745

    # 对细节系数进行阈值处理
    for k in range(1, len(coeffs)):
        coeffs[k] = pywt.threshold(coeffs[k], threshold, mode='soft')

    # 重构信号
    denoised_img = pywt.waverec(coeffs, wavelet, axis=0)

    # Reshape回原来的形状
    if img.ndim == 3:
        denoised_img = denoised_img.reshape(bands, height, width)

    return denoised_img


def main():
    img_array, samples_spectral, zn_content, som_content, wavelengths = load_mining_region_data(need_wavelengths=True)
    samples_spectral = samples_spectral.T
    y = zn_content

    # # 光谱微分变换
    # samples_spectral = first_order_differential(samples_spectral, wavelengths, axis=1)
    # img_array = first_order_differential(img_array, wavelengths, axis=0)

    # reconstructed_image = wavelet_denoising(img_array, wavelet='db4', level=4)
    reconstructed_samples_spectral = wavelet_denoising(samples_spectral.T, wavelet='db4', level=4)

    # 提取原始和重构的光谱曲线
    # original_spectra = img_array[:, 1, 1]
    # reconstructed_spectra = reconstructed_image[:, 1, 1]
    original_spectra = samples_spectral.T[:, 0]
    reconstructed_spectra = reconstructed_samples_spectral[:, 0]
    plt.plot(original_spectra)
    plt.plot(reconstructed_spectra)
    plt.show()


if __name__ == '__main__':
    main()